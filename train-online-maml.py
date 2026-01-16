"""Train Conditional DDIM with first-order Online MAML.

This script follows Finn et al. (2019), "Online Meta-Learning", adapting the
existing continual-learning setup to a meta-learning baseline.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Set

import numpy as np
import torch
import torch.optim as optim

import wandb

from src.ddim import build_conditional_ddim
from src.gr import GenerativeReplay
import src.utils as utils
from src.online_maml_utils import (
    Batch,
    clone_model_for_adaptation,
    meta_update,
    meta_update_functional,
    run_functional_inner_loop,
    run_inner_loop,
    support_query_batches,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_seen_tasks(
    model: torch.nn.Module,
    test_loaders: Dict[int, torch.utils.data.DataLoader],
    upto_task: int,
    fid_evaluator: utils.FIDEvaluator,
    num_inference_steps: int,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for eval_task in range(upto_task + 1):
        loader = test_loaders[eval_task]
        fid = fid_evaluator.fid_loader_vs_model(
            loader,
            model,
            num_inference_steps=num_inference_steps,
        )
        scores[eval_task] = fid
    return scores


def dump_metrics(metrics_path: Path, history: List[dict]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def augment_batches_with_replay(
    batches: Sequence[Batch],
    device: torch.device,
    gr: GenerativeReplay | None,
    unique_labels: Set[int],
) -> List[Batch]:
    prepared: List[Batch] = []
    for images, labels in batches:
        unique_labels.update(labels.tolist())
        images = images.to(device)
        labels = labels.to(device)
        if gr is not None:
            x_old, y_old = gr.replay()
            images = torch.cat([images, x_old.to(device)], dim=0)
            labels = torch.cat([labels, y_old.to(device)], dim=0)
        prepared.append((images, labels))
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Online MAML training entrypoint")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    cli_args = parser.parse_args()

    args = utils.load_config_from_json(cli_args.config)

    use_wandb = getattr(args, "use_wandb", False)
    if use_wandb:
        wandb.init(
            project=getattr(args, "wandb_project", "diffusion-continual-learning"),
            name=getattr(args, "wandb_run_name", "online-maml"),
            config=vars(args),
            dir=getattr(args, "output_dir", "./outputs"),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(getattr(args, "seed", 123))

    dataset_name = getattr(args, "dataset", "mnist")
    batch_size = getattr(args, "batch_size", 128)
    normalize = getattr(args, "normalize", True)
    greyscale = getattr(args, "greyscale", False)
    group_size = getattr(args, "group_size", 2)
    num_classes = getattr(args, "num_classes", 10)

    root = Path(getattr(args, "output_dir", "./outputs"))
    run_name = getattr(args, "wandb_run_name", f"omaml-{dataset_name}-{getattr(args, 'seed', 123)}")
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    cl_train_loaders, cl_test_loaders, full_train_loader, _ = utils.get_cl_dataset(
        dataset_name,
        batch_size=batch_size,
        normalize=normalize,
        greyscale=greyscale,
        group_size=group_size,
        n_classes=num_classes,
    )
    channels = full_train_loader.dataset[0][0].shape[0]
    image_size = full_train_loader.dataset[0][0].shape[1]

    print("Building model...")
    model = build_conditional_ddim(
        in_channel=channels,
        image_size=image_size,
        num_class_labels=num_classes,
        ewc_lambda=0.0,
        gr_kl=0.0,
    ).to(device)
    outer_optimizer = optim.Adam(model.parameters(), lr=getattr(args, "lr", 2e-4))

    inner_lr = getattr(args, "omaml_inner_lr", 1e-4)
    support_batches = getattr(args, "omaml_support_batches", 1)
    query_batches = getattr(args, "omaml_query_batches", 1)
    task_epochs = getattr(args, "omaml_epochs_per_task", 100)
    max_pairs_per_epoch = getattr(args, "omaml_pairs_per_epoch", None)
    inner_grad_clip = getattr(args, "omaml_inner_grad_clip", None)
    outer_grad_clip = getattr(args, "omaml_outer_grad_clip", None)
    fid_steps = getattr(args, "fid_num_inference_steps", 50)
    use_second_order = getattr(args, "omaml_second_order", False)
    use_generative_replay = getattr(args, "use_generative_replay", True)
    gr_alpha = getattr(args, "gr_alpha", 0.5)
    gr_pool_size = getattr(args, "gr_pool_size_per_class", 1000)
    gr_inference_steps = getattr(args, "gr_num_inference_steps", 50)
    gr_eta = getattr(args, "gr_eta", 0.0)

    fid_evaluator = utils.FIDEvaluator(device=device)
    history: List[dict] = []
    gr: GenerativeReplay | None = None
    meta_ckpt_dir = run_dir / "meta_updates"

    for task_id in sorted(cl_train_loaders.keys()):
        print(f"=== Task {task_id} / {len(cl_train_loaders)} ===")
        task_loader = cl_train_loaders[task_id]
        task_meta_losses: List[float] = []
        unique_labels: Set[int] = set()
        global_meta_updates = 0
        # if utils._should_save_step(global_meta_updates):
        #     utils._save_step_checkpoint(
        #         model,
        #         meta_ckpt_dir,
        #         task_id,
        #         global_meta_updates,
        #         unique_labels,
        #         device,
        #         wandb if use_wandb else None,
        #     )
        task_gr = gr if use_generative_replay else None
        for epoch in range(task_epochs):
            print(f"Task {task_id} epoch {epoch + 1}/{task_epochs}")
            pair_iter = support_query_batches(
                task_loader,
                support_batches=support_batches,
                query_batches=query_batches,
                max_pairs=max_pairs_per_epoch,
            )
            for support_list, query_list in pair_iter:
                support_batches_ready = augment_batches_with_replay(
                    support_list,
                    device,
                    task_gr,
                    unique_labels,
                )
                query_batches_ready = augment_batches_with_replay(
                    query_list,
                    device,
                    task_gr,
                    unique_labels,
                )

                if use_second_order:
                    fast_params, buffers, _ = run_functional_inner_loop(
                        model,
                        support_batches_ready,
                        inner_lr=inner_lr,
                        use_second_order=True,
                    )
                    meta_loss = meta_update_functional(
                        model,
                        fast_params,
                        buffers,
                        query_batches_ready,
                        outer_optimizer=outer_optimizer,
                        grad_clip=outer_grad_clip,
                    )
                else:
                    fast_model = clone_model_for_adaptation(model, device)
                    run_inner_loop(
                        fast_model,
                        support_batches_ready,
                        inner_lr=inner_lr,
                        device=device,
                        grad_clip=inner_grad_clip,
                    )
                    meta_loss = meta_update(
                        model,
                        fast_model,
                        query_batches_ready,
                        outer_optimizer=outer_optimizer,
                        device=device,
                        grad_clip=outer_grad_clip,
                    )
                if meta_loss is None:
                    continue
                task_meta_losses.append(meta_loss)
                global_meta_updates += 1
                if utils._should_save_step(global_meta_updates):
                    utils._save_step_checkpoint(
                        model,
                        meta_ckpt_dir,
                        task_id,
                        global_meta_updates,
                        unique_labels,
                        device,
                        wandb if use_wandb else None,
                    )
                if use_wandb:
                    wandb.log(
                        {
                            "task_id": task_id,
                            "meta_loss": meta_loss,
                            "global_meta_updates": global_meta_updates,
                        }
                    )
        checkpoint_path = run_dir / f"task{task_id}_model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        fid_scores = evaluate_seen_tasks(
            model,
            cl_test_loaders,
            task_id,
            fid_evaluator,
            num_inference_steps=fid_steps,
        )
        avg_fid = float(np.mean(list(fid_scores.values())))
        history.append(
            {
                "task_id": task_id,
                "meta_updates": len(task_meta_losses),
                "avg_meta_loss": float(np.mean(task_meta_losses)) if task_meta_losses else None,
                "fid_per_task": fid_scores,
                "avg_fid": avg_fid,
            }
        )
        dump_metrics(run_dir / "metrics.json", history)
        if use_wandb:
            log_payload = {"task_id": task_id, "avg_fid": avg_fid}
            for eval_task, fid in fid_scores.items():
                log_payload[f"fid/task_{eval_task}"] = fid
            wandb.log(log_payload)

        last_task = task_id == max(cl_train_loaders.keys())
        if use_generative_replay and not last_task:
            frozen_teacher = utils.freeze_model(model)
            old_classes = list(range((task_id + 1) * group_size))
            if gr is None:
                gr = GenerativeReplay(
                    frozen_teacher,
                    old_classes=old_classes,
                    batch_size=batch_size,
                    alpha=gr_alpha,
                    pool_size_per_class=gr_pool_size,
                    num_inference_steps=gr_inference_steps,
                    eta=gr_eta,
                    seed=getattr(args, "seed", 123),
                    device=device,
                )
            else:
                gr.update_teacher(frozen_teacher, old_classes=old_classes)

    print("Training complete. Metrics saved to", run_dir / "metrics.json")


if __name__ == "__main__":
    main()
