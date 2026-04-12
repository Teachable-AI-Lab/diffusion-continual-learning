"""Train Conditional DDIM with VR-MCL style meta-learning and generative replay.

This script adapts the VR-MCL algorithm to the diffusion continual-learning
setup used by the online MAML baseline. It keeps per-parameter inner learning
rates, maintains an EMA-adapted fast weight branch, and mixes old-task samples
through generative replay instead of experience replay.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm import tqdm

from src.ddim import build_conditional_ddim
from src.gr import GenerativeReplay
import src.utils as utils
from src.online_maml_utils import (
    Batch,
    support_query_batches,
    _prepare_diffusion_inputs,
    _stateless_unet_forward,
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
    seen_task_ids: Sequence[int],
    fid_evaluator: utils.FIDEvaluator,
    num_inference_steps: int,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for eval_task in seen_task_ids:
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


def _ddim_reconstruction_loss_functional(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    params: OrderedDict[str, torch.Tensor],
    buffers: OrderedDict[str, torch.Tensor],
    prepared: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if prepared is None:
        prepared = _prepare_diffusion_inputs(model, images, labels)
    noise, timesteps, noisy_images, class_labels = prepared
    model_pred = _stateless_unet_forward(model, params, buffers, noisy_images, timesteps, class_labels)
    return F.mse_loss(model_pred, noise, reduction="mean")


def _stack_batches(batches: Sequence[Batch], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.cat([b[0] for b in batches], dim=0).to(device)
    labels = torch.cat([b[1] for b in batches], dim=0).to(device)
    return images, labels


class VRMCLTrainer:
    """VR-MCL trainer (core logic mirroring vrmcl.py).

    This implements the same *structural* update used in /mnt/data/vrmcl.py:
      - Maintain two parameter tracks for meta-gradients:
          (a) main track (updated with standard grads),
          (b) an "EMA/last-step" track (used as the control variate).
      - Run inner-loop adaptation for BOTH tracks with shared per-parameter step sizes (task_lr).
      - After EACH inner step j, evaluate query loss and append to meta_loss lists.
      - Outer step:
          * backprop avg(meta_loss) into main params (+ task_lr)
          * backprop avg(meta_loss_last_step) into ema params (+ task_lr)
          * update task_lr with SGD
          * update main params either via Adam (default) or STORM/VR (asyn_update=True)

    Notes:
      - In the original code, the EMA network is owned by the model object.
        Here we keep an explicit EMA parameter buffer (self.ema_params) and
        refresh it with exponential moving average after each outer update.
      - The inner-loop LR tensor can easily blow up diffusion losses if
        alpha_initial is large (e.g., 0.15). Consider 1e-3..1e-2 for diffusion.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        outer_lr: float = 2e-4,
        alpha_initial: float = 0.15,
        second_order: bool = False,
        inner_batch_size: int = 32,
        meta_updates_per_batch: int = 5,
        grad_clip_norm: float | None = 2.0,
        ema_decay: float = 0.9,
        # STORM / VR (matches vrmcl.py when asyn_update=True)
        asyn_update: bool = False,
        storm_momentum: float = 0.8,
        storm_lr: float = 0.1,
        optim_wd: float = 0.0,
        optim_mom: float = 0.9,
    ) -> None:
        self.model = model
        self.device = device
        self.second_order = second_order
        self.inner_batch_size = inner_batch_size
        self.meta_updates_per_batch = meta_updates_per_batch
        self.grad_clip_norm = grad_clip_norm
        self.ema_decay = ema_decay
        self.asyn_update = asyn_update
        self.storm_momentum = storm_momentum
        self.storm_lr = storm_lr

        self.task_lr: OrderedDict[str, torch.nn.Parameter] = OrderedDict()
        for name, param in model.unet.named_parameters():
            lr_tensor = torch.ones_like(param, requires_grad=True, device=device) * alpha_initial
            self.task_lr[name + "_lr"] = torch.nn.Parameter(lr_tensor)

        # Outer optimizer is only used when asyn_update=False.
        self.outer_optimizer = optim.Adam(
            model.unet.parameters(),
            lr=outer_lr,
            weight_decay=optim_wd,
            betas=(optim_mom, 0.999),
        )
        self.lr_optimizer = optim.SGD(self.task_lr.values(), lr=outer_lr)

        # EMA parameter buffer (control variate branch)
        self.ema_params: OrderedDict[str, torch.Tensor] | None = None

        # STORM momentum buffer per parameter (only used if asyn_update=True)
        self._storm_momentum_buf: List[torch.Tensor | None] = []

    def begin_task(self) -> None:
        self.outer_optimizer.zero_grad(set_to_none=True)
        self.lr_optimizer.zero_grad(set_to_none=True)

        # initialize EMA buffer from current params at task start
        self.ema_params = OrderedDict((name, p.detach().clone()) for name, p in self.model.unet.named_parameters())

        # reset STORM momentum (per param)
        self._storm_momentum_buf = [None for _ in self.model.unet.parameters()]

    def _inner_step(
        self,
        fast_params: OrderedDict[str, torch.Tensor],
        buffers: OrderedDict[str, torch.Tensor],
        images: torch.Tensor,
        labels: torch.Tensor,
        prepared: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[OrderedDict[str, torch.Tensor], torch.Tensor]:
        """One inner-loop step: fast <- fast - task_lr * grad(loss).

        Returns (updated_fast_params, loss_tensor).
        """
        loss = _ddim_reconstruction_loss_functional(self.model, images, labels, fast_params, buffers, prepared)
        grads = torch.autograd.grad(
            loss,
            tuple(fast_params.values()),
            create_graph=self.second_order,
            retain_graph=self.second_order,
        )
        if self.grad_clip_norm is not None:
            grads = [torch.clamp(g, min=-self.grad_clip_norm, max=self.grad_clip_norm) for g in grads]

        updated: OrderedDict[str, torch.Tensor] = OrderedDict()
        for (name, param), grad in zip(fast_params.items(), grads):
            # NOTE: vrmcl.py uses raw task_lr (no positivity constraint).
            lr = self.task_lr[name + "_lr"]
            updated[name] = param - lr * grad
        return updated, loss

    def _update_ema_params(self) -> None:
        """EMA buffer update: ema <- decay * ema + (1-decay) * main."""
        if self.ema_params is None:
            self.ema_params = OrderedDict((n, p.detach().clone()) for n, p in self.model.unet.named_parameters())
            return
        with torch.no_grad():
            for name, p in self.model.unet.named_parameters():
                self.ema_params[name].mul_(self.ema_decay).add_(p.detach(), alpha=(1.0 - self.ema_decay))

    def _storm_step(self, ema_leaf_params: OrderedDict[str, torch.Tensor]) -> None:
        """STORM/VR outer step (mirrors vrmcl.py asyn_update branch).

        Updates *main* params using main grads and EMA(control-variate) grads:
            m <- g_main + beta * (m - g_ema)
            theta <- theta - s_lr * m
        """
        if not self._storm_momentum_buf:
            self._storm_momentum_buf = [None for _ in self.model.unet.parameters()]

        with torch.no_grad():
            for idx, (p_main, (_, p_ema_leaf)) in enumerate(zip(self.model.unet.parameters(), ema_leaf_params.items())):
                g_main = p_main.grad
                g_ema = p_ema_leaf.grad
                if g_main is None or g_ema is None:
                    continue
                if self._storm_momentum_buf[idx] is None:
                    p_main.data.add_(g_main, alpha=-self.storm_lr)
                    self._storm_momentum_buf[idx] = g_main.detach().clone()
                else:
                    m_prev = self._storm_momentum_buf[idx]
                    m = g_main + self.storm_momentum * (m_prev - g_ema)
                    p_main.data.add_(m, alpha=-self.storm_lr)
                    self._storm_momentum_buf[idx] = m.detach().clone()

    def observe(
        self,
        support_batches: Sequence[Batch],
        query_batches: Sequence[Batch],
        replay_active: bool = False,
    ) -> float:
        if not support_batches or not query_batches:
            return 0.0

        buffers = OrderedDict(self.model.unet.named_buffers())
        support_images, support_labels = _stack_batches(support_batches, self.device)
        query_images, query_labels = _stack_batches(query_batches, self.device)

        # Pre-sample diffusion noise/timesteps for query so BOTH branches see identical epsilon_b
        prepared_query = _prepare_diffusion_inputs(self.model, query_images, query_labels)

        total_meta_loss = 0.0

        real_batch_size = support_images.size(0)
        # print(f"VRMCLTrainer: observe called with support batch size {real_batch_size}, replay_active={replay_active}")
        if replay_active:
            # When replay enlarges batches, force a single inner step to keep adaptation cost stable.
            num_inner_steps = 1
        else:
            num_inner_steps = math.ceil(real_batch_size / self.inner_batch_size)

        # print(f"  Running {self.meta_updates_per_batch} meta-updates with {num_inner_steps} inner steps each.")

        for _ in range(self.meta_updates_per_batch):
            # -------------------------
            # Sample support permutation
            # -------------------------
            perm = torch.randperm(real_batch_size, device=self.device)
            x_s = support_images[perm]
            y_s = support_labels[perm]

            # -------------------------
            # Initialize fast weights
            # -------------------------
            # Main fast weights start from current main params (as in vrmcl.py)
            fast_main: OrderedDict[str, torch.Tensor] = OrderedDict(
                (name, p) for name, p in self.model.unet.named_parameters()
            )

            # "Last-step/EMA" fast weights start from an EMA buffer (control variate)
            if self.ema_params is None:
                self.ema_params = OrderedDict((name, p.detach().clone()) for name, p in self.model.unet.named_parameters())
            ema_leaf: OrderedDict[str, torch.Tensor] = OrderedDict(
                (name, p.detach().clone().requires_grad_(True)) for name, p in self.ema_params.items()
            )
            fast_ema: OrderedDict[str, torch.Tensor] = ema_leaf

            # -------------------------
            # Inner loop + meta-loss accumulation across inner steps
            # -------------------------
            meta_losses: List[torch.Tensor] = []
            meta_losses_ema: List[torch.Tensor] = []

            for j in range(num_inner_steps):
                start = j * self.inner_batch_size
                end = min((j + 1) * self.inner_batch_size, real_batch_size)
                x_inner = x_s[start:end].detach()
                y_inner = y_s[start:end].detach()

                prepared_inner = _prepare_diffusion_inputs(self.model, x_inner, y_inner)

                # inner updates (main + ema/control-variate)
                fast_main, _ = self._inner_step(fast_main, buffers, x_inner, y_inner, prepared_inner)
                fast_ema, _ = self._inner_step(fast_ema, buffers, x_inner, y_inner, prepared_inner)

                # query loss AFTER this inner step (this mirrors vrmcl.py: meta_loss_compute(..., j))
                q_main = _ddim_reconstruction_loss_functional(
                    self.model, query_images, query_labels, fast_main, buffers, prepared_query
                )
                q_ema = _ddim_reconstruction_loss_functional(
                    self.model, query_images, query_labels, fast_ema, buffers, prepared_query
                )
                meta_losses.append(q_main)
                meta_losses_ema.append(q_ema)

            # -------------------------
            # Outer loop (mirrors vrmcl.py)
            # -------------------------
            meta_loss = sum(meta_losses) / len(meta_losses)
            meta_loss_ema = sum(meta_losses_ema) / len(meta_losses_ema)

            # clear grads
            self.outer_optimizer.zero_grad(set_to_none=True)
            self.lr_optimizer.zero_grad(set_to_none=True)
            self.model.unet.zero_grad(set_to_none=True)

            # backprop main branch then ema branch (task_lr accumulates grads from both)
            meta_loss.backward(retain_graph=False)
            meta_loss_ema.backward(retain_graph=False)

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), self.grad_clip_norm)
                torch.nn.utils.clip_grad_norm_(list(ema_leaf.values()), self.grad_clip_norm)
                torch.nn.utils.clip_grad_norm_(self.task_lr.values(), self.grad_clip_norm)

            # update task_lr (opt_lr.step in vrmcl.py)
            self.lr_optimizer.step()

            # update outer parameters
            if self.asyn_update:
                # STORM/VR update uses grads from main vs control variate grads
                self._storm_step(ema_leaf)
            else:
                # vanilla outer optimizer step
                self.outer_optimizer.step()

            # refresh EMA buffer from updated main params
            self._update_ema_params()

            total_meta_loss += float(meta_loss.detach().item())

        # cleanup grads between observe calls
        self.outer_optimizer.zero_grad(set_to_none=True)
        self.lr_optimizer.zero_grad(set_to_none=True)
        self.model.unet.zero_grad(set_to_none=True)

        return total_meta_loss / float(self.meta_updates_per_batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="VR-MCL training entrypoint")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    cli_args = parser.parse_args()

    args = utils.load_config_from_json(cli_args.config)

    use_wandb = getattr(args, "use_wandb", False)
    print("Configuration loaded successfully:")
    print("-" * 30)
    for key, value in sorted(vars(args).items()):
        print(f"{key} ({type(value).__name__}): {value}")
    print("-" * 30)

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            dir=args.output_dir,
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
    run_name = getattr(args, "wandb_run_name", f"vrmcl-{dataset_name}-{getattr(args, 'seed', 123)}")
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

    outer_lr = getattr(args, "lr", 2e-4)
    alpha_initial = getattr(args, "vrmcl_alpha_initial", 1e-4)
    second_order = getattr(args, "vrmcl_second_order", False)
    inner_batch_size = getattr(args, "vrmcl_inner_batch_size", 32)
    meta_updates_per_batch = getattr(args, "vrmcl_meta_updates_per_batch", 5)
    grad_clip_norm = getattr(args, "vrmcl_grad_clip_norm", 2.0)
    ema_decay = getattr(args, "vrmcl_ema_decay", 0.9)
    # STORM / VR update (mirrors vrmcl.py when enabled)
    asyn_update = getattr(args, "vrmcl_asyn_update", False)
    storm_momentum = getattr(args, "vrmcl_storm_momentum", 0.8)
    storm_lr = getattr(args, "vrmcl_storm_lr", 1e-4)
    optim_wd = getattr(args, "vrmcl_optim_wd", 0.0)
    optim_mom = getattr(args, "vrmcl_optim_mom", 0.9)
    # Total meta-updates per task (cycles the loader as many times as needed)
    target_meta_updates = getattr(args, "vrmcl_pairs_per_epoch", None)
    support_batches = getattr(args, "vrmcl_support_batches", getattr(args, "omaml_support_batches", 1))
    query_batches = getattr(args, "vrmcl_query_batches", getattr(args, "omaml_query_batches", 1))

    fid_steps = getattr(args, "fid_num_inference_steps", 50)
    use_generative_replay = getattr(args, "use_generative_replay", True)
    gr_alpha = getattr(args, "gr_alpha", 0.5)
    gr_pool_size = getattr(args, "gr_pool_size_per_class", 1000)
    gr_inference_steps = getattr(args, "gr_num_inference_steps", 50)
    gr_eta = getattr(args, "gr_eta", 0.0)

    vrmcl_trainer = VRMCLTrainer(
        model=model,
        device=device,
        outer_lr=outer_lr,
        alpha_initial=alpha_initial,
        second_order=second_order,
        inner_batch_size=inner_batch_size,
        meta_updates_per_batch=meta_updates_per_batch,
        grad_clip_norm=grad_clip_norm,
        ema_decay=ema_decay,
        asyn_update=asyn_update,
        storm_momentum=storm_momentum,
        storm_lr=storm_lr,
        optim_wd=optim_wd,
        optim_mom=optim_mom,
    )

    fid_evaluator = utils.FIDEvaluator(device=device)
    history: List[dict] = []
    gr: GenerativeReplay | None = None
    meta_ckpt_dir = run_dir / "meta_updates"

    all_task_ids = sorted(cl_train_loaders.keys())
    shuffled_task_ids = all_task_ids.copy()
    if getattr(args, "randomize_task_order", False):
        random.shuffle(shuffled_task_ids)
        print("Randomized task order is enabled.")
    print("Task order (train_step -> original_task):", shuffled_task_ids)

    for train_step, task_id in enumerate(shuffled_task_ids):
        print(f"=== Task {task_id} (train_step {train_step + 1}/{len(shuffled_task_ids)}) ===")
        task_loader = cl_train_loaders[task_id]

        vrmcl_trainer.begin_task()
        task_meta_losses: List[float] = []
        unique_labels: Set[int] = set()
        global_meta_updates = 0

        task_gr = gr if use_generative_replay else None
        total_updates = target_meta_updates
        # print(f"Task {task_id}: running {total_updates if total_updates is not None else 'unbounded'} meta-updates")
        pbar = tqdm(total=total_updates, desc=f"Task {task_id} meta-updates", leave=False)

        while total_updates is None or global_meta_updates < total_updates:
            pair_iter = support_query_batches(
                task_loader,
                support_batches=support_batches,
                query_batches=query_batches,
                max_pairs=None,
            )
            for support_list, query_list in pair_iter:
                # Match vrmcl.py: inner-loop support comes from current data;
                # replay is mixed into the *query* (outer-loop) batch.
                support_ready: List[Batch] = []
                for images, labels in support_list:
                    unique_labels.update(labels.tolist())
                    support_ready.append((images.to(device), labels.to(device)))

                query_ready = augment_batches_with_replay(query_list, device, task_gr, unique_labels)
                # print(f"Meta-update {global_meta_updates + 1}: support batch size {support_ready[0][0].size(0)}, "
                    #   f"query batch size {query_ready[0][0].size(0)} (after replay)")
                meta_loss = vrmcl_trainer.observe(
                    support_ready,
                    query_ready,
                    replay_active=task_gr is not None,
                )
                task_meta_losses.append(meta_loss)
                global_meta_updates += 1
                if pbar is not None:
                    pbar.update(1)

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

                if total_updates is not None and global_meta_updates >= total_updates:
                    break

            if total_updates is not None and global_meta_updates >= total_updates:
                break

        if pbar is not None:
            pbar.close()

        checkpoint_path = run_dir / f"task{task_id}_model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        seen_task_ids = shuffled_task_ids[: train_step + 1]
        fid_scores = evaluate_seen_tasks(
            model,
            cl_test_loaders,
            seen_task_ids=seen_task_ids,
            fid_evaluator=fid_evaluator,
            num_inference_steps=fid_steps,
        )
        avg_fid = float(np.mean(list(fid_scores.values())))
        for eval_task, fid in fid_scores.items():
            print(f"FID for task {eval_task}: {fid:.4f}")
        print(f"Average FID up to task {task_id}: {avg_fid:.4f}")

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

        last_task = train_step == len(shuffled_task_ids) - 1
        if use_generative_replay and not last_task:
            frozen_teacher = utils.freeze_model(model)
            old_classes = sorted(
                {
                    class_id
                    for seen_task in seen_task_ids
                    for class_id in range(seen_task * group_size, (seen_task + 1) * group_size)
                }
            )
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
