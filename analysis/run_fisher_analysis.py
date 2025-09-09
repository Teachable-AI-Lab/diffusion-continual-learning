import argparse
from pathlib import Path

import torch
torch.backends.cuda.preferred_linalg_library("magma")
import wandb

import src.utils as utils
from src.experiment_runner import evaluate_fid

from analysis.common import set_seed, get_model, fisher_analysis


def main():
    parser = argparse.ArgumentParser(description="Run Fisher analysis on saved models")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment subfolder name inside output_dir where models are saved. Defaults to '<dataset>-fisher-only'",
    )
    parser.add_argument(
        "--epochs_per_task",
        type=int,
        default=None,
        help="Number of main epochs per task used during training; if None, inferred by scanning files.",
    )
    initial_args = parser.parse_args()

    try:
        args = utils.load_config_from_json(initial_args.config)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    exp_name = initial_args.exp_name or f"{args.wandb_run_name}-fisher-only"
    exp_dir = out_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or "run-fisher-analysis",
            config=vars(args),
            dir=args.output_dir,
        )
        try:
            wandb.define_metric("task_id")
            wandb.define_metric("fisher/*", step_metric="task_id")
            wandb.define_metric("eval/*", step_metric="task_id")
        except Exception:
            pass

    cl_train_loader, cl_test_loader, full_train_loader, full_test_loader = utils.get_cl_dataset(
        args.dataset,
        batch_size=args.batch_size,
        normalize=args.normalize,
        greyscale=args.greyscale,
        group_size=args.group_size,
        n_classes=args.num_classes,
    )
    im_size = full_train_loader.dataset[0][0].shape[1]
    channels = full_train_loader.dataset[0][0].shape[0]

    # Determine tasks and available checkpoints
    all_task_ids = list(range(len(cl_train_loader)))

    # Infer epochs per task if not provided
    epochs_per_task = initial_args.epochs_per_task
    if epochs_per_task is None:
        # Scan for files matching pattern model-task{tid}_{epoch}.pt for tid=0
        epochs = []
        for ep in range(0, 1000):  # reasonable upper bound
            if (exp_dir / f"model-task0_{ep}.pt").exists():
                epochs.append(ep)
            else:
                break
        if not epochs:
            raise FileNotFoundError(
                f"No checkpoints found in {exp_dir}. Expected files like model-task0_0.pt"
            )
        epochs_per_task = len(epochs)

    # Loop and run fisher on each saved checkpoint
    for task_id in all_task_ids:
        if args.use_wandb:
            try:
                wandb.define_metric(f"task/{task_id}/analysis_step")
                wandb.define_metric(f"task/{task_id}/*", step_metric=f"task/{task_id}/analysis_step")
            except Exception:
                pass

        for main_epoch in range(epochs_per_task):
            ckpt_path = exp_dir / f"model-task{task_id}_{main_epoch}.pt"
            if not ckpt_path.exists():
                print(f"[WARN] Missing checkpoint {ckpt_path}, skipping")
                continue

            # Recreate model with correct shape and load weights
            model = get_model(args, channels, im_size, device)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.eval()

            # Fisher analysis
            train_loader = cl_train_loader[task_id]
            fisher_analysis(
                model,
                args,
                train_loader,
                device,
                task_id,
                main_epoch=main_epoch,
                exp_dir=exp_dir,
            )

            # Optional FID evaluation to mimic original script
            fid = None
            try:
                fid = evaluate_fid(model, cl_test_loader[task_id], device)
            except Exception:
                pass
            if args.use_wandb and fid is not None:
                prefix = f"task/{task_id}"
                wandb.log({f"{prefix}/analysis_step": main_epoch, f"{prefix}/eval/fid": fid})

    print("Fisher analysis done. JSON results stored in:", str(exp_dir))


if __name__ == "__main__":
    main()
