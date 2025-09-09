import argparse
from pathlib import Path

import src.utils as utils

import torch
import torch.optim as optim
import wandb


from analysis.common import set_seed, get_model


def main():
    parser = argparse.ArgumentParser(description="Train per task and save model checkpoints (no Fisher analysis)")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    initial_args = parser.parse_args()

    try:
        args = utils.load_config_from_json(initial_args.config)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or "train-save-models",
            config=vars(args),
            dir=args.output_dir,
        )

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

    exp_name = f"{args.wandb_run_name}-fisher-only"
    exp_dir = out_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    all_task_ids = list(range(len(cl_train_loader)))

    for task_id in all_task_ids:
        model = get_model(args, channels, im_size, device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.use_wandb:
            try:
                wandb.define_metric(f"task/{task_id}/analysis_step")
                wandb.define_metric(f"task/{task_id}/*", step_metric=f"task/{task_id}/analysis_step")
            except Exception:
                pass

        for main_epoch in range(args.main_epochs):
            print(f"=== Main Epoch {main_epoch + 1}/{args.main_epochs} for Task {task_id} ===")
            train_loader = cl_train_loader[task_id]
            utils.train_one_task(
                model,
                train_loader,
                task_id,
                optimizer,
                ewc=None,
                gr=None,
                kl=False,
                save_path=exp_dir,
                num_epochs=args.epochs,
                device=device,
                wandb=wandb if args.use_wandb else None,
            )

            torch.save(model.state_dict(), exp_dir / f"model-task{task_id}_{main_epoch}.pt")

    print("Training done. Models saved to:", str(exp_dir))


if __name__ == "__main__":
    main()
