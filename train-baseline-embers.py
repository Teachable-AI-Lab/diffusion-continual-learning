import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import random
import argparse
from src.experiment_runner import *
import os
import wandb

import src.utils as utils
from src.ddim import build_conditional_ddim

def train_one_task(model, train_loader, class_id, optimizer,
                   num_epochs=10, save_path=None, device='cuda', wandb=None):
    unique_labels = set()
    for epoch in tqdm(range(num_epochs)):
        for batch in tqdm(train_loader):
            images, labels = batch
            unique_labels.update(labels.tolist())
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = 0
            timesteps, noise, noisy_images, model_pred = model.diffusion_loss(images, labels)
            ddim_loss = F.mse_loss(model_pred, noise, reduction="mean")

            loss = loss + ddim_loss
            loss.backward()
            optimizer.step()

        if wandb is not None:
            wandb.log({
                'loss/total': loss.item(),
                'epoch': epoch,
            })

        if save_path is not None and epoch % 10 == 0:
            out_dir = Path(save_path) / f"task_{class_id}" / f"epoch_{epoch:05d}"
            out_dir.mkdir(parents=True, exist_ok=True)


def train_all_classes(model, train_loader, optimizer,
                      num_epochs=10, save_path=None, device='cuda', wandb=None):
    """
    Train a diffusion model on the full dataset (all classes, no tasks) with resume support.

    Behavior:
    - Saves a rolling 'latest' checkpoint every 10 epochs (overwrites previous).
    - Saves archival checkpoints at fixed intervals defined as:
        interval = min(num_epochs // 2, 200)
      If interval == 0 (very short runs), only the final checkpoint is archived.
    - Can resume from a previous run if a 'latest.pt' checkpoint exists in save_path.

    Args:
        model: PyTorch module with a method `diffusion_loss(images, labels)`
        train_loader: DataLoader yielding (images, labels)
        optimizer: torch.optim.Optimizer
        num_epochs: total training epochs
        save_path: directory to save checkpoints. If None, no checkpoints will be saved.
        device: device string (e.g., 'cuda' or 'cpu')
        wandb: optional Weights & Biases run object for logging
    """
    model.to(device)

    # Setup checkpoint paths
    latest_ckpt_path = None
    arch_dir = None
    if save_path is not None:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        latest_ckpt_path = save_dir / 'latest.pt'
        arch_dir = save_dir / 'checkpoints'
        arch_dir.mkdir(parents=True, exist_ok=True)

    # Try resume
    start_epoch = 0
    if latest_ckpt_path is not None and latest_ckpt_path.exists():
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1  # resume from next epoch

    # Archival interval logic
    arch_interval = min(max(num_epochs // 2, 0), 200)

    for epoch in tqdm(range(start_epoch, num_epochs), desc='Epoch'):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for batch in tqdm(train_loader, leave=False, desc='Batch'):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Diffusion loss
            timesteps, noise, noisy_images, model_pred = model.diffusion_loss(images, labels)
            ddim_loss = F.mse_loss(model_pred, noise, reduction="mean")

            loss = ddim_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)

        if wandb is not None:
            wandb.log({
                'loss/total': avg_loss,
                'epoch': epoch,
            })

        # Save rolling latest every 10 epochs (and also at final epoch)
        if latest_ckpt_path is not None and ((epoch % 10 == 0) or (epoch == num_epochs - 1)):
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, latest_ckpt_path)

        # Save archival checkpoint at fixed interval boundaries and at final epoch
        if arch_dir is not None:
            should_archive = False
            if arch_interval > 0:
                should_archive = ((epoch + 1) % arch_interval == 0)
            # Always ensure final epoch is archived
            if epoch == num_epochs - 1:
                should_archive = True
            if should_archive:
                arch_path = arch_dir / f"epoch_{epoch+1:05d}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }, arch_path)

    return


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline training on all classes (no CL)")
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    initial_args = parser.parse_args()

    args = utils.load_config_from_json(initial_args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Derive experiment path first
    exp_path = root / f"{args.dataset}-baseline-{args.seed}"
    exp_path.mkdir(parents=True, exist_ok=True)

    # WandB (with simple resume using a per-experiment persisted run id)
    run = None
    if getattr(args, 'use_wandb', False):
        run_id_file = exp_path / '.wandb_run_id'
        if run_id_file.exists():
            run_id = run_id_file.read_text().strip()
        else:
            # prefer provided name, else derive a short id
            run_id = getattr(args, 'wandb_run_name', None) or f"baseline-{args.dataset}-{args.seed}"
            run_id_file.write_text(run_id)
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=run_id,
            resume="allow",
            dir=str(exp_path),
            config=vars(args),
        )

    # Data
    cl_train_loader, cl_test_loader, full_train_loader, full_test_loader = utils.get_cl_dataset(
        args.dataset,
        batch_size=args.batch_size,
        normalize=args.normalize,
        greyscale=args.greyscale,
        group_size=args.num_classes,
        n_classes=args.num_classes,
    )
    print(f"Loaded dataset {args.dataset} with {args.num_classes} classes.")
    sample_img, _ = full_train_loader.dataset[0]
    channels = sample_img.shape[0]
    im_size = sample_img.shape[1]

    # Model
    model = build_conditional_ddim(
        in_channel=channels,
        image_size=im_size,
        num_class_labels=args.num_classes
    ).to(device)
    print("Model built.")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_all_classes(
        model,
        cl_train_loader[0],
        optimizer,
        num_epochs=args.epochs,
        save_path=str(exp_path),
        device=str(device),
        wandb=run,
    )
    print("Training complete.")

    # Save final model (optional; latest and archive already saved during training)
    final_path = exp_path / 'final_model.pt'
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path)

    ## Run sampling for qualitative results
    model.eval()
    fid = evaluate_fid(model, cl_test_loader[0], device)
    print(f"Final FID across all tasks: {fid}")
    # TODO: log to wandb
    if args.use_wandb:
        wandb.log({
            f"fid_eval":1,
            f"final-fid": fid,
        })