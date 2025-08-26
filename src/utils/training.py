"""
Training utilities for continual learning with diffusion models.

Provides functions for training single tasks and continual learning scenarios.
"""

import torch
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid


def train_one_task(model, train_loader, task_id, optimizer, 
                   ewc=None,
                   num_epochs=10, save_path=None, device='cuda'):
    """
    Train the model on a single task/class.
    
    Args:
        model: Diffusion model with diffusion_loss method and sample method
        train_loader: DataLoader for the current task
        task_id: Identifier for the current task (used for saving)
        optimizer: PyTorch optimizer
        ewc: EWC regularization object (optional)
        num_epochs (int): Number of training epochs
        save_path (str): Path to save sample images during training
        device (str): Device to train on
    """
    model.train()
    
    for epoch in tqdm(range(num_epochs), desc=f"Training task {task_id}"):
        for batch in train_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Compute diffusion loss
            loss = model.diffusion_loss(images, labels)
            
            # Add EWC regularization if provided
            if ewc is not None:
                loss_ewc = ewc.loss()
                loss = loss + 10000 * loss_ewc

            loss.backward()
            optimizer.step()

        # Generate and save sample images every 2 epochs
        if save_path is not None and epoch % 2 == 0:
            _save_epoch_samples(model, task_id, epoch, save_path, device)



def _save_epoch_samples(model, task_id, epoch, save_path, device):
    """
    Helper function to save sample images during training.
    
    Args:
        model: Diffusion model with sample method
        task_id: Current task identifier
        epoch (int): Current epoch number
        save_path (str): Base path for saving images
        device (str): Device for sampling
    """
    model.eval()
    
    # Create output directory
    out_dir = Path(save_path) / f"task_{task_id}" / f"epoch_{epoch:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample images for each class
    rows = model.num_class_labels
    cols = 8
    all_tensors = []
    
    with torch.no_grad():
        for c in range(rows):
            pils = model.sample(
                batch_size=cols,
                labels=[c] * cols,
                num_inference_steps=50,
                device=device,
                guidance_scale=0.0,  # pure conditional
            )
            for im in pils:
                # Convert from [-1,1] to [0,1] float
                all_tensors.append((im + 1.0) * 0.5)

        # Create and save grid
        grid = make_grid(torch.stack(all_tensors, dim=0), nrow=cols, padding=2)
        grid_pil = TF.to_pil_image(grid.clamp(0, 1))
        out_file = out_dir / f"epoch_{epoch:05d}_grid.png"
        grid_pil.save(out_file)
    
    model.train()
