"""
Evaluation utilities for diffusion models.

Provides FID (Frechet Inception Distance) evaluation and other metrics.
"""

import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import random


class FIDEvaluator:
    """
    Frechet Inception Distance evaluator for comparing real and generated images.
    """
    
    def __init__(self, device=None):
        """
        Initialize the FID evaluator.
        
        Args:
            device: PyTorch device (defaults to CUDA if available, else CPU)
        """
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # normalize=True => inputs should be float in [0,1]
        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _to01_and_rgb(x: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to [0,1] range and ensure 3 channels.
        
        Args:
            x: Input tensor (B,C,H,W) in [0,1] or [-1,1] range
            
        Returns:
            Tensor in [0,1] range with 3 channels as uint8
        """
        if x.dtype.is_floating_point:
            if x.min() < 0.0:  # convert [-1,1] -> [0,1]
                x = (x + 1.0) * 0.5
        else:
            x = x.float() / 255.0
            
        # Ensure 3 channels (convert grayscale to RGB if needed)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        return x.clamp(0.0, 1.0).to(torch.uint8)

    @torch.no_grad()
    def fid_loader_vs_model(
        self,
        real_loader,
        model,
        num_inference_steps: int = 50,
        seed: int | None = 123,
        max_real: int | None = None,  # optional cap on #real images processed
    ) -> float:
        """
        Compute FID between real images from a DataLoader and model-generated images.
        
        Args:
            real_loader: DataLoader yielding (images, labels) pairs
            model: Generative model with sample() method
            num_inference_steps (int): Number of denoising steps for generation
            seed (int | None): Random seed for reproducible generation
            max_real (int | None): Maximum number of real images to process
            
        Returns:
            float: FID score (lower is better)
        """
        self.fid.reset()
        dev = self.device
        seed_base = seed if seed is not None else 0
        batch_idx = 0
        seen = 0
        
        for imgs_real, labels_real in real_loader:
            # Check if we've processed enough images
            if max_real is not None and seen >= max_real:
                break
            if max_real is not None and seen + imgs_real.size(0) > max_real:
                keep = max_real - seen
                imgs_real = imgs_real[:keep]
                labels_real = labels_real[:keep]

            # Process real images
            imgs_real = imgs_real.to(dev)
            # Convert from [-1,1] to [0,255] uint8
            imgs_real = (imgs_real + 1.0) * 127.5
            imgs_real = imgs_real.to(torch.uint8)
            
            # Ensure 3 channels for FID computation
            if imgs_real.size(1) == 1:
                imgs_real = imgs_real.repeat(1, 3, 1, 1)
                
            self.fid.update(imgs_real, real=True)

            # Generate fake images with matching labels
            seed_b = seed_base + batch_idx if seed is not None else None
            imgs_fake = model.sample(
                batch_size=labels_real.numel(),
                labels=labels_real.tolist(),  # Match labels exactly
                num_inference_steps=num_inference_steps,
                device=dev,
                seed=seed_b,
            )
            
            # Convert fake images to same format as real images
            imgs_fake = ((imgs_fake + 1.0) * 127.5).to(torch.uint8)
            if imgs_fake.size(1) == 1:
                imgs_fake = imgs_fake.repeat(1, 3, 1, 1)
                
            self.fid.update(imgs_fake, real=False)

            batch_idx += 1
            seen += imgs_real.size(0)

        return float(self.fid.compute().cpu().item())
