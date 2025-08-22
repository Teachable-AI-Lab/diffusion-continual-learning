import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import os
from tqdm import tqdm
from diffusers import UNet2DModel

class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule = 'linear', device='cpu'):
        self.timesteps = timesteps
        self.device = device
        # linear beta schedule
        # betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        if beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps).to(device)
        else:
            # linear beta schedule
            betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alpha_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = alpha_cumprod
        self.alpha_cumprod_prev = alpha_cumprod_prev

        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
        # posterior variance for reverse step
        self.posterior_variance = betas * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
    
    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in Nichol & Dhariwal 2021.
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps)
        # convert to cumulative product of alphas
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # betas are the amount of noise added at each step
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(max=0.999)

    def add_noise(self, x0, t, noise=None):
        """
        Forward q(x_t | x_0): sample xt = sqrt(alpha_cum[t]) * x0 + sqrt(1 - alpha_cum[t]) * ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1) * x0
            + self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1) * noise,
            noise
        )

    def step(self, pred_noise, t, xt):
        """
        Compute x_{t-1} from model prediction at timestep t.
        """
        # scalars for this t
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        a_cumprod_t = self.alpha_cumprod[t]
        a_cumprod_prev = self.alpha_cumprod_prev[t]

        # predict x0:
        x0_pred = (xt - self.sqrt_one_minus_alpha_cumprod[t] * pred_noise) / self.sqrt_alpha_cumprod[t]

        # posterior mean:
        coef1 = torch.sqrt(a_cumprod_prev) * beta_t / (1 - a_cumprod_t)
        coef2 = torch.sqrt(alpha_t) * (1 - a_cumprod_prev) / (1 - a_cumprod_t)
        posterior_mean = coef1.view(-1,1,1,1) * x0_pred + coef2.view(-1,1,1,1) * xt

        # add noise except for t=0
        noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
        return posterior_mean + torch.sqrt(self.posterior_variance[t]).view(-1,1,1,1) * noise
    
class DDPM(nn.Module):
    def __init__(self, 
                 unet: UNet2DModel, 
                 scheduler: NoiseScheduler, 
                 n_classes=0,
                 condition_dim=1,
                 device='cpu'):
        super().__init__()
        self.unet = unet.to(device)
        self.scheduler = scheduler
        self.device = device
        self.n_classes = n_classes
        if n_classes > 0:
            # UNet2DModel expects a batch of labels, not a single label
            self.label_embedding = nn.Embedding(n_classes, condition_dim).to(device)
        
    

    def forward(self, x0, t=None, y=None, add_noise=True):
        b = x0.shape[0]
        if t is None:
            # sample random timesteps for each example
            t = torch.randint(0, self.scheduler.timesteps, (b,), device=self.device).long()
        xt = x0
        noise = None
        if add_noise:
            # add noise
            xt, noise = self.scheduler.add_noise(x0, t)

        if self.n_classes > 0: # conditional DDPM
            # UNet2DModel expects a batch of labels, not a single label
            bs, C, H, W = xt.shape
            y_embed = self.label_embedding(y)
            y_embed = y_embed.view(bs, -1, 1, 1).expand(bs, -1, H, W)
            # concatenate label embedding to input
            xt = torch.cat((xt, y_embed), dim=1)

        # predict noise with UNet
        # UNet2DModel returns a ModelOutput with `.sample` containing the predicted noise
        pred = self.unet(xt, t, y).sample
        return pred, noise

    def training_loss(self, x0):
        pred_noise, noise = self(x0)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, num_samples_per_label, image_dim=(3, 32, 32), condition=None):
        # start from pure Gaussian noise
        C, H, W = image_dim
        total_samples = num_samples_per_label * len(condition) if condition is not None else num_samples_per_label
        labels = condition.repeat_interleave(num_samples_per_label, dim=0) if condition is not None else None
        img = torch.randn(total_samples, C, H, W, device=self.device)
        for t in reversed(range(self.scheduler.timesteps)):
            t_batch = torch.full((total_samples,), t, device=self.device, dtype=torch.long)
            pred_noise = self(img, t_batch, labels, add_noise=False)[0]
            img = self.scheduler.step(pred_noise, t, img)
        # clamp to [-1,1], convert to [0,1]
        return img.clamp(-1,1)

    @torch.no_grad()
    def sample_interval(self, batch_size, interval=100, image_dim=(3, 32, 32), condition=None):
        
        # start from pure Gaussian noise
        C, H, W = image_dim
        samples = []
        img = torch.randn(batch_size, C, H, W, device=self.device)
        for t in reversed(range(self.scheduler.timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            pred_noise = self.unet(img, t_batch, condition).sample
            img = self.scheduler.step(pred_noise, t, img)
            if t % interval == 0:
                samples.append(img.clone())
        # clamp to [-1,1], convert to [0,1]
        return torch.stack(samples, dim=1).clamp(-1, 1)