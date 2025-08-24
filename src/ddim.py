
# diffusion_models.py
"""
Pure class-conditional diffusion (no classifier-free guidance) using
Hugging Face diffusers + PyTorch with DDIM sampling.

- Class: ConditionalDDIM
    * UNet2DModel with class embeddings (class_embed_type="timestep")
    * DDIMScheduler
    * sample(...) -> List[PIL.Image.Image]
    * diffusion_loss(...) -> scalar loss for a training step

Inputs are expected in [-1, 1]; if [0, 1] is given, we auto-scale.
"""

from typing import Iterable, List, Optional, Sequence, Tuple, Union
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from diffusers import UNet2DModel, DDIMScheduler
except Exception as e:
    raise ImportError("Requires 'diffusers'. Install via `pip install diffusers`.") from e


def _arch_from_image_size(
    image_size: int,
) -> Tuple[Tuple[int, ...], Tuple[str, ...], Tuple[str, ...]]:
    if image_size == 32:
        block_out_channels = (128, 256, 256, 256)
        down_block_types = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
        up_block_types = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
    elif image_size == 64:
        block_out_channels = (128, 256, 384, 512)
        down_block_types = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
        up_block_types = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
    elif image_size == 128:
        block_out_channels = (128, 128, 256, 384, 512)
        down_block_types = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D")
        up_block_types = ("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
    else:
        raise ValueError(f"No defaults for image_size={image_size}. Pass custom blocks.")
    return block_out_channels, down_block_types, up_block_types


def _maybe_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    x_min, x_max = float(x.min()), float(x.max())
    if -0.05 <= x_min and x_max <= 1.05 and x_min >= 0.0:
        return x * 2.0 - 1.0
    return x


def _to_pil_list(x: torch.Tensor) -> List[Image.Image]:
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0  # [0,1]
    x = (x * 255.0).round().to(torch.uint8)

    b, c, h, w = x.shape
    imgs: List[Image.Image] = []
    for i in range(b):
        xi = x[i]
        if c == 1:
            imgs.append(Image.fromarray(xi[0].numpy(), mode="L"))
        elif c == 3:
            imgs.append(Image.fromarray(xi.permute(1, 2, 0).numpy(), mode="RGB"))
        else:
            imgs.append(Image.fromarray(xi[0].numpy(), mode="L"))
    return imgs


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class ConditionalDDIM(nn.Module):
    """
    Pure label-conditioned diffusion with UNet2DModel and DDIM.

    Parameters
    ----------
    in_channels : int
        Number of input channels (out_channels equals in_channels).
    image_size : int
        One of {32, 64, 128} for built-in defaults (or pass custom blocks).
    num_class_labels : int
        Number of class labels for conditioning.
    block_out_channels, down_block_types, up_block_types : optional
        If not given, default to the prompt's mappings for `image_size`.
    num_train_timesteps : int
    beta_schedule : str
    prediction_type : str
    layers_per_block : int

    Backward-compatibility:
        * `cfg_uncond_prob` is accepted but ignored (no unconditional id reserved).
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        num_class_labels: int,
        block_out_channels: Optional[Sequence[int]] = None,
        down_block_types: Optional[Sequence[str]] = None,
        up_block_types: Optional[Sequence[str]] = None,
        num_train_timesteps: int = 1000,
        # beta_schedule: str = "cosine",
        prediction_type: str = "epsilon",
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        # ignored, for BC with earlier version
        cfg_uncond_prob: float = 0.0,
    ) -> None:
        super().__init__()

        if cfg_uncond_prob and cfg_uncond_prob > 0.0:
            warnings.warn("cfg_uncond_prob is ignored in pure-conditional mode.", RuntimeWarning)

        if block_out_channels is None or down_block_types is None or up_block_types is None:
            bo, db, ub = _arch_from_image_size(int(image_size))
            block_out_channels = bo if block_out_channels is None else tuple(block_out_channels)
            down_block_types = db if down_block_types is None else tuple(down_block_types)
            up_block_types = ub if up_block_types is None else tuple(up_block_types)

        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.num_class_labels = int(num_class_labels)

        self.unet = UNet2DModel(
            sample_size=self.image_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            block_out_channels=tuple(block_out_channels),
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            layers_per_block=layers_per_block,
            # class_embed_type="timestep",
            num_class_embeds=self.num_class_labels,  # no reserved id
            norm_num_groups=norm_num_groups,
        )

        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            # beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

    # ---------- Training ----------

    def diffusion_loss(
        self,
        images: torch.Tensor,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Standard noise-prediction MSE loss (pure conditional).

        images : (B,C,H,W) in [-1,1] (auto-scales from [0,1])
        labels : (B,) in [0, num_class_labels-1]
        """
        device = images.device
        # images = _maybe_to_minus1_1(images)

        noise = torch.randn_like(images)
        b = images.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (b,), device=device, dtype=torch.long
        )
        noisy_images = self.scheduler.add_noise(images, noise, timesteps)

        # pure conditional: use labels as-is
        class_labels = labels.to(device)

        model_pred = self.unet(noisy_images, timesteps, class_labels).sample
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        else:
            target = self.scheduler.get_velocity(images, noise, timesteps)

        return F.mse_loss(model_pred, target, reduction="mean")

    # ---------- Sampling ----------

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        labels: Union[int, Iterable[int], torch.Tensor],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        save: Optional[str] = None,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        guidance_scale: Optional[float] = None,  # accepted for BC, but must be 0/None
    ) -> List[Image.Image]:
        """
        DDIM sampling (pure conditional).

        labels : int | iterable[int] | LongTensor
            If an int is given, it's repeated across the batch.
        guidance_scale : must be None or 0.0 (CFG is disabled in this pure version).
        """
        if guidance_scale not in (None, 0, 0.0):
            raise ValueError("This pure-conditional implementation does not support CFG. "
                             "Set guidance_scale=None/0, or use the earlier file.")

        self.unet.eval()

        device = device or next(self.unet.parameters()).device
        if isinstance(device, str):
            device = torch.device(device)

        # Prepare labels
        if isinstance(labels, int):
            class_labels = torch.full((batch_size,), labels, dtype=torch.long, device=device)
        else:
            class_labels = torch.as_tensor(list(labels) if not torch.is_tensor(labels) else labels,
                                           dtype=torch.long, device=device)
            if class_labels.numel() == 1 and batch_size > 1:
                class_labels = class_labels.repeat(batch_size)
            assert class_labels.shape[0] == batch_size, "labels length must match batch_size"

        # Init noise
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            latents = torch.randn((batch_size, self.in_channels, self.image_size, self.image_size),
                                  generator=g, device=device)
        else:
            latents = torch.randn((batch_size, self.in_channels, self.image_size, self.image_size), device=device)

        # DDIM setup
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        extra_step_kwargs = {"eta": eta} if "eta" in self.scheduler.step.__code__.co_varnames else {}

        for t in self.scheduler.timesteps:
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            eps = self.unet(latent_model_input, t, class_labels).sample
            step_out = self.scheduler.step(eps, t, latents, **extra_step_kwargs)
            latents = step_out.prev_sample



        if save is not None:
            latents = _to_pil_list(latents)
            save = str(save)
            root, ext = os.path.splitext(save)
            if batch_size == 1 and ext.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                latents[0].save(save)
            else:
                _ensure_dir(save)
                for i, im in enumerate(latents):
                    im.save(os.path.join(save, f"sample_{i:04d}.png"))

        return latents


def build_conditional_ddim(
    in_channel: int,
    image_size: int,
    num_class_labels: int,
    **kwargs,
) -> ConditionalDDIM:
    return ConditionalDDIM(
        in_channels=in_channel,
        image_size=image_size,
        num_class_labels=num_class_labels,
        **kwargs,
    )
