# generative_replay.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple, Union, Literal
import torch
from torch import Tensor
from tqdm import tqdm

LabelPolicy = Literal["round_robin", "balanced", "uniform"]


class GenerativeReplay:
    """
    Minimal on-the-fly generative replay from a frozen teacher.

    - Call `replay(policy=...)` to get a *balanced* batch of (x_old, y_old)
      according to the chosen label policy.
    - Expects `teacher.sample(..., save=None)` to return a *tensor* batch
      shaped [B, C, H, W].

    Args:
        teacher: frozen generator from the end of task (t-1) with .sample(...)
        old_classes: iterable of old class IDs (e.g., [0,1,2,3])
        batch_size: your training batch size (used to compute replay size)
        alpha: fraction of *new* real data you plan to mix externally
               (replay size is n_old = round((1 - alpha) * batch_size))
        num_inference_steps, eta, seed, device, guidance_scale:
            forwarded to teacher.sample(...)
    """

    def __init__(
        self,
        teacher,
        old_classes: Sequence[int],
        batch_size: int,
        alpha: float = 0.5,
        pool_size_per_class: int = 5000,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.teacher = teacher
        self.batch_size = int(batch_size)
        self.alpha = float(alpha)
        self.num_inference_steps = int(num_inference_steps)
        self.eta = float(eta)
        self.seed = seed
        self.pool_size_per_class = int(pool_size_per_class)

        # resolve device
        if device is None:
            try:
                device = next(self.teacher.unet.parameters()).device
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        # store classes & round-robin rotor
        self.old_classes = torch.tensor(sorted(set(int(c) for c in old_classes)), dtype=torch.long)
        self._rotor = 0  # advances so remainders get shared across calls

        self.build_pool()
    
    @torch.no_grad()
    def build_pool(self):
        # pre-generated pool of images for replay
        self.pool = []
        self.pool_labels = []
        bs = 500  # generate in batches to avoid OOM
        for class_id in tqdm(self.old_classes.tolist()):
            n_samples = 0
            while n_samples < self.pool_size_per_class:
                seed = self.seed + n_samples
                y_old = torch.full((bs,), class_id, dtype=torch.long, device=self.device)
                x_old = self.teacher.sample(
                    batch_size=bs,
                    labels=y_old,
                    num_inference_steps=self.num_inference_steps,
                    eta=self.eta,
                    save=None,                  # must return a tensor batch
                    seed=seed,            
                    device=self.device,
                )
                self.pool.append(x_old)
                self.pool_labels.append(y_old)
                n_samples += bs
        self.pool = torch.cat(self.pool, dim=0)
        self.pool_labels = torch.cat(self.pool_labels, dim=0)
        self.pool_size = self.pool.shape[0]
        print(f"GenerativeReplay: built pool of {self.pool_size} images for replay.")
        self.pool_indices = torch.randperm(self.pool_size, device=self.device)   
        self.pool_ptr = 0     

    def set_old_classes(self, old_classes: Sequence[int]) -> None:
        """Update the set of old classes (e.g., at the start of a new task)."""
        self.old_classes = torch.tensor(sorted(set(int(c) for c in old_classes)), dtype=torch.long)
        self._rotor = 0

    def update_teacher(self, new_teacher, old_classes: Optional[Sequence[int]] = None) -> None:
        """Swap in a new frozen teacher (e.g., after finishing the current task)."""
        self.teacher = new_teacher
        if old_classes is not None:
            self.set_old_classes(old_classes)
        # reset pool
        self.build_pool()

    def n_old(self) -> int:
        """Number of replay samples given (batch_size, alpha)."""
        return int(round((1.0 - self.alpha) * self.batch_size))

    @torch.no_grad()
    def replay(self):
        # randomly sample from the pre-generated pool
        n_replay_samples = self.n_old()
        x_old = self.pool[self.pool_ptr:self.pool_ptr + n_replay_samples]
        y_old = self.pool_labels[self.pool_ptr:self.pool_ptr + n_replay_samples]
        self.pool_ptr += n_replay_samples
        if self.pool_ptr + n_replay_samples > self.pool_size:
            self.pool_indices = torch.randperm(self.pool_size, device=self.device)   
            self.pool_ptr = 0
        return x_old, y_old


    # @torch.no_grad()
    # def _replay(
    #     self,
    #     policy: LabelPolicy = "round_robin",
    #     *,
    #     rng: Optional[torch.Generator] = None
    # ) -> Tuple[Tensor, Tensor]:
    #     """
    #     Return a replay batch (x_old, y_old) according to the chosen label policy.

    #     Args:
    #         policy: "round_robin" | "balanced" | "uniform"
    #         rng: optional torch.Generator for reproducible label sampling (uniform/balanced)

    #     Returns:
    #         x_old: [n_old, C, H, W]  tensor from teacher.sample(...)
    #         y_old: [n_old]           LongTensor of class IDs
    #     """
    #     if self.old_classes.numel() == 0:
    #         raise RuntimeError("GenerativeReplay: no old classes registered. Set them first.")

    #     n_old = self.n_old()
    #     if n_old <= 0:
    #         raise ValueError(
    #             f"GenerativeReplay: n_old <= 0 with batch_size={self.batch_size} and alpha={self.alpha}."
    #         )

    #     y_old = self._make_labels(n_old, policy=policy, rng=rng).to(self.device)

    #     x_old = self.teacher.sample(
    #         batch_size=n_old,
    #         labels=y_old,
    #         num_inference_steps=self.num_inference_steps,
    #         eta=self.eta,
    #         save=None,                  # must return a tensor batch
    #         seed=self.seed,             # seed for noise init (optional)
    #         device=self.device,
    #         # guidance_scale=self.guidance_scale,
    #     )

    #     if isinstance(x_old, list):
    #         raise ValueError(
    #             "GenerativeReplay: teacher.sample returned a list. "
    #             "Ensure save=None so it returns a tensor batch."
    #         )

    #     return x_old, y_old

    # # ---------------- internal: label builders ----------------

    # @torch.no_grad()
    # def _make_labels(
    #     self, n_old: int, *, policy: LabelPolicy, rng: Optional[torch.Generator]
    # ) -> Tensor:
    #     K = int(self.old_classes.numel())
    #     if policy == "round_robin":
    #         idx = (torch.arange(n_old) + self._rotor) % K
    #         labels = self.old_classes[idx]
    #         # advance rotor so any remainder distributes across calls
    #         step = (n_old % K) or 1
    #         self._rotor = (self._rotor + step) % K
    #         return labels

    #     elif policy == "balanced":
    #         # as equal as possible *within this call*
    #         base = n_old // K
    #         rem = n_old % K
    #         labels = []
    #         # optional small shuffle of the class order to randomize who gets the remainder
    #         order = self.old_classes.clone()
    #         if rng is not None:
    #             perm = torch.randperm(K, generator=rng)
    #             order = order[perm]
    #         for i, c in enumerate(order.tolist()):
    #             count = base + (1 if i < rem else 0)
    #             if count > 0:
    #                 labels.extend([c] * count)
    #         return torch.tensor(labels, dtype=torch.long)

    #     elif policy == "uniform":
    #         # i.i.d. uniform over old classes
    #         if rng is None:
    #             rng = torch.Generator(device="cpu")
    #         idx = torch.randint(low=0, high=K, size=(n_old,), generator=rng)
    #         return self.old_classes[idx]

    #     else:
    #         raise ValueError(f"Unknown policy: {policy!r}")
