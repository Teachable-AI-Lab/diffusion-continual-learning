import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, IterableDataset, Dataset, Subset
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import random
# simple_fid_tm.py
# import torch
# from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from copy import deepcopy
import os
import json
import argparse

def get_cl_dataset(name='mnist', batch_size=64, normalize=True, greyscale=False):
    if name.lower() == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        if normalize:
            transform = transforms.Compose([
                transforms.Pad(2),  # Padding to make it 32x32
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = datasets.MNIST(root='archive/data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='archive/data', train=False, download=True, transform=transform)
    elif name.lower() == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor()])
        if normalize:
            transform = transforms.Compose([
                transforms.Pad(2),  # Padding to make it 32x32
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = datasets.FashionMNIST(root='archive/data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='archive/data', train=False, download=True, transform=transform)
    elif name.lower() == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        if normalize:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        if greyscale:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = datasets.CIFAR10(root='archive/data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='archive/data', train=False, download=True, transform=transform)
    else:
        train_dataset = None
        test_dataset = None

     # how many classes per group?
    group_size = 2
    n_classes  = 10
    n_groups   = n_classes // group_size  # == 5

    # return a dictionary: class_id: dataloader
    # train_indices_per_class = {i: [] for i in range(10)}
    train_indices_per_group = {g: [] for g in range(n_groups)}
    print("Building DataLoaders for each class in train dataset...")
    # for idx, (_, label) in enumerate(tqdm(train_dataset)):
        # train_indices_per_class[label].append(idx)
    for idx, (_, label) in enumerate(tqdm(train_dataset)):
        g = label // group_size
        train_indices_per_group[g].append(idx)

    # 3) Build one DataLoader per class
    train_loaders = {}
    # for class_id, indices in sorted(train_indices_per_class.items()):
    for g, indices in sorted(train_indices_per_group.items()):
        subset = Subset(train_dataset, indices)
        # train_loaders[class_id] = DataLoader(
        train_loaders[g] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,    # adjust as needed
            pin_memory=True
        )

    # test_indices_per_class = {i: [] for i in range(10)}
    test_indices_per_group = {g: [] for g in range(n_groups)}
    print("Building DataLoaders for each class in MNIST test dataset...")
    # for idx, (_, label) in enumerate(tqdm(test_dataset)):
        # test_indices_per_class[label].append(idx)
    for idx, (_, label) in enumerate(tqdm(test_dataset)):
        g = label // group_size
        test_indices_per_group[g].append(idx)
    # 3) Build one DataLoader per class
    test_loaders = {}
    # for class_id, indices in sorted(test_indices_per_class.items()):
    for g, indices in sorted(test_indices_per_group.items()):
        subset = Subset(test_dataset, indices)
        # test_loaders[class_id] = DataLoader(
        test_loaders[g] = DataLoader(
            subset,
            batch_size=512,
            shuffle=True,
            num_workers=0,    # adjust as needed
            pin_memory=True
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    return train_loaders, test_loaders, train_loader, test_loader

def load_config_from_json(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {config_path}: {e}")

    config_args = argparse.Namespace(**config_dict)

    # if hasattr(config_args, 'dec_hidden_dim') and isinstance(config_args.dec_hidden_dim, list):
        # config_args.dec_hidden_dim = tuple(config_args.dec_hidden_dim)

    return config_args

def train_one_task(model, train_loader, class_id, optimizer, 
                   ewc=None,
                   gr=None,
                   kl=False,
                   num_epochs=10, save_path=None, device='cuda', wandb=None):
    
    for epoch in tqdm(range(num_epochs)):
        for batch in tqdm(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            if gr is not None:
                # combine with generated old data
                x_old, y_old = gr.replay()
                # randomly select half of the batch size from real data
                # x_new = images[:images.size(0)//2]
                # y_new = labels[:images.size(0)//2]
                images = torch.cat([images, x_old], dim=0)
                labels = torch.cat([labels, y_old], dim=0)
                # should have the same batch size = 256 + 64 = 320
                # assert images.size(0) == 320 # TODO: For now

                # shuffle the combined batch
                perm = torch.randperm(images.size(0))
                images = images[perm]
                labels = labels[perm]

            optimizer.zero_grad()
            loss = 0
            ddim_loss = model.diffusion_loss(images, labels)
            loss = loss + ddim_loss
            if ewc is not None:
                loss_ewc = ewc.loss(model)#.penalty() if ewc is not None else torch.zeros((), device=device)
                loss = loss + model.ewc_lambda * loss_ewc

            if kl and gr is not None:
                # compute the generative replay distillation loss
                b_old = x_old.size(0)
                t_kl = torch.randint(0, model.scheduler.num_train_timesteps, (b_old,), device=device).long()
                eps = torch.randn_like(x_old, device=device)
                x_noisy = model.scheduler.add_noise(x_old, eps, t_kl)

                with torch.no_grad():
                    eps_teacher = gr.teacher.unet(x_noisy, t_kl, y_old).sample
                eps_student = model.unet(x_noisy, t_kl, y_old).sample
                loss_kl = F.mse_loss(eps_student, eps_teacher)
                loss = loss + model.gr_kl * loss_kl

            loss.backward()
            optimizer.step()

        if wandb is not None:
            wandb.log({
                'loss/ddim': ddim_loss.item(),
                'loss/ewc': loss_ewc.item() if ewc is not None else 0.0,
                'loss/kl': loss_kl.item() if (kl and gr is not None) else 0.0,
                'loss/total': loss.item(),
                'epoch': epoch + num_epochs * class_id,
            })


        # visualize every 50 epochs
        if save_path is not None and epoch % 20 == 0:
            # sample 8 images for each label from 0 to 9
            out_dir = Path(save_path) / f"task_{class_id}" / f"epoch_{epoch:05d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            # num_classes = model.num_class_labels
            rows = model.num_class_labels
            cols = 8
            all_tensors = []
            for c in range(rows):
                pils = model.sample(
                    batch_size=cols,
                    labels=[c] * cols,
                    num_inference_steps=50,
                    device=device,
                    guidance_scale=0.0,  # pure conditional
                )
                for im in pils:
                    # to [C,H,W] float in [0,1]
                    all_tensors.append((im + 1.0) * 0.5)  # [0,1] float

            grid = make_grid(torch.stack(all_tensors, dim=0), nrow=cols, padding=2)  # 8 per row
            grid_pil = TF.to_pil_image(grid.clamp(0, 1)) # is grid_pil a PIL image? Yes, it is.
            if wandb is not None:
                wandb.log({f"samples/task{class_id}": wandb.Image(grid_pil, caption=f"Task {class_id} Epoch {epoch}")})
            out_file = out_dir / f"epoch_{epoch:05d}_grid.png"
            grid_pil.save(out_file)

    # return {
    #     "ddim_loss": ddim_loss.item(),
    #     "ewc_loss": loss_ewc.item() if ewc is not None else 0.0,
    #     "kl_loss": loss_kl.item() if (kl is not None and gr is not None) else 0.0,
    #     "loss": loss.item(),
    #     # return images
    #     "images": grid_pil if save_path is not None else None,
    # } 

def train_continual_learning(model, cl_train_loaders, optimizer, 
                             ewc=None,
                             num_epochs=10, save_path=None, device='cuda'):
    for task_id in sorted(cl_train_loaders.keys()):
        loader = cl_train_loaders[task_id]

        # Train current task with EWC penalty (if ewc is not None)
        train_one_task(
            model,
            loader,
            task_id=task_id,
            optimizer=optimizer,
            num_epochs=num_epochs,
            save_path=save_path,
            device=device,
            ewc=ewc,
        )

        # Consolidate Fisher on current task
        if ewc is not None:
            print(f"[EWC] Consolidating Fisher on task {task_id}...")
            ewc.consolidate(loader)



class FIDEvaluator:
    def __init__(self, device=None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # normalize=True => inputs should be float in [0,1]
        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _to01_and_rgb(x: torch.Tensor) -> torch.Tensor:
        """x: (B,C,H,W) float in [0,1] or [-1,1]; returns float in [0,1] with 3 channels."""
        if x.dtype.is_floating_point:
            if x.min() < 0.0:  # convert [-1,1] -> [0,1]
                x = (x + 1.0) * 0.5
        else:
            x = x.float() / 255.0
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return x.clamp(0.0, 1.0).to(torch.uint8)

    @torch.no_grad()
    def fid_loader_vs_model(
        self,
        real_loader,
        model,
        # num_classes: int | None = None,    # kept for signature compatibility (unused when match_labels=True)
        # total_fake: int = 10000,           # ignored when match_labels=True
        # batch_size_fake: int = 200,        # ignored when match_labels=True
        num_inference_steps: int = 50,
        seed: int | None = 123,
        # match_labels: bool = True,         # <--- new: use real labels for fake sampling
        max_real: int | None = None,       # optional cap on #real images processed
    ) -> float:
        self.fid.reset()
        dev = self.device
        seed_base = seed if seed is not None else 0
        bidx = 0
        seen = 0
        for imgs_real, labels_real in real_loader:
            if max_real is not None and seen >= max_real:
                break
            if max_real is not None and seen + imgs_real.size(0) > max_real:
                keep = max_real - seen
                imgs_real = imgs_real[:keep]
                labels_real = labels_real[:keep]

            # real -> [0,1], 3ch
            imgs_real = imgs_real.to(dev)
            imgs_real = (imgs_real + 1.0) * 127.5
            # imgs_real = self._to01_and_rgb(imgs_real)
            imgs_real = imgs_real.to(torch.uint8)
            if imgs_real.size(1) == 1:
                imgs_real = imgs_real.repeat(1, 3, 1, 1)
            # print(imgs_real)
            self.fid.update(imgs_real, real=True)

            # fake: sample with EXACT same labels/order
            # random label between 0 and 1
            # labels_real = random.choices(
            #     labels_real.tolist(),
            #     k=imgs_real.size(0)
            # )
            # labels_real = torch.tensor(labels_real, device=dev, dtype=torch.long)
            seed_b = seed_base + bidx if seed is not None else None
            imgs_pil = model.sample(
                batch_size=labels_real.numel(),
                labels=labels_real.tolist(),             # <- match labels
                num_inference_steps=num_inference_steps,
                device=dev,
                seed=seed_b,
            )
            imgs_fake = ((imgs_pil + 1.0) * 127.5).to(torch.uint8)  # [0,255] uint8
            if imgs_fake.size(1) == 1:
                imgs_fake = imgs_fake.repeat(1, 3, 1, 1)
            # print(imgs_fake)
            # imgs_fake = torch.stack([#print(im.max(), im.min())
                
                                    #   for im in imgs_pil]).to(dev)
            # break
            # print(imgs_fake)
            # imgs_fake = self._to01_and_rgb(imgs_fake)
            self.fid.update(imgs_fake, real=False)

            bidx += 1
            seen += imgs_real.size(0)

        return float(self.fid.compute().cpu().item())
    

@torch.no_grad() # return a frozen teacher model
def freeze_model(model):
    teacher = deepcopy(model)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    return teacher
    
    

#-----------------------------------------------------------------------------#
# Compare Fisher
import torch
import torch.nn.functional as F
from tqdm import tqdm

def compare_fisher_errors_streaming(
    model,
    t_level: int,
    loaders_by_class,                  # dict[int, DataLoader] -> (images, labels)
    mu: torch.Tensor,                  # v (flattened) in the same param order as unet.parameters()
    c_star: float,                     # optimal scalar coefficient
    target_class: int = 0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float64,
    max_samples: int | None = None,
    num_diag_probes: int = 15,          # z-probes for ||diag(E[g⊙g])||_F^2
):
    """
    Returns Frobenius errors (absolute and relative) for:
      - rank1 (c=1)
      - rank1_opt (c=c_star)
      - diag (estimated)
    Also returns components: vFv, ||F||_F, ||D||_F, ||v||^2, tr(F).
    """
    model.eval(); model = model.to(device)
    unet, scheduler = model.unet, model.scheduler

    # data loader
    if target_class not in loaders_by_class:
        raise KeyError(f"class_id {target_class} not found")
    loader = loaders_by_class[target_class]

    # v = mu (flattened), constants
    v = mu.detach().to(dtype).cpu()    # compute on CPU (memory-friendly)
    v_norm2 = float(v.dot(v))          # ||v||^2
    c_star = float(c_star)

    # Pre-generate Rademacher probes z for diagonal norm (CPU, int8 -> lightweight)
    D = v.numel()
    z_list = [torch.randint(0, 2, (D,), dtype=torch.int8).mul_(2).sub_(1) for _ in range(num_diag_probes)]

    # Accumulators
    N = 0
    sum_s2 = 0.0                       # for v^T F v = E[(g^T v)^2]
    sum_pair_sq = 0.0                  # for ||F||_F^2 = E[(g^T g')^2]
    num_pairs = 0
    u_tot = [0.0 for _ in range(num_diag_probes)]  # for ||diag(E[g⊙g])||_F via Hutchinson on diag
    trace_sum = 0.0                    # tr(F) = E[||g||^2]

    prev_g = None                      # hold one g to form a pair (adjacent samples)

    torch.set_grad_enabled(True)
    pbar = tqdm(loader, desc=f"compare_F_errors@t={t_level}")

    for images, labels in pbar:
        images = images.to(device); labels = labels.to(device)
        for img, lab in zip(images, labels):
            img = img.unsqueeze(0); lab = lab.unsqueeze(0)
            t = torch.full((1,), int(t_level), device=device, dtype=torch.long)

            # Draw noise, make noisy input
            noise = torch.randn_like(img, device=device)
            noisy_x = scheduler.add_noise(img, noise, t)

            # Forward UNet (conditional/unconditional)
            try:
                out = unet(noisy_x, t, lab)
            except TypeError:
                out = unet(noisy_x, t)
            pred_noise = out.sample if hasattr(out, "sample") else out

            # Per-sample MSE and grad wrt UNet params
            unet.zero_grad(set_to_none=True)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            # Flatten grad to CPU (float64) — one vector only
            g_list = [p.grad.reshape(-1) for p in unet.parameters() if p.requires_grad and p.grad is not None]
            if not g_list: raise RuntimeError("No grads on UNet parameters.")
            g = torch.cat(g_list).detach().cpu().to(dtype)

            # v^T F v = E[(g^T v)^2]
            s = float(g.dot(v))
            sum_s2 += s * s

            # tr(F) = E[||g||^2]
            trace_sum += float(g.pow(2).sum())

            # ||diag(E[g⊙g])||_F^2 via Hutchinson-on-diagonal:
            # For each probe z: m_z = E[(g⊙g)^T z] ≈ (1/N) Σ (g⊙g)^T z  ; then ||diag||_F^2 ≈ mean_k m_z^2
            g2 = g * g
            for k, z in enumerate(z_list):
                # cast z to float64 on the fly (keeps memory low)
                u_tot[k] += float((g2 * z.to(torch.float64)).sum())

            # ||F||_F^2 via adjacent independent pairs: E[(g^T g')^2]
            if prev_g is None:
                prev_g = g
            else:
                dot = float(prev_g.dot(g))
                sum_pair_sq += dot * dot
                num_pairs += 1
                prev_g = None  # disjoint pairs: (1,2), (3,4), ...

            N += 1
            unet.zero_grad(set_to_none=True)

            if max_samples is not None and N >= max_samples:
                break
        if max_samples is not None and N >= max_samples:
            break

    if N == 0:
        raise RuntimeError("No samples processed.")

    # Estimates
    vFv = sum_s2 / N
    trF = trace_sum / N

    # ||diag(E[g⊙g])||_F^2
    m_sq = [(u / N) ** 2 for u in u_tot]                 # m_z^2
    diagF_frob_sq = sum(m_sq) / max(num_diag_probes, 1)  # average over probes

    # ||F||_F^2
    if num_pairs == 0:
        raise RuntimeError("Need at least one pair to estimate ||F||_F^2. Increase max_samples.")
    frobF_sq = sum_pair_sq / num_pairs

    # Errors (Frobenius)
    err_rank1_sq      = frobF_sq - 2.0 * vFv + (v_norm2 ** 2)
    err_rank1opt_sq   = frobF_sq - 2.0 * c_star * vFv + (c_star ** 2) * (v_norm2 ** 2)
    err_diag_sq       = frobF_sq - diagF_frob_sq

    # Stabilize small negatives due to MC noise
    clip = lambda x: float(torch.clamp(torch.tensor(x, dtype=dtype), min=0.0))
    err_rank1    = clip(err_rank1_sq)    ** 0.5
    err_rank1opt = clip(err_rank1opt_sq) ** 0.5
    err_diag     = clip(err_diag_sq)     ** 0.5
    frobF        = frobF_sq ** 0.5

    return {
        "components": {
            "v_norm2": v_norm2,
            "vFv": vFv,
            "trF": trF,
            "||F||_F": frobF,
            "||diag(E[g⊙g])||_F": diagF_frob_sq ** 0.5,
            "num_samples": N,
            "num_pairs": num_pairs,
            "num_diag_probes": num_diag_probes,
        },
        "errors_frobenius": {
            "rank1_c1": {"abs": err_rank1,    "rel": err_rank1 / (frobF + 1e-12)},
            "rank1_c*": {"abs": err_rank1opt, "rel": err_rank1opt / (frobF + 1e-12), "c*": c_star},
            "diag":     {"abs": err_diag,     "rel": err_diag / (frobF + 1e-12)},
        },
    }

