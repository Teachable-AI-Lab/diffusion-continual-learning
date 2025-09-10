import math
from pathlib import Path
import json

import numpy as np
import torch
import wandb

import src.utils as utils
from src.ddim import build_conditional_ddim
from src.fisher_analysis import empirical_fisher_dense
from src.parameter_scoring import (
    compute_param_scores,
    compute_rank1_coeff_and_mean,
)

# Cache of previous top eigenvectors per task for comparison across analyses
_prev_top_eigvec = {}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def log_bar_chart(name: str, labels, values, xname="rank", yname="eigenvalue", prefix=None):
    if not wandb.run:
        return
    try:
        table = wandb.Table(
            data=[[int(l), float(v)] for l, v in zip(labels, values)],
            columns=[xname, yname],
        )
        try:
            chart = wandb.plot.bar(table, xname, yname, title=name)  # type: ignore[attr-defined]
        except Exception:
            chart = wandb.plot.line(table, x=xname, y=yname, title=name)
        key = f"charts/{name}" if not prefix else f"{prefix}/charts/{name}"
        wandb.log({key: chart})
        return
    except Exception:
        pass

    try:
        import matplotlib  # type: ignore
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore

        labels_i = [int(l) for l in labels]
        values_f = [float(v) for v in values]
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.bar(labels_i, values_f)
        ax.set_title(name)
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        key = f"charts/{name}" if not prefix else f"{prefix}/charts/{name}"
        wandb.log({key: wandb.Image(fig)})
        plt.close(fig)
        return
    except Exception:
        pass

    try:
        key = f"charts/{name}_hist" if not prefix else f"{prefix}/charts/{name}_hist"
        wandb.log({key: wandb.Histogram(np.array(values, dtype=float))})
    except Exception:
        pass


def get_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(args, channels, im_size, device):
    model_size = getattr(args, "model_size", "small")
    if model_size == "small":
        model = build_conditional_ddim(
            in_channel=channels,
            image_size=im_size,
            num_class_labels=args.num_classes,
            block_out_channels=(16,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            norm_num_groups=8,
            layers_per_block=1,
        ).to(device)
    elif model_size == "small-double":
        model = build_conditional_ddim(
            in_channel=channels,
            image_size=im_size,
            num_class_labels=args.num_classes,
            block_out_channels=(16, 16, 16),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            norm_num_groups=8,
            layers_per_block=1,
        ).to(device)
    elif model_size == "small-big":
        model = build_conditional_ddim(
            in_channel=channels,
            image_size=im_size,
            num_class_labels=args.num_classes,
            block_out_channels=(16, 16, 16),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            norm_num_groups=8,
            layers_per_block=2,
        ).to(device)
    elif model_size == "big":
        model = build_conditional_ddim(
            in_channel=channels,
            image_size=im_size,
            num_class_labels=args.num_classes,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_size: {model_size}")
    print(f"Model size: {get_model_size(model)} parameters")
    return model


def effective_rank_from_param_scores(param_scores: torch.Tensor, energy_threshold: float = 0.99, svals=None) -> int:
    if param_scores is None or param_scores.ndim != 2:
        raise ValueError("param_scores must be a 2D tensor (B, D)")
    B = param_scores.shape[0]
    if B == 0:
        return 0
    with torch.no_grad():
        if svals is None:
            svals = torch.linalg.svdvals(param_scores)
        lambdas = (svals ** 2) / float(B)
        energies = (lambdas ** 2)
        if energies.numel() == 0:
            return 0
        total_energy = energies.sum()
        if torch.isclose(total_energy, torch.tensor(0.0, device=energies.device)):
            return 0
        energies_sorted, _ = torch.sort(energies, descending=True)
        cumsum = torch.cumsum(energies_sorted, dim=0)
        target = float(energy_threshold) * float(total_energy)
        idx = torch.searchsorted(cumsum, torch.tensor(target, device=cumsum.device))
        rank = int(min(idx.item() + 1, energies_sorted.numel()))
        return rank


def fisher_analysis(model, args, train_loader, device, task_id, main_epoch, exp_dir: Path, time_level=None):
    eff_rank_list = []
    rank1_contribution = []
    top_eigvals_list = []
    fisher_max_samples = getattr(args, "fisher_max_samples", None)
    energy_threshold = getattr(args, "fisher_energy_threshold", 0.99)

    loaders_by_class = {0: train_loader}
    param_scores = compute_param_scores(
        model,
        loaders_by_class,
        device=device,
        target_class=0,
        max_samples=fisher_max_samples,
        time_level=time_level,
    )
    if param_scores.dtype in (torch.float16, torch.bfloat16):
        param_scores = param_scores.float()
    param_scores = (param_scores + 1e-12).contiguous()

    if not torch.isfinite(param_scores).all():
        print(f"[Fisher] Warning: Non-finite param_scores, fixing with nan_to_num for task {task_id}")
        param_scores = torch.nan_to_num(param_scores)

    B = param_scores.shape[0]
    eigvec_flag = bool(getattr(args, "eigvec_analysis", False))
    top_vec = None
    with torch.no_grad():
        try:
            if eigvec_flag:
                _U, S_svd, Vh_full = torch.linalg.svd(param_scores, full_matrices=False)
                top_vec = Vh_full[0, :]
            else:
                S_svd = torch.linalg.svdvals(param_scores)
        except:
            print(f"[Fisher] Warning: SVD failed once for task {task_id}")
            if eigvec_flag:
                _U, S_svd, Vh_full = torch.linalg.svd(param_scores, full_matrices=False)
                top_vec = Vh_full[0, :]
            else:
                S_svd = torch.linalg.svdvals(param_scores)
        lambdas = (S_svd ** 2) / float(B)
        frobF_sq_t = torch.sum(lambdas ** 2)
        frobF = float(torch.sqrt(frobF_sq_t + 1e-20))
        frobF_sq = float(frobF_sq_t)
        energies = (lambdas ** 2)
        total_energy = energies.sum()
        energies_sorted, _ = torch.sort(energies, descending=True)
        cumsum = torch.cumsum(energies_sorted, dim=0)
        target = float(energy_threshold) * float(total_energy)
        idx = torch.searchsorted(cumsum, torch.tensor(target, device=cumsum.device))
        eff_rank = int(min(idx.item() + 1, energies_sorted.numel()))
        highest_energy = energies_sorted[0].item() / float(total_energy) if total_energy.item() > 0 else 0.0

    if getattr(args, "check_fisher", False) and device.type == "cuda":
        try:
            F = empirical_fisher_dense(param_scores)
            norm_f = torch.linalg.norm(F, ord="fro").item()
            if not math.isclose(norm_f, frobF, rel_tol=1e-3):
                print(f"[Fisher] Warning: Frobenius norm mismatch ||F||_F={norm_f} vs {frobF}")
            del F
            del norm_f
        except Exception:
            print(f"[Fisher] Warning: empirical_fisher_dense failed, skipping dense F computation")

    eff_rank_list.append(float(eff_rank))
    rank1_contribution.append(float(highest_energy))

    with torch.no_grad():
        lambdas_sorted, _ = torch.sort(lambdas, descending=True)
        k = min(10, lambdas_sorted.numel())
        eigvals = [float(v) for v in lambdas_sorted[:k].tolist()]
        if k < 10:
            eigvals += [0.0] * (10 - k)
        top_eigvals_list.append(eigvals)

    c_stream, mu_stream, F_diag_stream = compute_rank1_coeff_and_mean(
        model, train_loader, device=device, max_samples=fisher_max_samples
    )
    mu_stream = mu_stream.to(param_scores.device, dtype=param_scores.dtype)
    v_norm2_stream = torch.dot(mu_stream, mu_stream).item()

    with torch.no_grad():
        Su_stream = param_scores @ mu_stream
        uTFu_stream = float(torch.sum(Su_stream.pow(2)) / float(B))
        err_cov_stream_abs = v_norm2_stream * v_norm2_stream
        err_rank1_stream_abs = math.sqrt(
            max(frobF_sq + (err_cov_stream_abs) - 2.0 * uTFu_stream, 0.0)
        )
        c_stream_f = float(c_stream)
        err_rank1opt_stream_abs = math.sqrt(
            max(frobF_sq + (c_stream_f ** 2) * (err_cov_stream_abs) - 2.0 * c_stream_f * uTFu_stream, 0.0)
        )
        diag_stream_norm_sq = float(torch.dot(F_diag_stream, F_diag_stream))
        err_diag_stream_abs = math.sqrt(max(frobF_sq - diag_stream_norm_sq, 0.0))

    cos_top_eig_mu = None
    cos_top_eig_prev = None
    if eigvec_flag and top_vec is not None:
        try:
            tvec = top_vec.to(mu_stream.device, dtype=mu_stream.dtype)
            v_norm = torch.linalg.norm(tvec) + 1e-12
            mu_norm = torch.linalg.norm(mu_stream) + 1e-12
            cos_top_eig_mu = float(torch.dot(tvec, mu_stream) / (v_norm * mu_norm))
            prev = _prev_top_eigvec.get(int(task_id))
            if prev is not None:
                prev = prev.to(tvec.device, dtype=tvec.dtype)
                pv = torch.linalg.norm(prev) + 1e-12
                cos_top_eig_prev = float(torch.dot(tvec, prev) / (v_norm * pv))
            _prev_top_eigvec[int(task_id)] = tvec.detach().cpu()
        except Exception:
            pass

    task_json = {
        "task_id": task_id,
        "main_epoch": main_epoch,
        "frobF": frobF,
        "eff_rank_0.99": eff_rank,
        "rank1_contribution": highest_energy,
        "err_rank1_stream_abs": err_rank1_stream_abs,
        "err_rank1opt_stream_abs": err_rank1opt_stream_abs,
        "err_diag_stream_abs": err_diag_stream_abs,
        "c_stream": float(c_stream),
        "ordered_top_eigenvalues": eigvals,
    }
    with open(exp_dir / f"{task_id}_{main_epoch}.json", "w") as f:
        f.write(json.dumps(task_json))
    print(
        f"[Fisher] Task {task_id} Epoch {main_epoch}: FrobF={frobF:.4f}, EffRank={eff_rank}, Rank1Ene={highest_energy:.4f}"
    )

    if getattr(args, "use_wandb", False):
        prefix = f"task/tid{task_id}_mep{main_epoch}"
        metrics = {
            f"{prefix}/analysis_step": main_epoch,
            f"{prefix}/fisher/frobF": frobF,
            f"{prefix}/fisher/eff_rank_0.99": eff_rank,
            f"{prefix}/fisher/rank1_contribution": highest_energy,
            f"{prefix}/fisher/err_rank1_stream_abs": err_rank1_stream_abs,
            f"{prefix}/fisher/err_rank1opt_stream_abs": err_rank1opt_stream_abs,
            f"{prefix}/fisher/err_diag_stream_abs": err_diag_stream_abs,
            f"{prefix}/fisher/c_stream": float(c_stream),
            f"{prefix}/fisher/ordered_top_eigenvalues": eigvals,
        }
        if cos_top_eig_mu is not None:
            metrics[f"{prefix}/fisher/cos_top_eig_mu"] = cos_top_eig_mu
        if cos_top_eig_prev is not None:
            metrics[f"{prefix}/fisher/cos_top_eig_prev"] = cos_top_eig_prev
        wandb.log(metrics)

    if getattr(args, "use_wandb", False):
        try:
            if len(top_eigvals_list) > 0 and len(top_eigvals_list[-1]) > 0:
                K = len(top_eigvals_list[-1])
                ranks = list(range(1, K + 1))
                vals = top_eigvals_list[-1]
                log_bar_chart(
                    "fisher_top_eigenvalues",
                    ranks,
                    vals,
                    xname="rank",
                    yname="eigenvalue",
                    prefix=f"task/tid{task_id}_mep{main_epoch}",
                )
        except Exception:
            pass
