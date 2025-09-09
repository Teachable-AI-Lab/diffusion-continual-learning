import argparse
import math
import os
from pathlib import Path

import numpy as np
import gc
import torch
import torch.optim as optim
import wandb
import json

import src.utils as utils
from src.ddim import build_conditional_ddim
from src.experiment_runner import evaluate_fid
from src.fisher_analysis import empirical_fisher_dense, optimal_rank1_coeff
from src.parameter_scoring import (
    compute_param_scores,
    compute_rank1_coeff_and_mean,
)

# Cache of previous top eigenvectors per task for comparison across analyses
_prev_top_eigvec = {}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Python's random may be used internally by some utils
    import random

    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def log_bar_chart(name: str, labels, values, xname="rank", yname="eigenvalue", prefix=None):
    """Log a bar chart to W&B, falling back gracefully if bar isn't available.

    labels: x-axis categories (e.g., [1, 2, 3, ...])
    values: heights per bar (floats)
    """
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
            # Fallback to a line plot if bar is unavailable
            chart = wandb.plot.line(table, x=xname, y=yname, title=name)
        key = f"charts/{name}" if not prefix else f"{prefix}/charts/{name}"
        wandb.log({key: chart})
        return
    except Exception:
        pass

    # Fallback: static Matplotlib bar image
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

    # Final fallback: histogram of values (distribution-only)
    try:
        key = f"charts/{name}_hist" if not prefix else f"{prefix}/charts/{name}_hist"
        wandb.log({key: wandb.Histogram(np.array(values, dtype=float))})
    except Exception:
        pass

def get_model_size(model):
    """Get the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(args, channels, im_size, device):
    """Create a diffusion model based on args."""
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
            block_out_channels=(32,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            norm_num_groups=8,
            layers_per_block=1,
        ).to(device)
    elif model_size == "big":
        model = build_conditional_ddim(
            in_channel=channels,
            image_size=im_size,
            num_class_labels=args.num_classes
        ).to(device)
    print(f"Model size: {get_model_size(model)} parameters")
    return model


def effective_rank_from_param_scores(
    param_scores: torch.Tensor, energy_threshold: float = 0.99, svals=None
) -> int:
    """Compute effective rank of F ~= (1/B) S^T S using spectrum of S=param_scores (B x D).

    If ``svals`` (the singular values of ``param_scores``) are provided, reuse them to
    avoid recomputing SVD. Energy is defined on the spectrum of F (eigenvalues λ_i = σ_i^2 / B)
    using sum of λ_i^2. Returns the min r s.t. cumulative energy >= threshold * total_energy.
    """
    if param_scores is None or param_scores.ndim != 2:
        raise ValueError("param_scores must be a 2D tensor (B, D)")
    B = param_scores.shape[0]
    if B == 0:
        return 0
    with torch.no_grad():
        # Singular values of S (length k=min(B,D))
        if svals is None:
            svals = torch.linalg.svdvals(param_scores)
        # Eigenvalues of F
        lambdas = (svals ** 2) / float(B)
        # Sort descending by energy contribution (λ^2)
        energies = (lambdas ** 2)
        if energies.numel() == 0:
            return 0
        total_energy = energies.sum()
        if torch.isclose(total_energy, torch.tensor(0.0, device=energies.device)):
            return 0
        energies_sorted, _ = torch.sort(energies, descending=True)
        cumsum = torch.cumsum(energies_sorted, dim=0)
        target = float(energy_threshold) * float(total_energy)
        # searchsorted to find first index where cumsum >= target
        idx = torch.searchsorted(cumsum, torch.tensor(target, device=cumsum.device))
        rank = int(min(idx.item() + 1, energies_sorted.numel()))
        return rank



def fisher_analysis(model, args, train_loader, device, task_id, main_epoch, exp_dir, time_level=None):
    """Compute empirical Fisher information matrix and related statistics. """
    eff_rank_list = []
    rank1_contribution = []
    top_eigvals_list = []  # store, e.g., top-10 eigenvalues
    fisher_max_samples = getattr(args, "fisher_max_samples", None)
    energy_threshold = getattr(args, "fisher_energy_threshold", 0.99)


    loaders_by_class = {0: train_loader}
    param_scores = compute_param_scores(
        model,
        loaders_by_class,
        device=device,
        target_class=0,
        max_samples=fisher_max_samples,
        time_level=time_level
    )
    # Ensure dtype supported by CUDA SVD and contiguity
    if param_scores.dtype in (torch.float16, torch.bfloat16):
        param_scores = param_scores.float()
    param_scores = (param_scores + 1e-12).contiguous()  # (B, D)

    if not torch.isfinite(param_scores).all():
        print(f"[Fisher] Warning: Non-finite param_scores, fixing with nan_to_num for task {task_id}")
        param_scores = torch.nan_to_num(param_scores)

    B = param_scores.shape[0]
    eigvec_flag = bool(getattr(args, "eigvec_analysis", False))
    top_vec = None
    # Compute SVD once and reuse everywhere; compute full SVD only if needed
    with torch.no_grad():
        if eigvec_flag:
            _U, S_svd, Vh_full = torch.linalg.svd(param_scores, full_matrices=False)
            top_vec = Vh_full[0, :]
        else:
            S_svd = torch.linalg.svdvals(param_scores)
        lambdas = (S_svd ** 2) / float(B)
        frobF_sq_t = torch.sum(lambdas ** 2)
        frobF = float(torch.sqrt(frobF_sq_t + 1e-20))
        frobF_sq = float(frobF_sq_t)
        # Sort descending by energy contribution (λ^2)
        energies = (lambdas ** 2)
        total_energy = energies.sum()
        energies_sorted, _ = torch.sort(energies, descending=True)
        cumsum = torch.cumsum(energies_sorted, dim=0)
        target = float(energy_threshold) * float(total_energy)
        idx = torch.searchsorted(cumsum, torch.tensor(target, device=cumsum.device))
        eff_rank = int(min(idx.item() + 1, energies_sorted.numel()))
        highest_energy = energies_sorted[0].item()/float(total_energy) if total_energy.item() > 0 else 0.0


    # Optional: still attempt dense F if explicitly desired (not required)
    # F = None
    if getattr(args, "check_fisher", False) and device.type == "cuda":
        try:
            F = empirical_fisher_dense(param_scores)  # (D, D)
            norm_f = torch.linalg.norm(F, ord="fro").item()
            if not math.isclose(norm_f, frobF, rel_tol=1e-3):
                print(f"[Fisher] Warning: Frobenius norm mismatch ||F||_F={norm_f} vs {frobF}")
            del F
            del norm_f
        except Exception:
            print(f"[Fisher] Warning: empirical_fisher_dense failed, skipping dense F computation")
            pass

    eff_rank_list.append(float(eff_rank))
    rank1_contribution.append(float(highest_energy))

    # Top eigenvalues of F: λ_i = σ_i^2 / B (reuse svals)
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

    # Memory-efficient error computations (no dense matrices)
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
        # Diagonal (streaming) uses provided F_diag_stream entries
        diag_stream_norm_sq = float(torch.dot(F_diag_stream, F_diag_stream))
        err_diag_stream_abs = math.sqrt(max(frobF_sq - diag_stream_norm_sq, 0.0))

    # Optional: top eigenvector analysis (right singular vector of S)
    cos_top_eig_mu = None
    cos_top_eig_prev = None
    if eigvec_flag and top_vec is not None:
        try:
            tvec = top_vec.to(mu_stream.device, dtype=mu_stream.dtype)
            # Cosine with mean grad direction
            v_norm = torch.linalg.norm(tvec) + 1e-12
            mu_norm = torch.linalg.norm(mu_stream) + 1e-12
            cos_top_eig_mu = float(torch.dot(tvec, mu_stream) / (v_norm * mu_norm))
            # Cosine with previous top eigenvector for this task (if present)
            prev = _prev_top_eigvec.get(int(task_id))
            if prev is not None:
                prev = prev.to(tvec.device, dtype=tvec.dtype)
                pv = torch.linalg.norm(prev) + 1e-12
                cos_top_eig_prev = float(torch.dot(tvec, prev) / (v_norm * pv))
            # Update cache (store on CPU to save GPU mem)
            _prev_top_eigvec[int(task_id)] = tvec.detach().cpu()
        except Exception:
            pass

    # Log scalars for this task
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
    print(f"[Fisher] Task {task_id} Epoch {main_epoch}: " f"FrobF={frobF:.4f}, EffRank={eff_rank}, Rank1Ene={highest_energy:.4f}")

    if args.use_wandb:
        # Namespace logs per task and step by analysis iteration within task
        prefix = f"task/{task_id}"
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

    # Add to cross-task arrays
    if args.use_wandb:
        try:
            if len(top_eigvals_list) > 0 and len(top_eigvals_list[-1]) > 0:
                K = len(top_eigvals_list[-1])
                ranks = list(range(1, K + 1))
                vals = top_eigvals_list[-1]
                log_bar_chart(
                    "fisher_top_eigenvalues", ranks, vals, xname="rank", yname="eigenvalue", prefix=f"task/{task_id}"
                )
        except Exception:
            pass


def main():
    # ------------------------------
    # Parse config
    # ------------------------------
    parser = argparse.ArgumentParser(description="Baseline training + Fisher analysis (no CL regularizers)")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    initial_args = parser.parse_args()

    try:
        args = utils.load_config_from_json(initial_args.config)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))

    # ------------------------------
    # Setup
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            dir=args.output_dir,
        )
        # Ensure charts use task_id as the x-axis when possible
        try:
            wandb.define_metric("task_id")
            wandb.define_metric("fisher/*", step_metric="task_id")
            wandb.define_metric("eval/*", step_metric="task_id")
        except Exception:
            pass

    # ------------------------------
    # Data
    # ------------------------------
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

    # Model is created fresh per task inside the loop

    # ------------------------------
    # Train across tasks (from scratch baseline, sequential without regularizers)
    # ------------------------------
    exp_name = f"{args.dataset}-fisher-only"
    exp_dir = out_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    all_task_ids = list(range(len(cl_train_loader)))

    # For cross-task charts

    for task_id in all_task_ids:
        # Fresh model for each task
        model = get_model(args, channels, im_size, device)


        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Define per-task metric stepping so x-axis is main_epoch within each task
        if args.use_wandb:
            try:
                wandb.define_metric(f"task/{task_id}/analysis_step")
                wandb.define_metric(f"task/{task_id}/*", step_metric=f"task/{task_id}/analysis_step")
            except Exception:
                pass

        for main_epoch in range(args.main_epochs):
            print(f"=== Main Epoch {main_epoch + 1}/{args.main_epochs} for Task {task_id} ===")
            # Train on current task
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

            # -----------------------------------------------------------
            # Fisher analysis per task
            # -----------------------------------------------------------
            fisher_analysis(
                model,
                args,
                train_loader,
                device,
                task_id,
                main_epoch=main_epoch,
                exp_dir=exp_dir,
            )

            # -----------------------------------------------------------
            # Evaluate FID on the current task
            # -----------------------------------------------------------
            fid = evaluate_fid(model, cl_test_loader[task_id], device)
            if args.use_wandb:
                prefix = f"task/{task_id}"
                wandb.log({f"{prefix}/analysis_step": main_epoch, f"{prefix}/eval/fid": fid})

    print("Done.")


if __name__ == "__main__":
    main()
