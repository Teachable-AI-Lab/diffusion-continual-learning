import argparse
import math
import os
from pathlib import Path

import numpy as np
import gc
import torch
import torch.optim as optim
import wandb

import src.utils as utils
from src.ddim import build_conditional_ddim
from src.experiment_runner import evaluate_fid
from src.fisher_analysis import empirical_fisher_dense, optimal_rank1_coeff
from src.parameter_scoring import (
    compute_param_scores,
    compute_rank1_coeff_and_mean,
)


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


def log_multi_line_chart(name: str, xs, series_dict, xname="task_id", yname=None):
    """Log a multi-series line chart to W&B. Falls back to per-series charts if needed."""
    if not wandb.run:
        return
    yname = yname or name
    keys = list(series_dict.keys())
    ys = [series_dict[k] for k in keys]
    try:
        chart = wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title=name, xname=xname, yname=yname)
        wandb.log({f"charts/{name}": chart})
        return
    except Exception:
        pass

    # Fallback 1: individual charts per series using wandb.plot.line
    any_logged = False
    for k, y in series_dict.items():
        try:
            table = wandb.Table(
                data=[[int(a), float(b)] for a, b in zip(xs, y) if b is not None],
                columns=[xname, yname],
            )
            chart = wandb.plot.line(table, x=xname, y=yname, title=f"{name} - {k}")
            wandb.log({f"charts/{name}/{k}": chart})
            any_logged = True
        except Exception:
            continue
    if any_logged:
        return

    # Fallback 2: static Matplotlib image
    try:
        import matplotlib  # type: ignore
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        for k, y in series_dict.items():
            xs_f, ys_f = [], []
            for a, b in zip(xs, y):
                if b is None:
                    continue
                xs_f.append(int(a))
                ys_f.append(float(b))
            if len(xs_f) == 0:
                continue
            ax.plot(xs_f, ys_f, label=str(k))
        ax.set_title(name)
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        try:
            ax.legend(loc="best", fontsize="small")
        except Exception:
            pass
        wandb.log({f"charts/{name}": wandb.Image(fig)})
        plt.close(fig)
    except Exception:
        # Give up silently if plotting fails entirely
        pass


def log_bar_chart(name: str, labels, values, xname="rank", yname="eigenvalue"):
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
        wandb.log({f"charts/{name}": chart})
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
        wandb.log({f"charts/{name}": wandb.Image(fig)})
        plt.close(fig)
        return
    except Exception:
        pass

    # Final fallback: histogram of values (distribution-only)
    try:
        wandb.log({f"charts/{name}_hist": wandb.Histogram(np.array(values, dtype=float))})
    except Exception:
        pass


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
    group_size = 50 if args.dataset == "imagenet64" else 2
    cl_train_loader, cl_test_loader, full_train_loader, full_test_loader = utils.get_cl_dataset(
        args.dataset,
        batch_size=args.batch_size,
        normalize=args.normalize,
        greyscale=args.greyscale,
        group_size=group_size,
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
    xs = []
    frobF_list = []
    err_diag_rel_list = []
    err_rank1_rel_list = []
    err_rank1opt_rel_list = []
    err_rank1_stream_rel_list = []
    err_rank1opt_stream_rel_list = []
    err_diag_stream_rel_list = []
    mu_cos_sim_list = []
    avg_fid_list = []
    eff_rank_list = []
    top_eigvals_list = []  # store, e.g., top-10 eigenvalues
    top_eig_mu_cos_list = []

    fisher_max_samples = getattr(args, "fisher_max_samples", None)
    fisher_dense = bool(getattr(args, "fisher_dense", True))

    for task_id in all_task_ids:
        # Fresh model for each task
        model = build_conditional_ddim(
            in_channel=channels,
            image_size=im_size,
            num_class_labels=args.num_classes,
            # Small UNet by default
            block_out_channels=(16,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            norm_num_groups=8,
            layers_per_block=1,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Train one task (no EWC, no GR, no distillation)
        train_loader = cl_train_loader[task_id]
        utils.train_one_task(
            model,
            train_loader,
            task_id,
            optimizer,
            ewc=None,
            gr=None,
            kl=False,
            num_epochs=args.epochs,
            device=device,
            wandb=wandb if args.use_wandb else None,
        )

        # Optimizer no longer needed beyond this point; free its state early
        try:
            optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        del optimizer

        # Save model snapshot per task
        torch.save(model.state_dict(), exp_dir / f"model-task{task_id}.pt")

        # -----------------------------------------------------------
        # Fisher analysis per task
        # -----------------------------------------------------------
        try:
            loaders_by_class = {0: train_loader}
            param_scores = compute_param_scores(
                model,
                loaders_by_class,
                device=device,
                target_class=0,
                max_samples=fisher_max_samples,
            )  # (B, D)

            B = param_scores.shape[0]
            # Compute SVD once and reuse everywhere; fallback to singular values-only if needed
            with torch.no_grad():
                try:
                    U_svd, S_svd, Vh_svd = torch.linalg.svd(param_scores, full_matrices=False)
                except Exception:
                    U_svd, Vh_svd = None, None
                    S_svd = torch.linalg.svdvals(param_scores)

                lambdas = (S_svd ** 2) / float(B)
                frobF_sq_t = torch.sum(lambdas ** 2)
                frobF = float(torch.sqrt(frobF_sq_t + 1e-20))
                frobF_sq = float(frobF_sq_t)

            # Optional: still attempt dense F if explicitly desired (not required)
            F = None
            if fisher_dense and device.type == "cuda":
                try:
                    F = empirical_fisher_dense(param_scores)  # (D, D)
                except Exception:
                    F = None

            # Effective rank and eigen spectrum (via SVD of param_scores)
            try:
                eff_rank = effective_rank_from_param_scores(
                    param_scores, energy_threshold=0.99, svals=S_svd
                )
                eff_rank_list.append(float(eff_rank))
            except Exception:
                eff_rank_list.append(None)

            # Top eigenvalues of F: λ_i = σ_i^2 / B (reuse svals)
            try:
                with torch.no_grad():
                    lambdas_sorted, _ = torch.sort(lambdas, descending=True)
                    k = min(10, lambdas_sorted.numel())
                    eigvals = [float(v) for v in lambdas_sorted[:k].tolist()]
                    if k < 10:
                        eigvals += [0.0] * (10 - k)
                    top_eigvals_list.append(eigvals)
            except Exception:
                top_eigvals_list.append([0.0] * 10)

            c_star, mu = optimal_rank1_coeff(param_scores, use_float64=False)
            mu = mu.to(param_scores.device, dtype=param_scores.dtype)
            v_norm2 = torch.dot(mu, mu).item()
            # proj = (param_scores @ mu)
            # vFv = torch.mean(proj.pow(2)).item()

            # Streaming variants
            c_stream, mu_stream, F_diag_stream = compute_rank1_coeff_and_mean(
                model, train_loader, device=device, max_samples=fisher_max_samples
            )
            mu_stream = mu_stream.to(param_scores.device, dtype=param_scores.dtype)
            v_norm2_stream = torch.dot(mu_stream, mu_stream).item()
            # proj_stream = (param_scores @ mu_stream)
            # vFv_stream = torch.mean(proj_stream.pow(2)).item()
            ## Compute similarity between true mean and calculated mean
            denom = (v_norm2_stream ** 0.5) * (v_norm2 ** 0.5) + 1e-12
            mu_cos_sim = float(torch.dot(mu_stream, mu).item() / denom)

            # Memory-efficient error computations (no dense matrices)
            with torch.no_grad():
                # Diagonal-only approximation: ||F - Diag(diag(F))||_F^2 = ||F||_F^2 - ||diag(F)||_2^2
                diag_F_vec = torch.mean(param_scores.pow(2), dim=0)  # (D,)
                diag_norm_sq = float(torch.sum(diag_F_vec.pow(2)))
                err_diag_abs = math.sqrt(max(frobF_sq - diag_norm_sq, 0.0))

                # Rank-1 with mu: err^2 = ||F||_F^2 + ||uu^T||_F^2 - 2 u^T F u
                # where ||uu^T||_F^2 = ||u||^4 and u^T F u = (1/B) ||S u||^2
                Su = param_scores @ mu  # (B,)
                uTFu = float(torch.sum(Su.pow(2)) / float(B))
                err_rank1_abs = math.sqrt(max(frobF_sq + (v_norm2 ** 2) - 2.0 * uTFu, 0.0))

                # Rank-1 optimal scaling c*: err^2 = ||F||_F^2 + c^2 ||u||^4 - 2 c (u^T F u)
                c_star_f = float(c_star)
                err_rank1opt_abs = math.sqrt(
                    max(frobF_sq + (c_star_f ** 2) * (v_norm2 ** 2) - 2.0 * c_star_f * uTFu, 0.0)
                )

                # Streaming variants
                Su_stream = param_scores @ mu_stream
                uTFu_stream = float(torch.sum(Su_stream.pow(2)) / float(B))
                err_rank1_stream_abs = math.sqrt(
                    max(frobF_sq + (v_norm2_stream ** 2) - 2.0 * uTFu_stream, 0.0)
                )
                c_stream_f = float(c_stream)
                err_rank1opt_stream_abs = math.sqrt(
                    max(frobF_sq + (c_stream_f ** 2) * (v_norm2_stream ** 2) - 2.0 * c_stream_f * uTFu_stream, 0.0)
                )
                # Diagonal (streaming) uses provided F_diag_stream entries
                diag_stream_norm_sq = float(torch.dot(F_diag_stream, F_diag_stream))
                err_diag_stream_abs = math.sqrt(max(frobF_sq - diag_stream_norm_sq, 0.0))

            # Relative
            err_rank1_rel = err_rank1_abs / (frobF + 1e-12)
            err_rank1opt_rel = err_rank1opt_abs / (frobF + 1e-12)
            err_diag_rel = err_diag_abs / (frobF + 1e-12)
            err_rank1_stream_rel = err_rank1_stream_abs / (frobF + 1e-12)
            err_rank1opt_stream_rel = err_rank1opt_stream_abs / (frobF + 1e-12)
            err_diag_stream_rel = err_diag_stream_abs / (frobF + 1e-12)

            # Log scalars for this task
            if args.use_wandb:
                wandb.log(
                    {
                        "task_id": task_id,
                        "fisher/frobF": frobF,
                        "fisher/eff_rank_0.99": eff_rank_list[-1] if len(eff_rank_list) > 0 else None,
                        "fisher/err_diag_rel": err_diag_rel,
                        "fisher/err_rank1_rel": err_rank1_rel,
                        "fisher/err_rank1opt_rel": err_rank1opt_rel,
                        "fisher/err_rank1_stream_rel": err_rank1_stream_rel,
                        "fisher/err_rank1opt_stream_rel": err_rank1opt_stream_rel,
                        "fisher/err_diag_stream_rel": err_diag_stream_rel,
                        "fisher/mu_cos_sim": mu_cos_sim,
                        "fisher/c_star": float(c_star),
                        "fisher/c_stream": float(c_stream),
                    }
                )

            # Add to cross-task arrays
            xs.append(task_id)
            frobF_list.append(frobF)
            err_diag_rel_list.append(err_diag_rel)
            err_rank1_rel_list.append(err_rank1_rel)
            err_rank1opt_rel_list.append(err_rank1opt_rel)
            err_rank1_stream_rel_list.append(err_rank1_stream_rel)
            err_rank1opt_stream_rel_list.append(err_rank1opt_stream_rel)
            err_diag_stream_rel_list.append(err_diag_stream_rel)
            mu_cos_sim_list.append(mu_cos_sim)
            # Cosine similarity between top eigenvector of F and mu
            # Top right singular vector of S corresponds to top eigenvector of F
            if Vh_svd is not None:
                v1 = Vh_svd[0]  # (D,)
                v1 = v1.to(mu.device, dtype=mu.dtype)
                denom = (torch.linalg.norm(v1) * torch.linalg.norm(mu) + 1e-12)
                cos_v1_mu = float(torch.dot(v1, mu).item() / denom)
                top_eig_mu_cos_list.append(cos_v1_mu)
                if args.use_wandb:
                    wandb.log({"task_id": task_id, "fisher/top_eigvec_mu_cos": cos_v1_mu})
            else:
                top_eig_mu_cos_list.append(None)

            # Update charts per task
            if args.use_wandb:
                log_multi_line_chart(
                    "fisher_rel_error_exact",
                    xs,
                    {
                        "diag": err_diag_rel_list,
                        "rank1": err_rank1_rel_list,
                        "rank1*": err_rank1opt_rel_list,
                    },
                    yname="Relative error",
                )
                log_multi_line_chart(
                    "fisher_rel_error_stream",
                    xs,
                    {
                        "diag_stream": err_diag_stream_rel_list,
                        "rank1_stream": err_rank1_stream_rel_list,
                        "rank1*_stream": err_rank1opt_stream_rel_list,
                    },
                    yname="Relative error",
                )
                # Per-task histogram/bar of sorted top eigenvalues
                try:
                    if len(top_eigvals_list) > 0 and len(top_eigvals_list[-1]) > 0:
                        K = len(top_eigvals_list[-1])
                        ranks = list(range(1, K + 1))
                        vals = top_eigvals_list[-1]
                        log_bar_chart(f"fisher_top_eigenvalues_for_task_{task_id}", ranks, vals, xname="rank", yname="eigenvalue")
                except Exception:
                    pass
        except Exception as e:
            print(f"[Fisher] Skipping Fisher analysis for task {task_id} due to error: {e}")

        # -----------------------------------------------------------
        # Evaluate FID on all seen tasks so far
        # -----------------------------------------------------------
        fids = []
        for eval_task_id in range(task_id + 1):
            fid = evaluate_fid(model, cl_test_loader[eval_task_id], device)
            fids.append(fid)
            if args.use_wandb:
                wandb.log({"task_id": task_id, f"eval/fid_task_{eval_task_id}": fid})
        avg_fid = float(sum(fids) / len(fids))
        if args.use_wandb:
            wandb.log({"task_id": task_id, "eval/avg_fid": avg_fid})
        # ------------------------------
        # Cleanup GPU memory to start fresh next loop
        # ------------------------------
        for _var in [
            'param_scores','F','F_diag_stream','mu','mu_stream','F_diag_mat','F_rank1','F_rank1_opt',
            'F_rank1_stream','F_rank1opt_stream','c_star','c_stream',
            # Added SVD and intermediate tensors for cleanup
            'U_svd','S_svd','Vh_svd','lambdas','lambdas_sorted','Su','Su_stream','diag_F_vec'
        ]:
            try:
                if _var in locals():
                    del locals()[_var]
            except Exception:
                pass
        try:
            del model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    print("Done.")


if __name__ == "__main__":
    main()
