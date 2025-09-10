import argparse
import json
from pathlib import Path

import torch
torch.backends.cuda.preferred_linalg_library("magma")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import wandb

import src.utils as utils
from src.experiment_runner import evaluate_fid
from src.parameter_scoring import compute_param_scores, compute_rank1_coeff_and_mean
from src.fisher_analysis import empirical_fisher_dense


from analysis.common import set_seed, get_model, log_bar_chart


def compute_fisher_spectrum(param_scores: torch.Tensor, need_vectors: bool = False):
    """Compute Fisher spectrum using full Fisher matrix when vectors are needed.

    If need_vectors is False, compute spectrum via BxB matrix (S S^T / B).

    Returns (eigvals_sorted, eigvecs) where eigvecs is None when need_vectors=False.
    eigvals are sorted descending.
    """
    assert param_scores.ndim == 2, "param_scores must be (B, D)"
    B = param_scores.shape[0]
    if param_scores.dtype in (torch.float16, torch.bfloat16):
        param_scores = param_scores.float()
    param_scores = (param_scores + 1e-12).contiguous()

    if need_vectors:
        # F = (1/B) S^T S  in R^{D x D}
        F = empirical_fisher_dense(param_scores)
        # symmetric PSD -> use eigh
        evals, evecs = torch.linalg.eigh(F)
        # Sort descending
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]
        return evals, evecs
    else:
        # K = (1/B) S S^T  in R^{B x B}; eigenvalues match non-zero eigenvalues of F
        K = (param_scores @ param_scores.T) / float(B)
        evals, _ = torch.linalg.eigh(K.contiguous())
        evals, _ = torch.sort(evals, descending=True)
        return evals, None


def main():
    parser = argparse.ArgumentParser(description="Run Fisher analysis using dense Fisher matrix or BxB fallback")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment subfolder name inside output_dir where models are saved. Defaults to '<dataset>-fisher-only'",
    )
    parser.add_argument(
        "--epochs_per_task",
        type=int,
        default=None,
        help="Number of main epochs per task used during training; if None, inferred by scanning files.",
    )
    parser.add_argument(
        "--need_vectors",
        action="store_true",
        help="If set, compute eigenvectors by forming the full Fisher matrix (D x D).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="How many top eigenvalues/eigenvectors to record and log.",
    )
    initial_args = parser.parse_args()

    try:
        args = utils.load_config_from_json(initial_args.config)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    exp_name = initial_args.exp_name or f"{args.wandb_run_name}-fisher-only"
    exp_dir = out_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_run_name}-fisher-dense" or ("fisher-dense-vectors" if initial_args.need_vectors else "fisher-bxb"),
            config=vars(args),
            dir=args.output_dir,
        )
        try:
            wandb.define_metric("task_id")
            wandb.define_metric("fisher/*", step_metric="task_id")
            wandb.define_metric("eval/*", step_metric="task_id")
        except Exception:
            pass

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

    # Determine tasks and available checkpoints
    all_task_ids = list(range(len(cl_train_loader)))

    # Infer epochs per task if not provided
    epochs_per_task = initial_args.epochs_per_task
    if epochs_per_task is None:
        epochs = []
        for ep in range(0, 1000):
            if (exp_dir / f"model-task0_{ep}.pt").exists():
                epochs.append(ep)
            else:
                break
        if not epochs:
            raise FileNotFoundError(
                f"No checkpoints found in {exp_dir}. Expected files like model-task0_0.pt"
            )
        epochs_per_task = len(epochs)

    fisher_max_samples = getattr(args, "fisher_max_samples", None)

    for task_id in all_task_ids:
        if args.use_wandb:
            try:
                wandb.define_metric(f"task/{task_id}/analysis_step")
                wandb.define_metric(f"task/{task_id}/*", step_metric=f"task/{task_id}/analysis_step")
            except Exception:
                pass

        for main_epoch in range(epochs_per_task):
            ckpt_path = exp_dir / f"model-task{task_id}_{main_epoch}.pt"
            if not ckpt_path.exists():
                print(f"[WARN] Missing checkpoint {ckpt_path}, skipping")
                continue

            model = get_model(args, channels, im_size, device)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.eval()

            # Build param-scores matrix S (B x D)
            train_loader = cl_train_loader[task_id]
            loaders_by_class = {0: train_loader}
            param_scores = compute_param_scores(
                model,
                loaders_by_class,
                device=device,
                target_class=0,
                max_samples=fisher_max_samples,
                time_level=None,
            )
            if not torch.isfinite(param_scores).all():
                print(f"[FisherDense] Non-finite param_scores, fixing with nan_to_num for task {task_id}")
                param_scores = torch.nan_to_num(param_scores)

            evals, evecs = compute_fisher_spectrum(param_scores, need_vectors=initial_args.need_vectors)

            # Compute metrics from spectrum
            with torch.no_grad():
                energies = evals.pow(2)
                total_energy = float(energies.sum().item()) if energies.numel() > 0 else 0.0
                frobF = float(torch.sqrt(energies.sum() + 1e-20).item()) if energies.numel() > 0 else 0.0
                energies_sorted = energies  # already sorted
                highest_energy = float(energies_sorted[0].item() / total_energy) if total_energy > 0 else 0.0

                # effective rank based on energy threshold
                energy_threshold = getattr(args, "fisher_energy_threshold", 0.99)
                cumsum = torch.cumsum(energies_sorted, dim=0)
                target = float(energy_threshold) * float(energies_sorted.sum().item())
                idx = torch.searchsorted(cumsum, torch.tensor(target, device=cumsum.device))
                eff_rank = int(min(idx.item() + 1, energies_sorted.numel())) if energies_sorted.numel() > 0 else 0

                # top-k eigenvalues list for logging
                k = min(initial_args.top_k, evals.numel())
                top_eigvals = [float(v) for v in evals[:k].tolist()]
                if k < initial_args.top_k:
                    top_eigvals += [0.0] * (initial_args.top_k - k)

            # Rank-1 and diag approximation errors (reuse streaming stats)
            c_stream, mu_stream, F_diag_stream = compute_rank1_coeff_and_mean(
                model, train_loader, device=device, max_samples=fisher_max_samples
            )
            mu_stream = mu_stream.to(param_scores.device, dtype=param_scores.dtype)
            v_norm2_stream = torch.dot(mu_stream, mu_stream).item()
            B = param_scores.shape[0]
            with torch.no_grad():
                Su_stream = param_scores @ mu_stream
                uTFu_stream = float(torch.sum(Su_stream.pow(2)) / float(B))
                err_cov_stream_abs = v_norm2_stream * v_norm2_stream
                frobF_sq = float(energies.sum().item()) if energies.numel() > 0 else 0.0
                err_rank1_stream_abs = (frobF_sq + err_cov_stream_abs - 2.0 * uTFu_stream)
                err_rank1_stream_abs = float(torch.sqrt(torch.tensor(max(err_rank1_stream_abs, 0.0))))
                c_stream_f = float(c_stream)
                err_rank1opt_stream_abs = (
                    frobF_sq + (c_stream_f ** 2) * (err_cov_stream_abs) - 2.0 * c_stream_f * uTFu_stream
                )
                err_rank1opt_stream_abs = float(torch.sqrt(torch.tensor(max(err_rank1opt_stream_abs, 0.0))))
                diag_stream_norm_sq = float(torch.dot(F_diag_stream, F_diag_stream))
                err_diag_stream_abs = float(torch.sqrt(torch.tensor(max(frobF_sq - diag_stream_norm_sq, 0.0))))

            # Optional cosine with mean grad using top eigenvector when available
            cos_top_eig_mu = None
            if evecs is not None and evecs.numel() > 0:
                top_vec = evecs[:, 0]
                try:
                    tvec = top_vec.to(mu_stream.device, dtype=mu_stream.dtype)
                    v_norm = torch.linalg.norm(tvec) + 1e-12
                    mu_norm = torch.linalg.norm(mu_stream) + 1e-12
                    cos_top_eig_mu = float(torch.dot(tvec, mu_stream) / (v_norm * mu_norm))
                except Exception:
                    pass

            # Persist results (separate file name to distinguish dense path)
            task_json = {
                "task_id": task_id,
                "main_epoch": main_epoch,
                "frobF": frobF,
                "eff_rank_0.99": eff_rank,
                "rank1_contribution": highest_energy,
                "err_rank1_stream_abs": err_rank1_stream_abs,
                "err_rank1opt_stream_abs": err_rank1opt_stream_abs,
                "err_diag_stream_abs": err_diag_stream_abs,
                "ordered_top_eigenvalues": top_eigvals,
                "used_dense_vectors": bool(initial_args.need_vectors),
            }
            (exp_dir / f"{task_id}_{main_epoch}_dense.json").write_text(json.dumps(task_json))
            print(
                f"[FisherDense] Task {task_id} Epoch {main_epoch}: FrobF={frobF:.4f}, EffRank={eff_rank}, TopK λ[0]={top_eigvals[0]:.4e}"
            )

            if args.use_wandb:
                prefix = f"task/tid{task_id}_mep{main_epoch}"
                metrics = {
                    f"{prefix}/analysis_step": main_epoch,
                    f"{prefix}/fisher_dense/frobF": frobF,
                    f"{prefix}/fisher_dense/eff_rank_0.99": eff_rank,
                    f"{prefix}/fisher_dense/rank1_contribution": highest_energy,
                    f"{prefix}/fisher_dense/err_rank1_stream_abs": err_rank1_stream_abs,
                    f"{prefix}/fisher_dense/err_rank1opt_stream_abs": err_rank1opt_stream_abs,
                    f"{prefix}/fisher_dense/err_diag_stream_abs": err_diag_stream_abs,
                }
                if cos_top_eig_mu is not None:
                    metrics[f"{prefix}/fisher_dense/cos_top_eig_mu"] = cos_top_eig_mu
                wandb.log(metrics)

                # bar chart for eigenvalues
                try:
                    ranks = list(range(1, len(top_eigvals) + 1))
                    log_bar_chart(
                        "fisher_dense_top_eigenvalues",
                        ranks,
                        top_eigvals,
                        xname="rank",
                        yname="eigenvalue",
                        prefix=prefix,
                    )
                except Exception:
                    pass

            # Optional FID evaluation
            try:
                fid = evaluate_fid(model, cl_test_loader[task_id], device)
                if args.use_wandb:
                    prefix = f"task/tid{task_id}_mep{main_epoch}"
                    wandb.log({f"{prefix}/analysis_step": main_epoch, f"{prefix}/eval/fid": fid})
            except Exception:
                pass

    print("Dense Fisher analysis complete. Results stored in:", str(exp_dir))


if __name__ == "__main__":
    main()
