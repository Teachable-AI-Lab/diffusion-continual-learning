import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import os
import gc
from src.parameter_scoring import *
from src.ewc import EWC
import src.utils as utils
import gc
from src.experiment_runner import *
from src.gr import GenerativeReplay
from src.ddim import build_conditional_ddim
from src.fisher_analysis import empirical_fisher_dense, optimal_rank1_coeff
import argparse
import wandb

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

#######################################################
parser = argparse.ArgumentParser(
    description="Continual Learning with Diffusion."
)
parser.add_argument(
    '--config',
    type=str,
    required=True,
    help='Path to the JSON configuration file.'
)
# Parse the command line to get the config file path
initial_args = parser.parse_args()

try:
    args = utils.load_config_from_json(initial_args.config)
except (FileNotFoundError, ValueError) as e:
    parser.error(str(e))
# --- Configuration is loaded into 'args' ---
print("Configuration loaded successfully:")
print("-" * 30)
# Print all loaded args and their types
for key, value in sorted(vars(args).items()): # Sort for consistent output
    print(f"{key} ({type(value).__name__}): {value}")

#######################################################
# initialize wandb
if args.use_wandb:
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        dir=args.output_dir,
        # reinit=True
    )
    # print("Initialized wandb with project:", args.wandb_project)

# Setup experiment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(args.output_dir)
ROOT.mkdir(exist_ok=True, parents=True)
print("Experiment logging directory:", ROOT)
# exit(0)

set_seed(args.seed)

# load datasets
print("Loading datasets...")
group_size = 50 if args.dataset == 'imagenet64' else 2
cl_train_loader, cl_test_loader, full_train_loader, full_test_loader = utils.get_cl_dataset(
        args.dataset, batch_size=args.batch_size, normalize=args.normalize, greyscale=args.greyscale,
        group_size=group_size, n_classes=args.num_classes
    )
im_size = full_train_loader.dataset[0][0].shape[1]
channels = full_train_loader.dataset[0][0].shape[0]
print("Image shape:", full_train_loader.dataset[0][0].shape)

# load model and initialize optimizer
print("Loading model...")
model = build_conditional_ddim(
    in_channel=channels,
    image_size=im_size,
    num_class_labels=args.num_classes,
    ewc_lambda=args.ewc_lambda,
    gr_kl=args.gr_kl,
    block_out_channels=(16,),
    down_block_types=("DownBlock2D",),
    up_block_types=("UpBlock2D",),
    norm_num_groups=8,
    layers_per_block=1
).to(device)
print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


# train model
#########################################################
print("Starting training...")
print("Continual learning on", len(cl_train_loader), "tasks.")
all_task_ids = list(range(len(cl_train_loader)))

ewc = None
gr = None
kl = args.use_distillation

for task_id in all_task_ids:
    c_stream, mu_stream, F_diag_stream = None, None, None
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Training on task {task_id}...")
    # if args.use_wandb:
        # wandb.log({"task_id": task_id})
    # Build experiment subdir name safely
    exp_path = f"{args.dataset}-{args.ewc_fisher_type}-{'gr' if args.use_generative_replay else ''}-{'distil' if args.use_distillation else ''}"
    train_loader = cl_train_loader[task_id]
    utils.train_one_task(model, train_loader, task_id, optimizer, 
                     ewc, 
                     gr,
                     kl,
                     args.epochs,
                     ROOT / exp_path,
                     device, wandb)
    # save model after each task
    model_path = ROOT / exp_path / f"model-task{task_id}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # ---------------------------------------------------------------
    # Fisher analysis: compute empirical Fisher and compare approximations
    # ---------------------------------------------------------------
    try:
        print(f"[Fisher] Computing empirical Fisher metrics for task {task_id} ...")
        # Use the current task's loader; wrap into a dict to reuse helper
        loaders_by_class = {0: train_loader}
        max_samples = getattr(args, "fisher_max_samples", None)

        # 1) Collect per-sample parameter scores (gradients)
        param_scores = compute_param_scores(
            model,
            loaders_by_class,
            device=device,
            target_class=0,
            max_samples=max_samples,
        )  # shape (B, D) on device

        B = param_scores.shape[0]
        F = None
        frobF = None
        frobF_sq = None

        if device.type == "cuda":
            try:
                F = empirical_fisher_dense(param_scores)  # (D, D)
                frobF = torch.linalg.norm(F).item()
                frobF_sq = frobF * frobF
            except Exception as _e:
                print(f"[Fisher] Dense Fisher failed ({_e}).")

        if frobF_sq is None:
            scale = 1.0 / math.sqrt(B)
            X = param_scores * scale                   # (B, D)
            K = X @ X.T                                # (B, B)
            frobF_sq = torch.sum(K.pow(2)).item()      # == ||F||_F^2
            frobF = math.sqrt(frobF_sq)

        # 2) Rank-1 stats from the same scores (for fairness)
        c_star, mu = optimal_rank1_coeff(param_scores, use_float64=False)
        mu = mu.to(param_scores.device, dtype=param_scores.dtype)
        v_norm2 = torch.dot(mu, mu).item()
        proj = (param_scores @ mu)
        vFv = torch.mean(proj.pow(2)).item()      # μ^T F μ = E[(μ^T g)^2]

        # 2b) Streaming rank-1 stats (two-pass, memory-friendly)
        c_stream, mu_stream, F_diag_stream = compute_rank1_coeff_and_mean(
            model, train_loader, device=device, max_samples=max_samples
        )
        mu_stream = mu_stream.to(param_scores.device, dtype=param_scores.dtype)
        c_stream_f = float(c_stream)
        v_norm2_stream = torch.dot(mu_stream, mu_stream).item()
        proj_stream = (param_scores @ mu_stream)
        vFv_stream = torch.mean(proj_stream.pow(2)).item()
        # Alignment between optimal μ and streaming μ
        denom = (v_norm2_stream ** 0.5) * (v_norm2 ** 0.5) + 1e-12
        mu_cos_sim = float(torch.dot(mu_stream, mu).item() / denom)

        # 3) Compute errors
        if F is not None:
            F_diag_mat = torch.diag(torch.diag(F))
            F_rank1 = mu.unsqueeze(1) @ mu.unsqueeze(0)
            F_rank1_opt = F_rank1 * c_star.to(F.dtype)

            err_diag_abs = torch.linalg.norm(F - F_diag_mat).item()
            err_rank1_abs = torch.linalg.norm(F - F_rank1).item()
            err_rank1opt_abs = torch.linalg.norm(F - F_rank1_opt).item()

            # Streaming variants against dense F
            F_rank1_stream = mu_stream.unsqueeze(1) @ mu_stream.unsqueeze(0)
            F_rank1opt_stream = F_rank1_stream * c_stream.to(F.dtype)
            err_rank1_stream_abs = torch.linalg.norm(F - F_rank1_stream).item()
            err_rank1opt_stream_abs = torch.linalg.norm(F - F_rank1opt_stream).item()
        else:
            # Diagonal Fisher vector from scores (exact)
            F_diag_vec = param_scores.pow(2).mean(dim=0)  # (D,)
            diagF_frob_sq = torch.sum(F_diag_vec.pow(2)).item()

            # Frobenius errors via identities
            err_rank1_sq = frobF_sq - 2.0 * vFv + (v_norm2 ** 2)
            err_rank1opt_sq = frobF_sq - 2.0 * float(c_star) * vFv + (float(c_star) ** 2) * (v_norm2 ** 2)
            err_diag_sq = max(frobF_sq - diagF_frob_sq, 0.0)

            err_rank1_abs = math.sqrt(max(err_rank1_sq, 0.0))
            err_rank1opt_abs = math.sqrt(max(err_rank1opt_sq, 0.0))
            err_diag_abs = math.sqrt(err_diag_sq)

            # Streaming errors via identities
            err_rank1_stream_sq = frobF_sq - 2.0 * vFv_stream + (v_norm2_stream ** 2)
            err_rank1opt_stream_sq = frobF_sq - 2.0 * c_stream_f * vFv_stream + (c_stream_f ** 2) * (v_norm2_stream ** 2)
            err_rank1_stream_abs = math.sqrt(max(err_rank1_stream_sq, 0.0))
            err_rank1opt_stream_abs = math.sqrt(max(err_rank1opt_stream_sq, 0.0))

        err_rank1_rel = err_rank1_abs / (frobF + 1e-12)
        err_rank1opt_rel = err_rank1opt_abs / (frobF + 1e-12)
        err_diag_rel = err_diag_abs / (frobF + 1e-12)
        err_rank1_stream_rel = err_rank1_stream_abs / (frobF + 1e-12)
        err_rank1opt_stream_rel = err_rank1opt_stream_abs / (frobF + 1e-12)

        print(
            f"[Fisher][task {task_id}] ||F||_F={frobF:.4f} | "
            f"diag abs/rel={err_diag_abs:.4f}/{err_diag_rel:.4f} | "
            f"rank1 abs/rel={err_rank1_abs:.4f}/{err_rank1_rel:.4f} | "
            f"rank1* abs/rel={err_rank1opt_abs:.4f}/{err_rank1opt_rel:.4f} (c*={float(c_star):.4e})\n"
            f"[stream] rank1 abs/rel={err_rank1_stream_abs:.4f}/{err_rank1_stream_rel:.4f} | "
            f"rank1* abs/rel={err_rank1opt_stream_abs:.4f}/{err_rank1opt_stream_rel:.4f} | "
            f"cos(mu_opt, mu_stream)={mu_cos_sim:.4f}"
        )

        # Persist small summary per task
        (ROOT / exp_path).mkdir(parents=True, exist_ok=True)
        summary_path = ROOT / exp_path / f"fisher_metrics_task{task_id}.txt"
        with open(summary_path, "w") as f:
            f.write(
                "\n".join([
                    f"frobF: {frobF}",
                    f"err_diag_abs: {err_diag_abs}",
                    f"err_diag_rel: {err_diag_rel}",
                    f"err_rank1_abs: {err_rank1_abs}",
                    f"err_rank1_rel: {err_rank1_rel}",
                    f"err_rank1opt_abs: {err_rank1opt_abs}",
                    f"err_rank1opt_rel: {err_rank1opt_rel}",
                    f"err_rank1_stream_abs: {err_rank1_stream_abs}",
                    f"err_rank1_stream_rel: {err_rank1_stream_rel}",
                    f"err_rank1opt_stream_abs: {err_rank1opt_stream_abs}",
                    f"err_rank1opt_stream_rel: {err_rank1opt_stream_rel}",
                    f"c_star: {float(c_star)}",
                    f"c_stream: {float(c_stream_f)}",
                    f"mu_cos_sim: {mu_cos_sim}",
                    f"num_param: {int(param_scores.shape[1])}",
                    f"num_samples: {int(param_scores.shape[0])}",
                ])
            )

        # Plot comparisons: absolute and relative errors per approximation
        try:
            labels = [
                "diag",
                "rank1",
                "rank1*",
                "rank1_stream",
                "rank1*_stream",
            ]
            abs_vals = [
                err_diag_abs,
                err_rank1_abs,
                err_rank1opt_abs,
                err_rank1_stream_abs,
                err_rank1opt_stream_abs,
            ]
            rel_vals = [
                err_diag_rel,
                err_rank1_rel,
                err_rank1opt_rel,
                err_rank1_stream_rel,
                err_rank1opt_stream_rel,
            ]

            import matplotlib.pyplot as plt

            # Absolute error bar chart
            fig_abs, ax_abs = plt.subplots(figsize=(7, 4))
            ax_abs.bar(labels, abs_vals, color=["#6baed6", "#9ecae1", "#3182bd", "#bcbddc", "#756bb1"])
            ax_abs.set_ylabel("Frobenius error (abs)")
            ax_abs.set_title(f"Fisher approx errors (abs) – task {task_id}")
            ax_abs.grid(True, axis='y', alpha=0.3)
            fig_abs.tight_layout()
            abs_path = ROOT / exp_path / f"fisher_errors_abs_task{task_id}.png"
            fig_abs.savefig(abs_path)
            plt.close(fig_abs)

            # Relative error bar chart
            fig_rel, ax_rel = plt.subplots(figsize=(7, 4))
            ax_rel.bar(labels, rel_vals, color=["#6baed6", "#9ecae1", "#3182bd", "#bcbddc", "#756bb1"])
            ax_rel.set_ylabel("Frobenius error (rel)")
            ax_rel.set_title(f"Fisher approx errors (rel) – task {task_id}")
            ax_rel.grid(True, axis='y', alpha=0.3)
            fig_rel.tight_layout()
            rel_path = ROOT / exp_path / f"fisher_errors_rel_task{task_id}.png"
            fig_rel.savefig(rel_path)
            plt.close(fig_rel)

            if args.use_wandb:
                wandb.log({
                    f"fisher/task{task_id}/plot_abs": wandb.Image(str(abs_path)),
                    f"fisher/task{task_id}/plot_rel": wandb.Image(str(rel_path)),
                })
        except Exception as _plot_e:
            print(f"[Fisher] Plotting skipped due to error: {_plot_e}")

        if args.use_wandb:
            wandb.log({
                f"fisher/task{task_id}/frobF": frobF,
                f"fisher/task{task_id}/err_diag_abs": err_diag_abs,
                f"fisher/task{task_id}/err_diag_rel": err_diag_rel,
                f"fisher/task{task_id}/err_rank1_abs": err_rank1_abs,
                f"fisher/task{task_id}/err_rank1_rel": err_rank1_rel,
                f"fisher/task{task_id}/err_rank1opt_abs": err_rank1opt_abs,
                f"fisher/task{task_id}/err_rank1opt_rel": err_rank1opt_rel,
                f"fisher/task{task_id}/c_star": float(c_star),
                f"fisher/task{task_id}/err_rank1_stream_abs": err_rank1_stream_abs,
                f"fisher/task{task_id}/err_rank1_stream_rel": err_rank1_stream_rel,
                f"fisher/task{task_id}/err_rank1opt_stream_abs": err_rank1opt_stream_abs,
                f"fisher/task{task_id}/err_rank1opt_stream_rel": err_rank1opt_stream_rel,
                f"fisher/task{task_id}/c_stream": float(c_stream_f),
                f"fisher/task{task_id}/mu_cos_sim": mu_cos_sim,
                f"fisher/task{task_id}/num_param": int(param_scores.shape[1]),
                f"fisher/task{task_id}/num_samples": int(param_scores.shape[0]),
            })

        try:
            del F
            del F_diag_mat
            del F_rank1
            del F_rank1_opt
            del F_rank1_stream
            del F_rank1opt_stream
        except NameError:
            pass
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"[Fisher] Skipping Fisher analysis for task {task_id} due to error: {e}")

    # test fid on each of previous tasks
    fids = []
    for eval_task_id in range(task_id + 1):
        fid = evaluate_fid(model, cl_test_loader[eval_task_id], device)
        print(f"Task {task_id}, Task {eval_task_id}, FID: {fid:.2f}")
        fids.append(fid)
        # TODO: log to wandb
        if args.use_wandb:
            wandb.log({
                # "task_id": task_id,
                "eval/task_id": eval_task_id,
                # f"fid-class{eval_task_id}": fid,
                f"fid-task{eval_task_id}": fid,
            })
    avg_fid = sum(fids) / len(fids) # this is the average fid over all seen classes so far
    print(f"Average FID after task {task_id}: {avg_fid:.2f}")
    if args.use_wandb:
        wandb.log({
            "task_id": task_id,
            "avg_fid": avg_fid,
        })
    
    # skip if last task
    if task_id == all_task_ids[-1]:
        print("Last task reached, skipping EWC and GR updates.")
        break

    # adding continual learning components
    if args.use_ewc:
        if ewc is None:
            # create a new EWC object
            fisher_type = args.ewc_fisher_type
            if (c_stream is not None and mu_stream is not None and F_diag_stream is not None):
                c, mu, diag = c_stream, mu_stream, F_diag_stream
            else:
                c, mu, diag = compute_rank1_coeff_and_mean(
                    model, train_loader, device=device, max_samples=None#500
                )
            # save the fisher information too
            torch.save((c, mu, diag), ROOT / exp_path / f"fisher-task{task_id}.pt")

            frozen_model = utils.freeze_model(model)
            ewc = EWC(frozen_model, fisher_type, c=c, mu=mu, diag=diag)
        else:
            # add a new task to the existing EWC object
            c, mu, diag = compute_rank1_coeff_and_mean(
                model, train_loader, device=device, max_samples=None#500
            )
            # save the fisher information too
            torch.save((c, mu, diag), ROOT / exp_path / f"fisher-task{task_id}.pt")

            frozen_model = utils.freeze_model(model)
            ewc.add_task(frozen_model, c=c, mu=mu, diag=diag)

    if args.use_generative_replay:
        if gr is None:
            frozen_model = utils.freeze_model(model)
            gr = GenerativeReplay(frozen_model, old_classes=list(range((task_id + 1)*2)), 
                                 alpha=args.gr_alpha, 
                                 batch_size=args.batch_size, 
                                 pool_size_per_class=args.gr_pool_size_per_class,
                                 num_inference_steps=args.gr_num_inference_steps,
                                 eta=args.gr_eta,
                                 seed=args.seed,
                                 device=device)
        else:
            frozen_model = utils.freeze_model(model)
            gr.update_teacher(frozen_model, old_classes=list(range((task_id + 1)*2)))
