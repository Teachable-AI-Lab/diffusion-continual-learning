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
from src.parameter_scoring import *
from src.ewc import EWC
import src.utils as utils
from src.experiment_runner import *
from src.gr import GenerativeReplay
from src.ddim import build_conditional_ddim
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

## default args
args.power_iters = getattr(args, 'power_iters', 1)
args.check_fisher = getattr(args, 'check_fisher', False)

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
# group_size = 2
# if args.dataset == "cifar100":
#     group_size = 5
# if args.dataset == "imagenet64":
#     group_size = 50
print(f"Using group size of {args.group_size} for dataset {args.dataset}.")
cl_train_loader, cl_test_loader, full_train_loader, full_test_loader = utils.get_cl_dataset(
        args.dataset, batch_size=args.batch_size, normalize=args.normalize, greyscale=args.greyscale,
        group_size=args.group_size, n_classes=args.num_classes
    )
im_size = full_train_loader.dataset[0][0].shape[1]
channels = full_train_loader.dataset[0][0].shape[0]
print("Image shape:", full_train_loader.dataset[0][0].shape)

# load model and initialize optimizer
print("Loading model...")

if getattr(args, 'model_size', "big") == "big":
    model = build_conditional_ddim(
        in_channel=channels,
        image_size=im_size,
        num_class_labels=args.num_classes,
        ewc_lambda=args.ewc_lambda,
        gr_kl=args.gr_kl).to(device)
else:
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
        layers_per_block=1,
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Training on task {task_id}...")
    # if args.use_wandb:
        # wandb.log({"task_id": task_id})
    # Build experiment path segments robustly
    _exp_segments = [str(args.dataset), str(args.ewc_fisher_type)]
    if args.use_generative_replay:
        _exp_segments.append("gr")
    if args.use_distillation:
        _exp_segments.append("distil")
    exp_path = "-".join(_exp_segments)
    train_loader = cl_train_loader[task_id]
    utils.train_one_task(model, train_loader, task_id, optimizer, 
                     ewc, 
                     gr,
                     kl,
                     args.epochs,
                     ROOT / exp_path,
                    #  None,
                     device, wandb)
    # save model after each task
    model_path = ROOT / exp_path / f"model-task{task_id}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    if args.check_fisher:
        loaders_by_class = {0: train_loader}
        param_scores = compute_param_scores(
            model,
            loaders_by_class,
            device=device,
            target_class=0,
            max_samples=None,
        )  # shape (B, D) on device
        B = param_scores.shape[0]
        F = None
        if device.type == "cuda":
            try:
                F = empirical_fisher_dense(param_scores)  # (D, D)
                frobF = torch.linalg.norm(F).item()
                frobF_sq = frobF * frobF
            except Exception as _e:
                print(f"[Fisher] Dense Fisher failed ({_e}).")
            if F is not None:
                c_rank, mu_rank, diag_rank = compute_rank1_coeff_and_mean(
                    model, train_loader, device=device, max_samples=None
                )
                # Ensure vectors match param_scores device/dtype for safe matmul/dot
                mu_rank = mu_rank.to(param_scores.device, dtype=param_scores.dtype)
                c_eig, mu_eig = compute_top_eigenpair_two_pass(
                    model,
                    train_loader,
                    device=device,
                    max_samples=None,
                    power_iters=args.power_iters,
                )
                mu_eig = mu_eig.to(param_scores.device, dtype=param_scores.dtype)

                ## Find the Norm of F - mu^T @ mu (use floats to avoid Tensor truthiness issues)
                mu_rank_norm = float(torch.dot(mu_rank, mu_rank))  # ||mu_rank||^2
                c_rank_f = float(c_rank)
                Su_rank = float(torch.sum((param_scores @ mu_rank).pow(2)) / float(B))
                err_rank_abs = math.sqrt(
                    max(
                        frobF_sq + (c_rank_f * c_rank_f) * (mu_rank_norm * mu_rank_norm) - 2.0 * c_rank_f * Su_rank,
                        0.0,
                    )
                )

                mu_eig_norm = float(torch.dot(mu_eig, mu_eig))  # ||mu_eig||^2
                c_eig_f = float(c_eig)
                Su_eig = float(torch.sum((param_scores @ mu_eig).pow(2)) / float(B))
                err_eig_abs = math.sqrt(
                    max(
                        frobF_sq + (c_eig_f * c_eig_f) * (mu_eig_norm * mu_eig_norm) - 2.0 * c_eig_f * Su_eig,
                        0.0,
                    )
                )

                ## Find Norm of F - diag
                diag_sq_sum = float(torch.dot(diag_rank, diag_rank))
                err_diag_abs = math.sqrt(max(frobF_sq - diag_sq_sum, 0.0))

                if args.use_wandb:
                    # Compute relative errors once for reuse
                    err_rank_rel = err_rank_abs / (frobF + 1e-12)
                    err_eig_rel = err_eig_abs / (frobF + 1e-12)
                    err_diag_rel = err_diag_abs / (frobF + 1e-12)

                    # Log scalars
                    wandb.log({
                        f"fisher/{task_id}/frobF": frobF,
                        f"fisher/{task_id}/err_rank_rel": err_rank_rel,
                        f"fisher/{task_id}/err_eig_rel": err_eig_rel,
                        f"fisher/{task_id}/err_diag_rel": err_diag_rel,
                    })

                    # Also log a bar chart and histogram for the three errors (one per task)
                    try:
                        # Bar chart via W&B plot API
                        table = wandb.Table(columns=["metric", "value"],
                                            data=[["rank", float(err_rank_rel)],
                                                  ["eig", float(err_eig_rel)],
                                                  ["diag", float(err_diag_rel)]])
                        bar = wandb.plot.bar(table, "metric", "value",
                                             title=f"Fisher relative errors (task {task_id})")
                        wandb.log({f"fisher/{task_id}/rel_errors_bar": bar})

                        # Histogram of the three values
                        wandb.log({
                            f"fisher/{task_id}/rel_errors_hist": wandb.Histogram([
                                float(err_rank_rel), float(err_eig_rel), float(err_diag_rel)
                            ])
                        })
                    except Exception as _plot_e:
                        # Fallback: ignore plotting failures but keep scalars
                        print(f"[W&B] Plot logging failed for task {task_id}: {_plot_e}")
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
        if args.ewc_fisher_type == "top_eig":
            if "c_eig" in locals() and "mu_eig" in locals():
                c, mu = c_eig, mu_eig
            else:
                c, mu = compute_top_eigenpair_two_pass(model, train_loader, device=device, max_samples=None, power_iters=args.power_iters)
            diag = None
        else:
            if "c_rank" in locals() and "mu_rank" in locals() and "diag_rank" in locals():
                c, mu, diag = c_rank, mu_rank, diag_rank
            else:
                c, mu, diag = compute_rank1_coeff_and_mean(
                    model, train_loader, device=device, max_samples=None
                )
        if ewc is None:
            # create a new EWC object
            fisher_type = args.ewc_fisher_type
            # save the fisher information too
            torch.save((c, mu, diag), ROOT / exp_path / f"fisher-task{task_id}.pt")
            frozen_model = utils.freeze_model(model)
            ewc = EWC(frozen_model, fisher_type, c=c, mu=mu, diag=diag)
        else:
            # add a new task to the existing EWC object
            torch.save((c, mu, diag), ROOT / exp_path / f"fisher-task{task_id}.pt")

            frozen_model = utils.freeze_model(model)
            ewc.add_task(frozen_model, c=c, mu=mu, diag=diag)

    if args.use_generative_replay:
        if gr is None:
            frozen_model = utils.freeze_model(model)
            gr = GenerativeReplay(frozen_model, old_classes=list(range((task_id + 1) * args.group_size)),
                                 alpha=args.gr_alpha, 
                                 batch_size=args.batch_size, 
                                 pool_size_per_class=args.gr_pool_size_per_class,
                                 num_inference_steps=args.gr_num_inference_steps,
                                 eta=args.gr_eta,
                                 seed=args.seed,
                                 device=device)
        else:
            frozen_model = utils.freeze_model(model)
            gr.update_teacher(
                frozen_model,
                old_classes=list(range((task_id + 1) * args.group_size))
            )