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

    # block_out_channels=(16,),
    # down_block_types=("DownBlock2D",),
    # up_block_types=("UpBlock2D",),
    # norm_num_groups=8,
    # layers_per_block=1,
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
    exp_path = f"{args.dataset}-{args.ewc_fisher_type}-{"gr" if args.use_generative_replay else ""}-{"distil" if args.use_distillation else ""}"
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
