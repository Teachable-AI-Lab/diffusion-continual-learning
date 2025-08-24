"""
Main experimental script for running the diffusion continual learning experiments.
This contains the core experiment logic from the notebook.
"""

import os
import math
import time
import random
from pathlib import Path
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from src.ddim import build_conditional_ddim
import src.utils as utils
from src.parameter_scoring import compute_param_scores, compute_rank1_coeff_and_mean
from src.ewc import EWC
from src.fisher_analysis import empirical_fisher_dense, optimal_rank1_coeff
from src.experimental_utils import compare_fisher_errors_streaming, plot_fisher_errors, analyze_fisher_approximations


def setup_experiment(seed=123, device=None):
    """Setup experiment with reproducible seed and device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Device:", device)
    
    # Reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    # Create output directory
    ROOT = Path("./ddim_unit_tests")
    ROOT.mkdir(exist_ok=True, parents=True)
    
    return device, ROOT


def load_datasets(batch_size=128):
    """Load continual learning datasets."""
    cl_cifar_train_loaders, cl_cifar_test_loaders, cifar_train_loader, cifar_test_loader = utils.get_cl_dataset(
        'cifar10', batch_size=batch_size, normalize=True, greyscale=False
    )
    cl_mnist_train_loaders, cl_mnist_test_loaders, mnist_train_loader, mnist_test_loader = utils.get_cl_dataset(
        'mnist', batch_size=batch_size, normalize=True, greyscale=True
    )
    
    return {
        'cl_cifar_train': cl_cifar_train_loaders,
        'cl_cifar_test': cl_cifar_test_loaders,
        'cifar_train': cifar_train_loader,
        'cifar_test': cifar_test_loader,
        'cl_mnist_train': cl_mnist_train_loaders,
        'cl_mnist_test': cl_mnist_test_loaders,
        'mnist_train': mnist_train_loader,
        'mnist_test': mnist_test_loader
    }


def create_models(device):
    """Create MNIST and CIFAR models."""
    mnist_model = build_conditional_ddim(
        in_channel=1,
        image_size=32,
        num_class_labels=4,
    ).to(device)

    print("MNIST model parameters:", sum(p.numel() for p in mnist_model.parameters() if p.requires_grad))

    cifar_model = build_conditional_ddim(
        in_channel=3,
        image_size=32,
        num_class_labels=2,
    ).to(device)

    print("CIFAR model parameters:", sum(p.numel() for p in cifar_model.parameters() if p.requires_grad))

    mnist_opt = optim.Adam(mnist_model.parameters(), lr=2e-4)
    cifar_opt = optim.Adam(cifar_model.parameters(), lr=2e-4)
    
    return mnist_model, cifar_model, mnist_opt, cifar_opt


def train_task_0(model, train_loader, optimizer, n_epochs, ROOT, device, model_name):
    """Train the first task and save the model."""
    utils.train_one_task(model, train_loader, 0, optimizer, None, n_epochs, ROOT, device)
    
    # Save the model
    model_path = ROOT / f"{model_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def compute_fisher_information(model, loaders, device):
    """Compute Fisher information using both methods."""
    print("Computing Fisher information...")
    
    # Method 1: Streaming computation
    c, mu, diag = compute_rank1_coeff_and_mean(
        model, 0, loaders, device=device, target_class=0, max_samples=None
    )
    
    # Method 2: Batch computation for comparison
    param_scores = compute_param_scores(
        model, 0, loaders, device=device, target_class=0, max_samples=None
    )
    c_batch, mu_batch = optimal_rank1_coeff(param_scores, eps=1e-12, use_float64=False)
    
    print(f"Streaming optimal coefficient c*: {c.item()}")
    print(f"Batch optimal coefficient c*: {c_batch.item()}")
    
    return c, mu, diag, param_scores


def train_continual_learning_tasks(model, loaders, optimizer, ewc_type, c, mu, diag, n_epochs, ROOT, device, model_suffix):
    """Train continual learning with different EWC variants."""
    if ewc_type == "diag":
        ewc = EWC(model, "diag", diag=diag)
    elif ewc_type == "rank1_opt":
        ewc = EWC(model, "rank1_opt", mu=mu, c=c)
    elif ewc_type == "rank1":
        ewc = EWC(model, "rank1", mu=mu, c=c)
    else:
        ewc = None
    
    utils.train_one_task(model, loaders[1], 1, optimizer, ewc, n_epochs, ROOT, device)
    
    # Save the model
    model_path = ROOT / f"mnist_model_large-task1-{model_suffix}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def analyze_fisher_errors_across_timesteps(model, loaders, device, ROOT):
    """Analyze Fisher matrix approximation errors across different timestep levels."""
    t_levels = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    diag_errors = []
    rank1_errors = []
    rank1_optimal_errors = []
    
    for t_level in t_levels:
        print(f"Computing param scores for t_level={t_level}")
        param_scores = compute_param_scores(
            model, t_level, loaders, device=device, target_class=0, max_samples=None
        )
        
        analysis_results = analyze_fisher_approximations(param_scores)
        
        diag_errors.append(analysis_results['diagonal_error'])
        rank1_errors.append(analysis_results['rank1_error'])
        rank1_optimal_errors.append(analysis_results['rank1_optimal_error'])

    # Plot and save results
    plot_fisher_errors(t_levels, diag_errors, rank1_errors, rank1_optimal_errors, 
                      save_path=ROOT / "ewc_fisher_errors.png")
    
    return {
        't_levels': t_levels,
        'diag_errors': diag_errors,
        'rank1_errors': rank1_errors,
        'rank1_optimal_errors': rank1_optimal_errors
    }


def evaluate_fid(model, test_loader, device):
    """Evaluate FID score."""
    fid_eval = utils.FIDEvaluator(device=device)
    fid_score = fid_eval.fid_loader_vs_model(test_loader, model)
    print(f"FID score: {fid_score:.3f}")
    return fid_score


def sample_and_visualize(model, device, n_samples=16):
    """Generate and visualize samples from the model."""
    # Random labels between 0 and 1
    labels = torch.randint(0, 2, (n_samples,), device=device)
    samples = model.sample(n_samples, labels=labels, num_inference_steps=50, device=device, seed=123)
    samples = (samples + 1) / 2  # to [0, 1]
    grid_img = utils.make_grid(samples, nrow=4)
    
    plt.figure(figsize=(6,6))
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title("Model Samples")
    plt.show()
    
    return samples


def run_full_experiment():
    """Run the complete experimental pipeline."""
    print("Starting full experiment...")
    
    # Setup
    device, ROOT = setup_experiment()
    datasets = load_datasets()
    mnist_model, cifar_model, mnist_opt, cifar_opt = create_models(device)
    
    # Train Task 0
    print("Training Task 0...")
    train_task_0(mnist_model, datasets['cl_mnist_train'][0], mnist_opt, 200, ROOT, device, "mnist_model_large")
    
    # Compute Fisher information
    c, mu, diag, param_scores = compute_fisher_information(mnist_model, datasets['cl_mnist_train'], device)
    
    # Analyze Fisher approximation errors
    print("Analyzing Fisher approximation errors...")
    error_analysis = analyze_fisher_errors_across_timesteps(mnist_model, datasets['cl_mnist_train'], device, ROOT)
    
    # Train continual learning tasks with different EWC variants
    print("Training Task 1 with different EWC variants...")
    
    # Load base model for each variant
    base_state = torch.load(ROOT / "mnist_model_large.pth", map_location=device)
    
    # Diagonal EWC
    mnist_model.load_state_dict(base_state)
    mnist_opt = optim.Adam(mnist_model.parameters(), lr=2e-4)
    train_continual_learning_tasks(mnist_model, datasets['cl_mnist_train'], mnist_opt, "diag", 
                                 c, mu, diag, 200, ROOT, device, "diag")
    
    # Rank-1 optimal EWC
    mnist_model.load_state_dict(base_state)
    mnist_opt = optim.Adam(mnist_model.parameters(), lr=2e-4)
    train_continual_learning_tasks(mnist_model, datasets['cl_mnist_train'], mnist_opt, "rank1_opt", 
                                 c, mu, diag, 200, ROOT, device, "rank1_opt")
    
    # Rank-1 EWC
    mnist_model.load_state_dict(base_state)
    mnist_opt = optim.Adam(mnist_model.parameters(), lr=2e-4)
    train_continual_learning_tasks(mnist_model, datasets['cl_mnist_train'], mnist_opt, "rank1", 
                                 c, mu, diag, 200, ROOT, device, "rank1")
    
    # Evaluate all variants
    print("Evaluating all model variants...")
    variants = ["rank1_opt", "rank1", "diag"]
    results = {}
    
    for variant in variants:
        model_path = ROOT / f"mnist_model_large-task1-{variant}.pth"
        if model_path.exists():
            mnist_model.load_state_dict(torch.load(model_path, map_location=device))
            fid_score = evaluate_fid(mnist_model, datasets['cl_mnist_test'][0], device)
            results[variant] = {'fid': fid_score}
            
            # Generate samples
            print(f"Generating samples for {variant}...")
            samples = sample_and_visualize(mnist_model, device)
    
    print("Experiment completed!")
    return results, error_analysis


if __name__ == "__main__":
    results, error_analysis = run_full_experiment()
