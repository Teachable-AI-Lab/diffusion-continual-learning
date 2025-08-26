"""
This contains the core experiment logic from the notebook.
"""

import random
from pathlib import Path
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

from src.utils.data_loading import get_cl_dataset
from src.utils.training import train_one_task
from src.utils.evaluation import FIDEvaluator
from src.utils.ddim import create_models_with_optimizers
from src.utils.parameter_scoring import compute_param_scores, compute_rank1_coeff_and_mean
from src.utils.ewc import EWC
from src.utils.fisher_analysis import empirical_fisher_dense, optimal_rank1_coeff
from src.utils.fisher_utils import compare_fisher_errors_streaming, plot_fisher_errors, analyze_fisher_approximations


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



def create_models(device, model_size="normal", learning_rate=2e-4):
    """Create MNIST and CIFAR models using the utility function."""
    return create_models_with_optimizers(device, model_size=model_size, learning_rate=learning_rate)


def train_task_0(model, train_loader, optimizer, n_epochs, ROOT, device, model_name):
    """Train the first task and save the model."""
    train_one_task(model, train_loader, 0, optimizer, None, n_epochs, ROOT, device)
    
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
    
    train_one_task(model, loaders[1], 1, optimizer, ewc, n_epochs, ROOT, device)
    
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
    fid_eval = FIDEvaluator(device=device)
    fid_score = fid_eval.fid_loader_vs_model(test_loader, model)
    print(f"FID score: {fid_score:.3f}")
    return fid_score


def sample_and_visualize(model, device, n_samples=16):
    """Generate and visualize samples from the model."""
    # Random labels between 0 and 1
    labels = torch.randint(0, 2, (n_samples,), device=device)
    samples = model.sample(n_samples, labels=labels, num_inference_steps=50, device=device, seed=123)
    samples = (samples + 1) / 2  # to [0, 1]
    grid_img = make_grid(samples, nrow=4)
    
    plt.figure(figsize=(6,6))
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title("Model Samples")
    plt.show()
    
    return samples


def load_datasets(batch_size=128):
    """Load all required datasets for the experiment."""
    cl_cifar_train_loaders, cl_cifar_test_loaders, cifar_train_loader, cifar_test_loader = get_cl_dataset(
        'cifar10', batch_size=batch_size, normalize=True, greyscale=False
    )
    cl_mnist_train_loaders, cl_mnist_test_loaders, mnist_train_loader, mnist_test_loader = get_cl_dataset(
        'mnist', batch_size=batch_size, normalize=True, greyscale=True
    )
    
    return {
        'cl_mnist_train': cl_mnist_train_loaders,
        'cl_mnist_test': cl_mnist_test_loaders,
        'cl_cifar_train': cl_cifar_train_loaders,
        'cl_cifar_test': cl_cifar_test_loaders,
        'mnist_train': mnist_train_loader,
        'mnist_test': mnist_test_loader,
        'cifar_train': cifar_train_loader,
        'cifar_test': cifar_test_loader
    }


def full_fisher_analysis(model, loaders, device, ROOT):
    """Perform full Fisher matrix analysis (only for small models)."""
    print("Performing full Fisher matrix analysis...")
    
    # Compute parameter scores
    param_scores = compute_param_scores(
        model, 0, loaders, device=device, target_class=0, max_samples=None
    )
    
    # Compute full Fisher matrix
    Fisher = empirical_fisher_dense(param_scores).to('cpu')
    param_scores = param_scores.to('cpu')
    
    # Compute optimal rank-1 coefficient
    c_analysis, mu_analysis = optimal_rank1_coeff(param_scores, eps=1e-12, use_float64=False)
    
    # Diagonal approximation
    F_diag = torch.diag(torch.diag(Fisher))
    err_diag = torch.linalg.norm(Fisher - F_diag)
    
    # Rank-1 approximation with optimal coefficient
    F_r1_score = mu_analysis.unsqueeze(1) @ mu_analysis.unsqueeze(0) * c_analysis
    err_r1_score = torch.linalg.norm(Fisher - F_r1_score)
    
    print(f"‖F-F_diag‖_F = {err_diag:.10f},  ‖F-F_r1_score‖_F = {err_r1_score:.10f}")
    
    return {
        'Fisher': Fisher,
        'param_scores': param_scores,
        'diagonal_error': err_diag.item(),
        'rank1_optimal_error': err_r1_score.item(),
        'c_analysis': c_analysis,
        'mu_analysis': mu_analysis
    }

