"""
Main experiment runner that handles training, continual learning, Fisher analysis, and evaluation.
"""

import argparse
import torch
from torch import optim
from pathlib import Path

from src.experiment_runner import (
    setup_experiment, load_datasets, create_models,
    train_task_0, compute_fisher_information,
    train_continual_learning_tasks, analyze_fisher_errors_across_timesteps,
    evaluate_fid, sample_and_visualize, full_fisher_analysis
)


def run_full_experiment(dataset='mnist', model_size='normal', n_epochs=200, batch_size=128):
    """
    Run the complete experiment pipeline.
    
    Args:
        dataset: 'mnist' or 'cifar' - which dataset to use
        model_size: 'small', 'normal', or 'large' - model size
        n_epochs: number of epochs for training
        batch_size: batch size for data loaders
    """
    print(f"Starting experiment with dataset={dataset}, model_size={model_size}")
    
    # Setup experiment
    device, ROOT = setup_experiment(seed=123)
    
    # Load datasets
    print("Loading datasets...")
    datasets = load_datasets(batch_size=batch_size)
    
    # Select appropriate dataset
    if dataset.lower() == 'mnist':
        cl_train_loaders = datasets['cl_mnist_train']
        cl_test_loaders = datasets['cl_mnist_test']
        model_name = f"mnist_model_{model_size}"
        in_channel = 1
        num_classes = 4
    elif dataset.lower() == 'cifar':
        cl_train_loaders = datasets['cl_cifar_train']
        cl_test_loaders = datasets['cl_cifar_test']
        model_name = f"cifar_model_{model_size}"
        in_channel = 3
        num_classes = 2
    else:
        raise ValueError("Dataset must be 'mnist' or 'cifar'")
    
    # Create models
    print(f"Creating {model_size} models...")
    if dataset.lower() == 'mnist':
        mnist_model, _, mnist_opt, _ = create_models(device, model_size=model_size)
        model = mnist_model
        optimizer = mnist_opt
    else:
        _, cifar_model, _, cifar_opt = create_models(device, model_size=model_size)
        model = cifar_model
        optimizer = cifar_opt
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Train Task 0
    print("Training Task 0...")
    train_task_0(model, cl_train_loaders[0], optimizer, n_epochs, ROOT, device, model_name)
    
    # Load the trained model
    model.load_state_dict(torch.load(ROOT / f"{model_name}.pth", map_location=device))
    
    # Compute Fisher information
    print("Computing Fisher information...")
    c, mu, diag, param_scores = compute_fisher_information(model, cl_train_loaders, device)
    
    # Full Fisher analysis (only for small models)
    if model_size == 'small':
        print("Performing full Fisher matrix analysis (small model detected)...")
        fisher_results = full_fisher_analysis(model, cl_train_loaders, device, ROOT)
    else:
        print("Skipping full Fisher matrix analysis (model too large)")
        fisher_results = None
    
    # Analyze Fisher errors across timesteps
    print("Analyzing Fisher approximation errors across timesteps...")
    error_analysis = analyze_fisher_errors_across_timesteps(model, cl_train_loaders, device, ROOT)
    
    # Train continual learning tasks with different EWC variants
    print("Training continual learning tasks...")
    base_state = torch.load(ROOT / f"{model_name}.pth", map_location=device)
    
    ewc_variants = ['diag', 'rank1_opt', 'rank1']
    for variant in ewc_variants:
        print(f"Training with EWC variant: {variant}")
        model.load_state_dict(base_state)
        new_optimizer = optim.Adam(model.parameters(), lr=2e-4)
        train_continual_learning_tasks(
            model, cl_train_loaders, new_optimizer, variant, 
            c, mu, diag, n_epochs, ROOT, device, variant
        )
    
    # Evaluate models
    print("Evaluating models...")
    results = {}
    
    # Evaluate base model (Task 0 only)
    model.load_state_dict(base_state)
    fid_base = evaluate_fid(model, cl_test_loaders[0], device)
    results['base_fid'] = fid_base
    
    # Evaluate continual learning variants
    for variant in ewc_variants:
        try:
            model_path = ROOT / f"{model_name}-task1-{variant}.pth"
            model.load_state_dict(torch.load(model_path, map_location=device))
            fid_score = evaluate_fid(model, cl_test_loaders[0], device)
            results[f'{variant}_fid'] = fid_score
        except FileNotFoundError:
            print(f"Model file not found for variant {variant}")
            results[f'{variant}_fid'] = None
    
    # Generate and visualize samples from best model
    print("Generating samples...")
    best_variant = min(ewc_variants, key=lambda v: results.get(f'{v}_fid', float('inf')))
    model.load_state_dict(torch.load(ROOT / f"{model_name}-task1-{best_variant}.pth", map_location=device))
    samples = sample_and_visualize(model, device, n_samples=16)
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Dataset: {dataset.upper()}")
    print(f"Model size: {model_size}")
    print(f"Optimal coefficient c*: {c.item():.6f}")
    print(f"Base model FID: {results['base_fid']:.3f}")
    for variant in ewc_variants:
        if results[f'{variant}_fid'] is not None:
            print(f"{variant.upper()} EWC FID: {results[f'{variant}_fid']:.3f}")
    print(f"Best EWC variant: {best_variant.upper()}")
    
    if fisher_results:
        print(f"Full Fisher diagonal error: {fisher_results['diagonal_error']:.10f}")
        print(f"Full Fisher rank-1 optimal error: {fisher_results['rank1_optimal_error']:.10f}")
    
    print("Fisher error analysis saved to:", ROOT / "ewc_fisher_errors.png")
    print("="*50)
    
    return {
        'results': results,
        'error_analysis': error_analysis,
        'fisher_results': fisher_results,
        'c_optimal': c.item(),
        'model_path': ROOT
    }


def main():
    """Command-line interface for running experiments."""
    parser = argparse.ArgumentParser(description='Run diffusion continual learning experiment')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar'], default='mnist',
                      help='Dataset to use (mnist or cifar)')
    parser.add_argument('--model_size', type=str, choices=['small', 'normal', 'large'], default='normal',
                      help='Model size (small, normal, large)')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size')
    
    args = parser.parse_args()
    
    return run_full_experiment(
        dataset=args.dataset,
        model_size=args.model_size,
        n_epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    results = main()
    print(f"Experiment completed. Results saved to: {results['model_path']}")
