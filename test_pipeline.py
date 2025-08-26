"""
Quick test script to validate the experiment pipeline without heavy training.
"""

import torch
import sys
from pathlib import Path

def test_imports():
    """Test if all imports work correctly."""
    print("Testing imports...")
    try:
        from src.experiment_runner import (
            setup_experiment, load_datasets, create_models,
            train_task_0, compute_fisher_information,
            train_continual_learning_tasks, analyze_fisher_errors_across_timesteps,
            evaluate_fid, sample_and_visualize, full_fisher_analysis
        )
        from src.run_experiment import run_full_experiment
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_setup_and_data_loading():
    """Test experiment setup and data loading."""
    print("\nTesting setup and data loading...")
    try:
        from src.experiment_runner import setup_experiment, load_datasets
        
        # Setup
        device, ROOT = setup_experiment(seed=123)
        print(f"✓ Setup successful, device: {device}")
        
        # Load datasets (small batch for testing)
        datasets = load_datasets(batch_size=8)
        print("✓ Datasets loaded successfully")
        print(f"  - MNIST train loaders: {len(datasets['cl_mnist_train'])}")
        print(f"  - CIFAR train loaders: {len(datasets['cl_cifar_train'])}")
        
        return True, device, ROOT, datasets
    except Exception as e:
        print(f"✗ Setup/data loading error: {e}")
        return False, None, None, None


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    try:
        from src.experiment_runner import create_models
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test different model sizes
        for size in ['small', 'normal']:  # Skip 'large' for quick testing
            print(f"  Testing {size} models...")
            mnist_model, cifar_model, mnist_opt, cifar_opt = create_models(device, model_size=size)
            
            mnist_params = sum(p.numel() for p in mnist_model.parameters() if p.requires_grad)
            cifar_params = sum(p.numel() for p in cifar_model.parameters() if p.requires_grad)
            
            print(f"    {size} MNIST model: {mnist_params:,} parameters")
            print(f"    {size} CIFAR model: {cifar_params:,} parameters")
        
        print("✓ Model creation successful")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False


def test_parameter_scoring():
    """Test parameter scoring with minimal data."""
    print("\nTesting parameter scoring...")
    try:
        from src.experiment_runner import setup_experiment, load_datasets, create_models
        from src.utils.parameter_scoring import compute_param_scores
        
        device, ROOT = setup_experiment(seed=123)
        datasets = load_datasets(batch_size=4)  # Very small batch
        mnist_model, _, _, _ = create_models(device, model_size='small')
        
        # Test with very few samples
        param_scores = compute_param_scores(
            mnist_model, 
            t_level=0, 
            loaders_by_class=datasets['cl_mnist_train'], 
            device=device, 
            target_class=0, 
            max_samples=2  # Only 2 samples for quick test
        )
        
        print(f"✓ Parameter scoring successful, shape: {param_scores.shape}")
        return True, param_scores
    except Exception as e:
        print(f"✗ Parameter scoring error: {e}")
        return False, None


def test_fisher_information():
    """Test Fisher information computation."""
    print("\nTesting Fisher information computation...")
    try:
        from src.experiment_runner import setup_experiment, load_datasets, create_models
        from src.utils.parameter_scoring import compute_rank1_coeff_and_mean
        from src.utils.fisher_analysis import optimal_rank1_coeff
        
        device, ROOT = setup_experiment(seed=123)
        datasets = load_datasets(batch_size=4)
        mnist_model, _, _, _ = create_models(device, model_size='small')
        
        # Test streaming computation
        c, mu, diag = compute_rank1_coeff_and_mean(
            mnist_model, 0, datasets['cl_mnist_train'], device=device, 
            target_class=0, max_samples=2
        )
        
        print(f"✓ Fisher information computed:")
        print(f"  - Optimal coefficient c*: {c.item():.6f}")
        print(f"  - Mu shape: {mu.shape}")
        print(f"  - Diagonal shape: {diag.shape}")
        
        return True, c, mu, diag
    except Exception as e:
        print(f"✗ Fisher information error: {e}")
        return False, None, None, None


def test_ewc_setup():
    """Test EWC setup."""
    print("\nTesting EWC setup...")
    try:
        from src.experiment_runner import setup_experiment, load_datasets, create_models
        from src.utils.parameter_scoring import compute_rank1_coeff_and_mean
        from src.utils.ewc import EWC
        
        device, ROOT = setup_experiment(seed=123)
        datasets = load_datasets(batch_size=4)
        mnist_model, _, _, _ = create_models(device, model_size='small')
        
        # Get Fisher info
        c, mu, diag = compute_rank1_coeff_and_mean(
            mnist_model, 0, datasets['cl_mnist_train'], device=device, 
            target_class=0, max_samples=2
        )
        
        # Test different EWC variants
        ewc_diag = EWC(mnist_model, "diag", diag=diag)
        ewc_rank1_opt = EWC(mnist_model, "rank1_opt", mu=mu, c=c)
        ewc_rank1 = EWC(mnist_model, "rank1", mu=mu, c=c)
        
        print("✓ EWC setup successful:")
        print(f"  - Diagonal EWC loss: {ewc_diag.loss().item():.6f}")
        print(f"  - Rank-1 optimal EWC loss: {ewc_rank1_opt.loss().item():.6f}")
        print(f"  - Rank-1 EWC loss: {ewc_rank1.loss().item():.6f}")
        
        return True
    except Exception as e:
        print(f"✗ EWC setup error: {e}")
        return False


def test_fid_evaluation():
    """Test FID evaluation setup."""
    print("\nTesting FID evaluation...")
    try:
        from src.experiment_runner import setup_experiment, load_datasets, create_models
        from src.utils.evaluation import FIDEvaluator
        
        device, ROOT = setup_experiment(seed=123)
        datasets = load_datasets(batch_size=4)
        mnist_model, _, _, _ = create_models(device, model_size='small')
        
        # Just test if FID evaluator can be created and initialized
        fid_eval = FIDEvaluator(device=device)
        print("✓ FID evaluator created successfully")
        
        # Test sampling (without FID computation which is expensive)
        n_samples = 4
        labels = torch.randint(0, 2, (n_samples,), device=device)
        samples = mnist_model.sample(n_samples, labels=labels, num_inference_steps=10, device=device, seed=123)
        print(f"✓ Model sampling successful, shape: {samples.shape}")
        
        return True
    except Exception as e:
        print(f"✗ FID evaluation error: {e}")
        return False


def run_all_tests():
    """Run all tests in sequence."""
    print("="*50)
    print("RUNNING PIPELINE TESTS")
    print("="*50)
    
    # Test 1: Imports
    if not test_imports():
        return False
    
    # Test 2: Setup and data loading
    success, device, ROOT, datasets = test_setup_and_data_loading()
    if not success:
        return False
    
    # Test 3: Model creation
    if not test_model_creation():
        return False
    
    # Test 4: Parameter scoring
    success, param_scores = test_parameter_scoring()
    if not success:
        return False
    
    # Test 5: Fisher information
    success, c, mu, diag = test_fisher_information()
    if not success:
        return False
    
    # Test 6: EWC setup
    if not test_ewc_setup():
        return False
    
    # Test 7: FID evaluation
    if not test_fid_evaluation():
        return False
    
    print("\n" + "="*50)
    print("✓ ALL TESTS PASSED!")
    print("The pipeline is ready to run full experiments.")
    print("="*50)
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\nTo run a full experiment, use:")
        print("python src/run_experiment.py --dataset mnist --model_size small --epochs 5")
    else:
        print("\nSome tests failed. Please check the errors above.")
        sys.exit(1)
