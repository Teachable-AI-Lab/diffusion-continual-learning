"""
Data loading utilities for continual learning datasets.

Provides functions for loading and organizing datasets by class/task for continual learning scenarios.
"""

import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def get_cl_dataset(name='mnist', batch_size=64, normalize=True, greyscale=False):
    """
    Create continual learning datasets organized by groups/tasks.
    
    Args:
        name (str): Dataset name ('mnist', 'fmnist', 'cifar10')
        batch_size (int): Batch size for individual task loaders
        normalize (bool): Whether to normalize the data
        greyscale (bool): Convert CIFAR10 to greyscale (only applicable for CIFAR10)
        
    Returns:
        tuple: (train_loaders, test_loaders, full_train_loader, full_test_loader)
            - train_loaders: dict mapping group_id -> DataLoader for that group
            - test_loaders: dict mapping group_id -> DataLoader for that group  
            - full_train_loader: DataLoader for entire training set
            - full_test_loader: DataLoader for entire test set
    """
    # Define transforms based on dataset
    if name.lower() == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        if normalize:
            transform = transforms.Compose([
                transforms.Pad(2),  # Padding to make it 32x32
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = datasets.MNIST(root='archive/data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='archive/data', train=False, download=True, transform=transform)
        
    elif name.lower() == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor()])
        if normalize:
            transform = transforms.Compose([
                transforms.Pad(2),  # Padding to make it 32x32
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = datasets.FashionMNIST(root='archive/data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='archive/data', train=False, download=True, transform=transform)
        
    elif name.lower() == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        if normalize:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        if greyscale:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = datasets.CIFAR10(root='archive/data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='archive/data', train=False, download=True, transform=transform)
        
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # Organize data by groups (2 classes per group)
    group_size = 2
    n_classes = 10
    n_groups = n_classes // group_size  # == 5

    # Build indices for each group in training set
    train_indices_per_group = {g: [] for g in range(n_groups)}
    print("Building DataLoaders for each group in train dataset...")
    for idx, (_, label) in enumerate(tqdm(train_dataset)):
        g = label // group_size
        train_indices_per_group[g].append(idx)

    # Create DataLoaders for each training group
    train_loaders = {}
    for g, indices in sorted(train_indices_per_group.items()):
        subset = Subset(train_dataset, indices)
        train_loaders[g] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    # Build indices for each group in test set  
    test_indices_per_group = {g: [] for g in range(n_groups)}
    print("Building DataLoaders for each group in test dataset...")
    for idx, (_, label) in enumerate(tqdm(test_dataset)):
        g = label // group_size
        test_indices_per_group[g].append(idx)

    # Create DataLoaders for each test group
    test_loaders = {}
    for g, indices in sorted(test_indices_per_group.items()):
        subset = Subset(test_dataset, indices)
        test_loaders[g] = DataLoader(
            subset,
            batch_size=512,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    # Full dataset loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loaders, test_loaders, train_loader, test_loader
