import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

def seed_worker(worker_id):
    """Function to set random seed for DataLoader workers"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_data_path():
    """Get the path to data directory"""
    # Get the project root directory (assuming we're in src/)
    root_dir = Path(__file__).parent.parent
    # Create data directory if it doesn't exist
    data_dir = root_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)

def compute_mean_std(dataset):
    """Compute the mean and standard deviation of a dataset"""
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += images.size(0)
    
    mean /= total_samples
    std /= total_samples
    
    return mean.tolist(), std.tolist()

def get_cifar_mean_std(data_path, cache_file='cifar10_mean_std.json'):
    """Get CIFAR-10 mean and std, using a cache file if available"""
    cache_path = os.path.join(data_path, cache_file)
    
    if os.path.exists(cache_path):
        # Load cached mean and std from file
        with open(cache_path, 'r') as f:
            stats = json.load(f)
            print(f"Loaded mean and std from cache: {stats}")
            return stats['mean'], stats['std']
    else:
        # Compute mean and std
        print("Cache file not found. Computing mean and std...")
        temp_transform = transforms.Compose([transforms.ToTensor()])
        temp_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=temp_transform)
        mean, std = compute_mean_std(temp_dataset)
        
        # Save to cache file
        with open(cache_path, 'w') as f:
            json.dump({'mean': mean, 'std': std}, f)
            print(f"Saved computed mean and std to cache: {cache_path}")
        
        return mean, std

def get_cifar_loaders(batch_size=128, is_train_augmentation=True):
    """Create CIFAR-10 train and test data loaders"""
    # Get data directory path
    data_path = get_data_path()
    
    # Retrieve mean and std (either from cache or by computing)
    mean, std = get_cifar_mean_std(data_path)
    
    if is_train_augmentation:
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                shear=(-5, 5),
                fill=0
            ),
            transforms.RandomPerspective(
                distortion_scale=0.2,
                p=0.1,
                fill=0
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # Training transforms without augmentation
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # Test transforms (no augmentation needed)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load datasets with respective transforms
    train_dataset = datasets.CIFAR10(
        data_path, train=True, download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        data_path, train=False,
        transform=test_transform
    )
    
    g = torch.Generator()
    g.manual_seed(42)  # Use the same seed as in set_seed()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=11,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    return train_loader, test_loader