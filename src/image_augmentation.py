import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import get_data_path
from torchvision import datasets

def show_images(images, titles, rows=2, cols=5):
    """Display multiple images in a grid"""
    plt.figure(figsize=(15, 6))
    for idx, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def get_augmented_images(image):
    """Apply different augmentations to an image"""
    # Define various augmentations
    augmentations = {
        'Original': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'Rotation': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(30),
        ]),
        'Horizontal Flip': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=1.0),
        ]),
        'Affine': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]),
        'Brightness': transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.5),
        ]),
        'Gaussian Noise': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'Gaussian Blur': transforms.Compose([
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=3),
        ]),
        'Random Erasing': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.1)),
        ]),
        'Sharpness': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
        ]),
        'Perspective': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        ]),
    }
    
    # Apply each augmentation
    augmented_images = []
    titles = []
    
    for name, transform in augmentations.items():
        img_tensor = transform(image)
        
        # Add Gaussian noise manually (as it's not a built-in transform)
        if name == 'Gaussian Noise':
            noise = torch.randn_like(img_tensor) * 0.1
            img_tensor = torch.clamp(img_tensor + noise, 0, 1)
        
        augmented_images.append(img_tensor)
        titles.append(name)
    
    return augmented_images, titles

def main():
    # Get data directory path
    data_path = get_data_path()
    # Load MNIST dataset
    dataset = datasets.MNIST(data_path, 
                             train=True, 
                             download=True)
    
    # Get a random image
    idx = np.random.randint(len(dataset))
    image, label = dataset[idx]
    
    print(f"Selected digit: {label}")
    
    # Get augmented versions
    augmented_images, titles = get_augmented_images(image)
    
    # Display all versions
    show_images(augmented_images, titles)

if __name__ == "__main__":
    main() 