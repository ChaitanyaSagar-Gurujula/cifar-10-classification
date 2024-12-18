import torch
import torch.nn.functional as F
import torch.optim as optim
from src.model import get_model
from src.dataset import get_cifar_loaders
from src.utils import count_parameters
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # This can slow down training
    torch.use_deterministic_algorithms(True, warn_only=True) # Enforce deterministic algorithms

def calculate_accuracy(model, data_loader, device, desc="Accuracy"):
    """
    Calculate accuracy and loss for a given model and data loader.
    
    Args:
        model: The neural network model
        data_loader: DataLoader containing the data
        device: Device to run the computation on
        desc: Description for the progress bar
    
    Returns:
        tuple: (accuracy, loss, correct_count, total_count)
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    
    pbar = tqdm(data_loader, desc=desc, 
                total=len(data_loader),
                bar_format='{l_bar}{bar:30}{r_bar}')
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            running_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            current_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += current_correct
            total += len(data)
            
            # Update progress bar
            current_acc = 100 * correct / total
            current_loss = running_loss / total
            pbar.set_description(
                f'{desc}: {current_acc:.2f}% | Loss: {current_loss:.4f}'
            )
    
    return current_acc, current_loss, correct, total

def train(model, train_loader, test_loader, optimizer, scheduler, device, num_epochs=5):
    """
    Train the model for multiple epochs
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of epochs to train for
    
    Returns:
        dict: Training history containing accuracies and losses
    """
    history = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'learning_rates': [],
        'best_model_info': {
            'epoch': 0,
            'train_acc': 0,
            'train_loss': 0,
            'test_acc': 0,
            'test_loss': 0
        },
        'total_params': count_parameters(model)
    }
    
    best_acc = 0.0
    total_steps = num_epochs * len(train_loader)
    current_step = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("=" * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        

        desc =f"Realtime Epoch Training"

        pbar = tqdm(train_loader, desc=desc, 
                total=len(train_loader),
                bar_format='{l_bar}{bar:30}{r_bar}')
        
        # Training loop
        for data, target in pbar:
            current_step += 1
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = F.nll_loss(output, target)
            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Calculate batch accuracy
            pred = output.argmax(dim=1, keepdim=True)
            batch_acc = 100 * pred.eq(target.view_as(pred)).sum().item() / len(data)
            
            pbar.set_description(
                f'{desc} Accuracy: {batch_acc:.2f}% | Loss: {loss.item():.4f}'
            )
        
        
        # Calculate final training accuracy for this epoch
        print("\nCalculating final training accuracy...")
        train_acc, train_loss, train_correct, train_total = calculate_accuracy(
            model, train_loader, device, desc=f"Epoch {epoch} Training"
        )
        
        # Calculate test accuracy
        print("\nCalculating test accuracy...")
        test_acc, test_loss, test_correct, test_total = calculate_accuracy(
            model, test_loader, device, desc=f"Epoch {epoch} Testing"
        )
        
        # Save metrics
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"Training - Accuracy: {train_acc:.2f}%, Loss: {train_loss:.4f}")
        print(f"Testing  - Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}")
        
        # Save best model and its details
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"New best accuracy! Saving model...")
            torch.save(model.state_dict(), 'best_model.pth')
            
            # Store best model information
            history['best_model_info'] = {
                'epoch': epoch,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'test_acc': test_acc,
                'test_loss': test_loss
            }
    
    # Print conclusion after training
    print_training_conclusion(history, model, device)
    return history

def print_training_conclusion(history, model, device):
    """Print final training conclusions and best model details"""
    print("\n" + "="*50)
    print("TRAINING COMPLETED - FINAL RESULTS")
    print("="*50)
    
    print(f"\nModel Architecture:")
    print(f"└── Total Parameters: {history['total_params']:,}")
    
    best_info = history['best_model_info']
    
    print(f"\nBest Model achieved at Epoch {best_info['epoch']}:")
    print(f"├── Training Accuracy: {best_info['train_acc']:.2f}%")
    print(f"├── Training Loss: {best_info['train_loss']:.4f}")
    print(f"├── Test Accuracy: {best_info['test_acc']:.2f}%")
    print(f"└── Test Loss: {best_info['test_loss']:.4f}")
    
    print("\nModel saved as: 'best_model.pth'")
    print("="*50)

def visualize_misclassified(model, test_loader, device, num_images=25):
    """
    Visualize misclassified images using the trained model
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run inference on
        num_images: Number of misclassified images to show
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            # Find misclassified indices
            incorrect_idx = pred.ne(target).nonzero(as_tuple=True)[0]
            
            for idx in incorrect_idx:
                if len(misclassified_images) >= num_images:
                    break
                    
                misclassified_images.append(data[idx].cpu())
                misclassified_labels.append(target[idx].cpu())
                predicted_labels.append(pred[idx].cpu())
                
            if len(misclassified_images) >= num_images:
                break
    
    # Plot the misclassified images
    plt.figure(figsize=(15, 10))
    for idx in range(len(misclassified_images)):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(misclassified_images[idx].squeeze(), cmap='gray')
        plt.title(f'True: {misclassified_labels[idx]}\nPred: {predicted_labels[idx]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified.png')
    plt.show()

def main(model_name, is_train_augmentation):
    # Set seeds first
    set_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Select model (you can change this to 'simple' or 'deep')
    model_class = get_model(model_name)
    model = model_class().to(device)
    print(f"Selected model: {model_name}")
    print(f"Total parameters: {count_parameters(model)}")
    print(f"Training Data Augmented: {is_train_augmentation}")

    # Get data loaders
    train_loader, test_loader = get_cifar_loaders(batch_size=128, is_train_augmentation=is_train_augmentation)
    
    # Optimizer and Scheduler setup for multiple epochs
    num_epochs = 15
    learning_rate = 0.01
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Scheduler for the entire training duration
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        div_factor=10,
        final_div_factor=1000, # final LR = initial_lr / final_div_factor
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Train the model
    history = train(model, train_loader, test_loader, optimizer, scheduler, device, num_epochs=num_epochs)
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    
    # Visualize misclassified images
    #visualize_misclassified(model, test_loader, device)
    
    # Plot training history
    #plot_training_history(history)
    
    # Plot detailed learning rate changes
    #plot_lr_changes(history, num_epochs)

def plot_training_history(history):
    """Plot training metrics including learning rate"""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rates'], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    
    # Add vertical lines for epoch boundaries
    steps_per_epoch = len(history['learning_rates']) // len(history['train_acc'])
    for epoch in range(1, len(history['train_acc'])):
        plt.axvline(x=epoch * steps_per_epoch, color='r', linestyle='--', alpha=0.3)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_lr_changes(history, num_epochs):
    """Create a detailed view of learning rate changes"""
    plt.figure(figsize=(12, 6))
    
    steps = len(history['learning_rates'])
    steps_per_epoch = steps // num_epochs
    
    plt.plot(history['learning_rates'], label='Learning Rate')
    
    # Add epoch boundaries
    for epoch in range(1, num_epochs):
        plt.axvline(x=epoch * steps_per_epoch, color='r', linestyle='--', alpha=0.3)
        plt.text(epoch * steps_per_epoch, max(history['learning_rates'])*1.1, 
                f'Epoch {epoch+1}', ha='center')
    
    # Add annotations for phases
    warmup_steps = int(steps * 0.3)
    plt.annotate('Warmup Phase', 
                xy=(warmup_steps//2, history['learning_rates'][warmup_steps//2]),
                xytext=(warmup_steps//2, max(history['learning_rates'])*1.2),
                ha='center',
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('Annealing Phase', 
                xy=(warmup_steps + (steps-warmup_steps)//2, history['learning_rates'][warmup_steps]),
                xytext=(warmup_steps + (steps-warmup_steps)//2, max(history['learning_rates'])*1.2),
                ha='center',
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.title('Learning Rate Changes During Training')
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='light',
                        choices=[ 'light'],
                        help='Model architecture to use')
    parser.add_argument('--is_train_aug', type=lambda x: x.lower() in ['true', '1'],
                        default=True,
                        help='Training data to be augmented or not (True/False)')
    args = parser.parse_args()

    main(model_name=args.model, is_train_augmentation=args.is_train_aug)