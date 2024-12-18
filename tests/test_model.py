import pytest
import torch
from src.model import SuperLightMNIST
from src.dataset import get_mnist_loaders
from src.train import calculate_accuracy

def test_model_parameters():
    """Test if model has less than 20k parameters"""
    model = SuperLightMNIST()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"

def test_model_performance():
    """Test if saved model achieves required accuracy on both training and test sets"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperLightMNIST().to(device)
    
    # Load the best saved model
    try:
        model.load_state_dict(
            torch.load(
                'best_model.pth',
                map_location=device,
                weights_only=True  # Add this parameter
            )
        )
    except FileNotFoundError:
        pytest.skip("No saved model found. Run training first.")
    
    # Get both train and test loaders
    train_loader, test_loader = get_mnist_loaders(batch_size=128, is_train_augmentation=False)
    
    # Evaluate on training set
    train_acc, train_loss, train_correct, train_total = calculate_accuracy(
        model, train_loader, device, desc="Validating Training Accuracy"
    )
    
    # Evaluate on test set
    test_acc, test_loss, test_correct, test_total = calculate_accuracy(
        model, test_loader, device, desc="Validating Test Accuracy"
    )
    
    # Print detailed results
    print(f"\nModel Performance:")
    print(f"Training - Accuracy: {train_acc:.2f}%, Loss: {train_loss:.4f}")
    print(f"Testing  - Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}")
    
    # Assert both accuracies meet requirements
    assert train_acc >= 99.4, f"Training accuracy {train_acc:.2f}% is less than 99.4%"
    assert test_acc >= 99.4, f"Test accuracy {test_acc:.2f}% is less than 99.4%" 

def test_batch_normalization():
    """Test if BatchNorm layers are working correctly"""
    model = SuperLightMNIST()
    model.train()  # Set to train mode to update running stats
    
    # Create dummy input and do multiple forward passes
    x = torch.randn(4, 1, 28, 28)  # Larger batch size
    
    # Do multiple forward passes to accumulate statistics
    with torch.no_grad():
        for _ in range(10):  # Multiple passes
            output = model(x)
    
    # Switch to eval mode
    model.eval()
    
    # Check if BatchNorm statistics are initialized
    assert hasattr(model.bn1, 'running_mean'), "BatchNorm1 has no running mean"
    assert hasattr(model.bn1, 'running_var'), "BatchNorm1 has no running variance"
    
    # Check if statistics are being tracked
    assert torch.any(model.bn1.running_mean != 0), "BatchNorm running mean not updated"
    assert torch.any(model.bn1.running_var != 1), "BatchNorm running variance not updated"

def test_dropout():
    """Test if Dropout layers behave differently in train/eval modes"""
    model = SuperLightMNIST()
    x = torch.randn(100, 1, 28, 28)
    
    # Test in training mode
    model.train()
    with torch.no_grad():
        train_outputs = [model(x) for _ in range(5)]
    
    # Test in eval mode
    model.eval()
    with torch.no_grad():
        eval_outputs = [model(x) for _ in range(5)]
    
    # Calculate variations
    train_var = torch.var(torch.stack([out.sum() for out in train_outputs]))
    eval_var = torch.var(torch.stack([out.sum() for out in eval_outputs]))
    
    # Training should have more variance due to dropout
    assert train_var > eval_var, "Dropout not affecting training variance"


