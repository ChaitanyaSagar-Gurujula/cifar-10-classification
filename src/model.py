import torch
import torch.nn as nn
import torch.nn.functional as F

class LightningCIFAR10(nn.Module):
    def __init__(self):
        super(LightningCIFAR10, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)  # Input: 3x32x32 -> Output: 4x30x30
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # Input: 4x30x30 -> Output: 8x28x28
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)  # Input: 8x28x28 -> Output: 8x26x26
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=2)  # Input: 8x26x26 -> Output: 8x24x24

        self.conv5 = nn.Conv2d(8, 16, kernel_size=3, padding=0)  # Input: 3x32x32 -> Output: 4x30x30
        self.conv6 = nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2)  # Input: 4x30x30 -> Output: 8x28x28
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, padding=0, groups=32)  # Input: 8x28x28 -> Output: 8x26x26
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=0, dilation=2)  # Input: 8x26x26 -> Output: 8x24x24

        self.conv9 = nn.Conv2d(64, 10, kernel_size=3, padding=0)  # Input: 8x26x26 -> Output: 8x24x24

        # Batch normalization layers for each convolutional block
        self.bn1 = nn.BatchNorm2d(8)   # BatchNorm after conv1
        self.bn2 = nn.BatchNorm2d(16)   # BatchNorm after conv2
        self.bn3 = nn.BatchNorm2d(16)   # BatchNorm after conv3
        self.bn4 = nn.BatchNorm2d(32)   # BatchNorm after conv4
        self.bn5 = nn.BatchNorm2d(16)   # BatchNorm after conv3
        self.bn6 = nn.BatchNorm2d(32)  # BatchNorm after conv4
        self.bn7 = nn.BatchNorm2d(32)  # BatchNorm after conv5
        self.bn8 = nn.BatchNorm2d(64)  # BatchNorm after conv5

        # 1D Convolution for pointwise convolution after depthwise seperable convolution
        self.conv1d_1 = nn.Conv1d(16, 32, kernel_size=1)  # Input: (batch, 8, 144) -> Output: (batch, 4, 144)
        self.bn1d_1 = nn.BatchNorm1d(32)  # BatchNorm for conv1d_1

        # 1D Convolution for channel reduction
        self.conv1d_2 = nn.Conv1d(32, 8, kernel_size=1)  # Input: (batch, 8, 144) -> Output: (batch, 4, 144)
        self.bn1d_2 = nn.BatchNorm1d(8)  # BatchNorm for conv1d_1

        # 1D Convolution for pointwise convolution after depthwise seperable convolution
        self.conv1d_3 = nn.Conv1d(32, 64, kernel_size=1)  # Input: (batch, 8, 144) -> Output: (batch, 4, 144)
        self.bn1d_3 = nn.BatchNorm1d(64)  # BatchNorm for conv1d_1

        # Dropout layers for regularization
        self.dropout_1 = nn.Dropout(0.05)  # Dropout with probability 0.05

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)  # Conv1: (batch, 4, 26, 26)
        x = self.bn1(x)    # BatchNorm after conv1
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv2(x)  # Conv2: (batch, 8, 24, 24)
        x = self.bn2(x)    # BatchNorm after conv2
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization

        x = self.conv3(x)  # Conv2: (batch, 8, 24, 24)
        x = self.bn3(x)    # BatchNorm after conv2
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization

        # Reshape for 1D convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 16, -1)  # Reshape to (batch, 8, 196)

        # First 1D convolution
        x = self.conv1d_1(x)  # Conv1d_1: (batch, 4, 144)
        x = self.bn1d_1(x)    # BatchNorm after conv1d_1
        x = F.relu(x)         # ReLU activation

        # Reshape back to 2D
        x = x.view(batch_size, 32, 32, 32)  # Reshape to (batch, 4, 12, 12)

        x = self.conv4(x)  # Conv2: (batch, 8, 24, 24)
        x = self.bn4(x)    # BatchNorm after conv2
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization

        # Reshape for 1D convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 32, -1)  # Reshape to (batch, 8, 196)

        # First 1D convolution
        x = self.conv1d_2(x)  # Conv1d_1: (batch, 4, 144)
        x = self.bn1d_2(x)    # BatchNorm after conv1d_1
        x = F.relu(x)         # ReLU activation

        # Reshape back to 2D
        x = x.view(batch_size, 8, 30, 30)  # Reshape to (batch, 4, 12, 12)

        # Second convolutional block
        # First convolutional block
        x = self.conv5(x)  # Conv1: (batch, 4, 26, 26)
        x = self.bn5(x)    # BatchNorm after conv1
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv6(x)  # Conv2: (batch, 8, 24, 24)
        x = self.bn6(x)    # BatchNorm after conv2
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization

        x = self.conv7(x)  # Conv2: (batch, 8, 24, 24)
        x = self.bn7(x)    # BatchNorm after conv2
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization

        # Reshape for 1D convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 32, -1)  # Reshape to (batch, 8, 196)

        # First 1D convolution
        x = self.conv1d_3(x)  # Conv1d_1: (batch, 4, 144)
        x = self.bn1d_3(x)    # BatchNorm after conv1d_1
        x = F.relu(x)         # ReLU activation

        # Reshape back to 2D
        x = x.view(batch_size, 64, 11, 11)  # Reshape to (batch, 4, 12, 12)

        x = self.conv8(x)  # Conv2: (batch, 8, 24, 24)
        x = self.bn8(x)    # BatchNorm after conv2
        x = F.relu(x)      # ReLU activation
        #x = self.dropout_1(x)  # Dropout for regularization

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (3, 3))  # Adaptive average pooling: (batch, 16, 3, 3)

        x = self.conv9(x)  # Conv6: (batch, 10, 1, 1)

        # Flatten and fully connected layer
        x = x.view(batch_size, -1)  # Flatten to (batch, 10*1*1)

        return F.log_softmax(x, dim=1)  # Log-Softmax for classification
    
class LightningCIFAR10_1(nn.Module):
    def __init__(self):
        super(LightningCIFAR10_1, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=0)  # Input: 3x32x32 -> Output: 4x30x30
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=0)  # Input: 4x30x30 -> Output: 8x28x28
        self.conv3_1 = nn.Conv2d(8, 8, kernel_size=3, padding=0)  # Input: 8x28x28 -> Output: 8x26x26
        self.conv4_1 = nn.Conv2d(8, 8, kernel_size=3, padding=0)  # Input: 8x26x26 -> Output: 8x24x24
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, padding=0)  # Input: 4x12x12 -> Output: 8x10x10
        self.conv4 = nn.Conv2d(8, 12, kernel_size=3, padding=0)  # Input: 8x10x10 -> Output: 12x8x8
        self.conv5 = nn.Conv2d(12, 16, kernel_size=3, padding=0)  # Input: 12x8x8 -> Output: 16x6x6
        self.conv6 = nn.Conv2d(16, 10, kernel_size=3, padding=0)  # Input: 16x3x3 -> Output: 10x1x1


        # Batch normalization layers for each convolutional block
        self.bn1 = nn.BatchNorm2d(4)   # BatchNorm after conv1
        self.bn2 = nn.BatchNorm2d(8)   # BatchNorm after conv2
        self.bn3_1 = nn.BatchNorm2d(8)   # BatchNorm after conv3
        self.bn4_1 = nn.BatchNorm2d(8)   # BatchNorm after conv4
        self.bn3 = nn.BatchNorm2d(8)   # BatchNorm after conv3
        self.bn4 = nn.BatchNorm2d(12)  # BatchNorm after conv4
        self.bn5 = nn.BatchNorm2d(16)  # BatchNorm after conv5

        # 1D Convolution for channel reduction
        self.conv1d_1 = nn.Conv1d(8, 4, kernel_size=1)  # Input: (batch, 8, 144) -> Output: (batch, 4, 144)
        self.bn1d_1 = nn.BatchNorm1d(4)  # BatchNorm for conv1d_1

        # Dropout layers for regularization
        self.dropout_1 = nn.Dropout(0.05)  # Dropout with probability 0.05

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)  # Conv1: (batch, 4, 26, 26)
        x = self.bn1(x)    # BatchNorm after conv1
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv2(x)  # Conv2: (batch, 8, 24, 24)
        x = self.bn2(x)    # BatchNorm after conv2
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv3_1(x)  # Conv2: (batch, 8, 24, 24)
        x = self.bn3_1(x)    # BatchNorm after conv2
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv4_1(x)  # Conv2: (batch, 8, 24, 24)
        x = self.bn4_1(x)    # BatchNorm after conv2
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = F.max_pool2d(x, 2)  # Max pooling: (batch, 8, 12, 12)

        # Reshape for 1D convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 8, -1)  # Reshape to (batch, 8, 196)

        # First 1D convolution
        x = self.conv1d_1(x)  # Conv1d_1: (batch, 4, 144)
        x = self.bn1d_1(x)    # BatchNorm after conv1d_1
        x = F.relu(x)         # ReLU activation

        # Reshape back to 2D
        x = x.view(batch_size, 4, 12, 12)  # Reshape to (batch, 4, 12, 12)

        # Second convolutional block
        x = self.conv3(x)  # Conv3: (batch, 8, 10, 10)
        x = self.bn3(x)    # BatchNorm after conv3
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv4(x)  # Conv4: (batch, 12, 8, 8)
        x = self.bn4(x)    # BatchNorm after conv4
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv5(x)  # Conv5: (batch, 16, 6, 6)
        x = self.bn5(x)    # BatchNorm after conv5
        x = F.relu(x)      # ReLU activation

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (3, 3))  # Adaptive average pooling: (batch, 16, 3, 3)

        x = self.conv6(x)  # Conv6: (batch, 10, 1, 1)

        # Flatten and fully connected layer
        x = x.view(batch_size, -1)  # Flatten to (batch, 10*1*1)

        return F.log_softmax(x, dim=1)  # Log-Softmax for classification
    

# Dictionary mapping model names to their classes
MODEL_REGISTRY = {
    'light': LightningCIFAR10
}

def get_model(model_name):
    """Factory function to get a model by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]
