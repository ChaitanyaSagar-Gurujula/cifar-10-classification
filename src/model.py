import torch
import torch.nn as nn
import torch.nn.functional as F


class LightningCIFAR10(nn.Module):
    def __init__(self):
        super(LightningCIFAR10, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16) 
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1,  dilation=2)

        self.conv5 = nn.Conv2d(8, 12, kernel_size=3, padding=1)  
        self.conv6 = nn.Conv2d(12, 16, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)
        self.conv8 = nn.Conv2d(32, 64, kernel_size=3, padding=1,  dilation=2)

        self.conv9 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  
        self.conv10 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)  
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64) 
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=2)

        # Fully connected layer to map to output classes
        self.fc1 = nn.Linear(64 * 3 * 3, 144)  
        self.fc2 = nn.Linear(144, 10)  

        # Batch normalization layers for each convolutional block
        self.bn1 = nn.BatchNorm2d(8)   
        self.bn2 = nn.BatchNorm2d(16)  
        self.bn3 = nn.BatchNorm2d(16)  
        self.bn4 = nn.BatchNorm2d(32)  
        self.bn5 = nn.BatchNorm2d(12)  
        self.bn6 = nn.BatchNorm2d(16)  
        self.bn7 = nn.BatchNorm2d(16)  
        self.bn8 = nn.BatchNorm2d(64)  
        self.bn9 = nn.BatchNorm2d(32)  
        self.bn10 = nn.BatchNorm2d(64) 
        self.bn11 = nn.BatchNorm2d(64) 
        self.bn12 = nn.BatchNorm2d(64) 

        # 1D Convolution for pointwise convolution after depthwise seperable convolution
        self.conv1d_1 = nn.Conv1d(16, 32, kernel_size=1)  
        self.bn1d_1 = nn.BatchNorm1d(32)  

        # 1D Convolution for channel reduction
        self.conv1d_2 = nn.Conv1d(32, 8, kernel_size=1)  
        self.bn1d_2 = nn.BatchNorm1d(8) 

        # 1D Convolution for pointwise convolution after depthwise seperable convolution
        self.conv1d_3 = nn.Conv1d(16, 32, kernel_size=1) 
        self.bn1d_3 = nn.BatchNorm1d(32)  

        # 1D Convolution for channel reduction
        self.conv1d_4 = nn.Conv1d(64, 16, kernel_size=1)  
        self.bn1d_4 = nn.BatchNorm1d(16) 

        # 1D Convolution for pointwise convolution after depthwise seperable convolution
        self.conv1d_5 = nn.Conv1d(64, 64, kernel_size=1) 
        self.bn1d_5 = nn.BatchNorm1d(64) 

        # Dropout layers for regularization
        self.dropout_1 = nn.Dropout(0.05)  # Dropout with probability 0.05

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)  
        x = self.bn1(x)    
        x = F.relu(x)      
        x = self.dropout_1(x)  
        x = self.conv2(x)  
        x = self.bn2(x)    
        x = F.relu(x)      
        x = self.dropout_1(x)  
        x = self.conv3(x)  
        x = self.bn3(x)    
        x = F.relu(x)      
        x = self.dropout_1(x) 

        # Reshape for 1D convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 16, -1)  

        # First 1D convolution
        x = self.conv1d_1(x) 
        x = self.bn1d_1(x)   
        x = F.relu(x)        

        # Reshape back to 2D
        x = x.view(batch_size, 32, 32, 32)  

        x = self.conv4(x)  
        x = self.bn4(x)    
        x = F.relu(x)      
        x = self.dropout_1(x)  


        # Reshape for 1D convolution
        x = x.view(batch_size, 32, -1) 

        # First 1D convolution
        x = self.conv1d_2(x)  
        x = self.bn1d_2(x)    
        x = F.relu(x)         

        # Reshape back to 2D
        x = x.view(batch_size, 8, 30, 30)  


        # Second Convolution Block
        x = self.conv5(x)  
        x = self.bn5(x)    
        x = F.relu(x)      
        x = self.dropout_1(x)  
        x = self.conv6(x)  
        x = self.bn6(x)    
        x = F.relu(x)      
        x = self.dropout_1(x)  
        x = self.conv7(x)  
        x = self.bn7(x)    
        x = F.relu(x)      
        x = self.dropout_1(x)  

        # Reshape for 1D convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 16, -1)

        # First 1D convolution
        x = self.conv1d_3(x) 
        x = self.bn1d_3(x)   
        x = F.relu(x)      

        # Reshape back to 2D
        x = x.view(batch_size, 32, 30, 30) 

        x = self.conv8(x) 
        x = self.bn8(x)   
        x = F.relu(x)     
        x = self.dropout_1(x) 


        # Reshape for 1D convolution
        x = x.view(batch_size, 64, -1) 

        # First 1D convolution
        x = self.conv1d_4(x)  
        x = self.bn1d_4(x)    
        x = F.relu(x)         

        # Reshape back to 2D
        x = x.view(batch_size, 16, 28, 28) 

        # Third Convolution Block
        x = self.conv9(x)  
        x = self.bn9(x)    
        x = F.relu(x)      
        x = self.dropout_1(x)  
        x = self.conv10(x)
        x = self.bn10(x)  
        x = F.relu(x)     
        x = self.dropout_1(x)  
        x = self.conv11(x) 
        x = self.bn11(x)   
        x = F.relu(x)      
        x = self.dropout_1(x) 

        # Reshape for 1D convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 64, -1) 

        # First 1D convolution
        x = self.conv1d_5(x) 
        x = self.bn1d_5(x)   
        x = F.relu(x)        

        # Reshape back to 2D
        x = x.view(batch_size, 64, 14, 14) 

        x = self.conv12(x) 
        x = self.bn12(x)   
        x = F.relu(x)      
        x = self.dropout_1(x)  

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (3, 3))  

        # Flatten and fully connected layer
        x = x.view(batch_size, -1)  
        x = self.fc1(x)  
        x = self.fc2(x)  

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
