
# CIFAR10 Classification Project

A CIFAR 10 classifier with test accuracy above 85% accuracy.

## Project Overview

This project implements a CNN model for CIFAR10 classification with the following constraints and features:

- Model Parameters: < 200K
- Test Accuracy: > 99.4%
- Model Performance Tracking
- Code Quality Checks

## Project Structure

```
project_root/
│
├── src/
│   ├── __init__.py
│   ├── model.py      # Model architecture
│   ├── train.py      # Training script
│   └── dataset.py    # Data loading utilities
│
├── tests/
│   ├── __init__.py
│   └── test_model.py # Test cases
│
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml  # CI/CD pipeline
│
├── setup.py          # Package setup
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cifar-10-classification.git
cd cifar-10-classification
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate # on Windows use `.venv\Scripts\activate`
```

3. Install dependencies and package:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Training the Model

To train the model, run the following command:

```bash
# From project root directory
export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/train.py  --model light --is_train_aug True
```
or
```bash
python -m src.train --model light --is_train_aug True
```
This will:
- Download CIFAR10 dataset (if not present)
- Train the model for specified epochs
- Save the best model as 'best_model.pth'
- Display training progress and results

### Running Tests

To run the test suite, use the following commands:

```bash
# Run all tests
pytest tests/ -v
```

## GitHub Setup and CI/CD

1. Create a new GitHub repository.

2. Push your code to GitHub:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/cifar-10-classification.git
git push -u origin main
```

3. GitHub Actions will automatically:
   - Run tests on every push and pull request.
   - Check code formatting.
   - Generate test coverage reports.
   - Create releases with trained models (on main branch).

4. View pipeline results:
   - Go to your repository on GitHub.
   - Navigate to **Actions** tab.
   - Click on the latest run.
   - Click on the **ml-pipeline** workflow.
   - Click on the job you're interested in (e.g. tests or linting).
   - Click on the log link under "Logs".

Note: Currently Github Actions won't be triggered while pushing changes to Github.

## Model Architecture

``` bash 
Model Architecture:
└── Total Parameters: 184,982
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 32, 32]             224
       BatchNorm2d-2            [-1, 8, 32, 32]              16
           Dropout-3            [-1, 8, 32, 32]               0
            Conv2d-4           [-1, 16, 32, 32]           1,168
       BatchNorm2d-5           [-1, 16, 32, 32]              32
           Dropout-6           [-1, 16, 32, 32]               0
            Conv2d-7           [-1, 16, 32, 32]             160
       BatchNorm2d-8           [-1, 16, 32, 32]              32
           Dropout-9           [-1, 16, 32, 32]               0
           Conv1d-10             [-1, 32, 1024]             544
      BatchNorm1d-11             [-1, 32, 1024]              64
           Conv2d-12           [-1, 32, 30, 30]           9,248
      BatchNorm2d-13           [-1, 32, 30, 30]              64
          Dropout-14           [-1, 32, 30, 30]               0
           Conv1d-15               [-1, 8, 900]             264
      BatchNorm1d-16               [-1, 8, 900]              16
           Conv2d-17           [-1, 12, 30, 30]             876
      BatchNorm2d-18           [-1, 12, 30, 30]              24
          Dropout-19           [-1, 12, 30, 30]               0
           Conv2d-20           [-1, 16, 30, 30]           1,744
      BatchNorm2d-21           [-1, 16, 30, 30]              32
          Dropout-22           [-1, 16, 30, 30]               0
           Conv2d-23           [-1, 16, 30, 30]             160
      BatchNorm2d-24           [-1, 16, 30, 30]              32
          Dropout-25           [-1, 16, 30, 30]               0
           Conv1d-26              [-1, 32, 900]             544
      BatchNorm1d-27              [-1, 32, 900]              64
           Conv2d-28           [-1, 64, 28, 28]          18,496
      BatchNorm2d-29           [-1, 64, 28, 28]             128
          Dropout-30           [-1, 64, 28, 28]               0
           Conv1d-31              [-1, 16, 784]           1,040
      BatchNorm1d-32              [-1, 16, 784]              32
           Conv2d-33           [-1, 32, 28, 28]           4,640
      BatchNorm2d-34           [-1, 32, 28, 28]              64
          Dropout-35           [-1, 32, 28, 28]               0
           Conv2d-36           [-1, 64, 14, 14]          18,496
      BatchNorm2d-37           [-1, 64, 14, 14]             128
          Dropout-38           [-1, 64, 14, 14]               0
           Conv2d-39           [-1, 64, 14, 14]             640
      BatchNorm2d-40           [-1, 64, 14, 14]             128
          Dropout-41           [-1, 64, 14, 14]               0
           Conv1d-42              [-1, 64, 196]           4,160
      BatchNorm1d-43              [-1, 64, 196]             128
           Conv2d-44           [-1, 64, 12, 12]          36,928
      BatchNorm2d-45           [-1, 64, 12, 12]             128
          Dropout-46           [-1, 64, 12, 12]               0
           Linear-47                  [-1, 144]          83,088
           Linear-48                   [-1, 10]           1,450
================================================================
Total params: 184,982
Trainable params: 184,982
Non-trainable params: 0
----------------------------------------------------------------
```

**Highlights**:
- Used 3 Convolutional blocks with Batch Normalization and Dropout for regularization.
- Used Adaptive Global Average Pooling followed by FC layer to map to output classes.
- Used OneCycleLR Scheduler for learning rate optimization.
- Used Adam Optimizer with weight decay for better convergence.
- Used Albumentation augmentation library for better model generalization.


## Training Configuration

- **Optimizer:** AdamW
- **Learning Rate:** OneCycleLR (max_lr=0.01)
- **Batch Size:** 128
- **Epochs:** Configurable (default=200)

#### Logs:
[View Training Logs](./cuda_training_logs.log)
``` bash 
==================================================
TRAINING COMPLETED - FINAL RESULTS
==================================================

Model Architecture:
└── Total Parameters: 184,982

Best Model achieved at Epoch 182:
├── Test Accuracy: 86.38%
└── Test Loss: 0.4861

Model saved as: 'best_model.pth'
==================================================
```


## License

Distributed under the MIT License. See LICENSE for more information.


