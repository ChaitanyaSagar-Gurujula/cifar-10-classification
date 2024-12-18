
# MNIST Model Fine-Tuned Project

A lightweight MNIST classifier with test accuracy above 99.4% accuracy.

## Project Overview

This project implements a lightweight CNN model for MNIST digit classification with the following constraints and features:

- Model Parameters: < 8
- Training Accuracy: > 99.4%
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
git clone https://github.com/yourusername/mnist-fine-tuned.git
cd mnist-model-mlops
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
python src/train.py  --model super_light --is_train_aug True
```
or
```bash
python -m src.train --model super_light --is_train_aug True
```
This will:
- Download MNIST dataset (if not present)
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
git remote add origin https://github.com/yourusername/mnist-fine-tuned-pro.git
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

- Used 2 Convolutional blocks with Batch Normalization and Dropout for regularization.
- Used Adaptive Global Average Pooling followed by Convolution layer to map to output classes.
- Used OneCycleLR Scheduler for learning rate optimization.
- Used Adam Optimizer with weight decay for better convergence.
- Used Random Affine, Perspective Augmentations for better model generalization.

## Training Configuration

- **Optimizer:** AdamW
- **Learning Rate:** OneCycleLR (max_lr=0.01)
- **Batch Size:** 128
- **Epochs:** Configurable (default=20)

## Models

### LightMNIST:

#### Target:

 - Basic Setup with working Model of final receptive field value 28
 - Get understanding of the performance of the model with simple structure.

#### Results:
Parameters: 4738

##### Without Training Data Augmentation:

 - Best Train Accuracy: 99.80
 - Best Test Accuracy: 99.09 (15th Epoch)

##### With Training Data Augmentation:

 - Best Train Accuracy: 99.11
 - Best Test Accuracy: 99.27 (15th Epoch)

#### Analysis: 
- Initial model works fine. But target is not achieved with or without data augmentation.
- Model trained without data augmentation is causing overfitting.
- Model training with data augmentation seems fine, but it doesnt hit our test acurracy target of 99.4 within 15 epochs.
- Lets add batch normalization and dropout regularization to see if it can avoid overfitting and also for faster convergence.

#### Logs:
- [View Training Logs without Data Augmentation](./training-logs/Light%20Model%20Training%20Logs%20without%20Augmentation.log)
- [View Training Logs with Data Augmentation](./training-logs/Light%20Model%20Training%20Logs%20with%20Augmentation.log)
---
### LightestMNIST:

#### Target:

- Model of final receptive field value 28
- Efficiently using batch normalization and dropout for performance improvements.

#### Results:
Parameters: 4274

##### Without Training Data Augmentation:

 - Best Train Accuracy: 99.69
 - Best Test Accuracy: 99.30 (15th Epoch)

##### With Training Data Augmentation:

 - Best Train Accuracy: 99.18
 - Best Test Accuracy: 99.42 (14th Epoch)

#### Analysis: 
- Reached Closer to the target accuracy. 
- Overfitting got reduced in Model trained without data augmentation.
- Model training with data augmentation has hit our target accuracy for the last two epochs.
- Lets add GAP in the last layers to see if it can help in improving the performance further.

#### Logs:
- [View Training Logs without Data Augmentation](./training-logs/Lightest%20Model%20Training%20Logs%20without%20Augmentation.log)
- [View Training Logs with Data Augmentation](./training-logs/Lightest%20Model%20Training%20Logs%20with%20Augmentation.log)
---
### SuperLightMNIST:

#### Target:

- Model of final receptive field value 28
- Trying GAP to see if it can help in performance improvement.
For this, removed the transition layer(Max pooling and 1D conv) for the second convolution block  and replace it with GAP instead.
This can cause slight increase in params but it should be fine as the total params are less than threshold (8k params).

#### Results:
Parameters: 4842

##### Without Training Data Augmentation:

 - Best Train Accuracy: 99.72
 - Best Test Accuracy: 99.42 (15th Epoch)

##### With Training Data Augmentation:

 - Best Train Accuracy: 99.26
 - Best Test Accuracy: 99.48 (13th Epoch)

#### Analysis: 
- Achieved Target Accuracy!
- After adding GAP, model trained without data augmentation has hit the target accuracy but only once.
- Model training with data augmentation has hit our target accuracy consistently for the last 5 epochs.
- Used AdaptiveGAP and it has helped in improving model accuracy consistently.

#### Logs:

- [View Training Logs without Data Augmentation](./training-logs/SuperLight%20Model%20Training%20Logs%20without%20Augmentation.log)
- [View Training Logs with Data Augmentation](./training-logs/SuperLight%20Model%20Training%20Logs%20with%20Augmentation.log)
- [View CUDA Training Logs without Data Augmentation](./training-logs/CUDA%20SuperLight%20Model%20Training%20Logs%20without%20Augmentation.log)
- [View CUDA Training Logs with Data Augmentation](./training-logs/SuperLight%20Model%20Training%20Logs%20with%20Augmentation.log)

#### Cloud Training Screenshots

##### Without Data Augmentation:
![Cloud Training without Aug Pre](./cloud-training-screenshots/Cloud%20Training%20Without%20Data%20Augmentation_prefinal.png)
![Cloud Training without Aug](./cloud-training-screenshots/Cloud%20Training%20Without%20Data%20Augmentation_final.png)


##### With Data Augmentation:
![Final model Cloud Training pre](./cloud-training-screenshots/Cloud%20Training%20With%20Data%20Augmentation_prefinal.png)
![Final model Cloud Training](./cloud-training-screenshots/Cloud%20Training%20With%20Data%20Augmentation_final.png)

#### PyTorch Model Files obtained from CUDA Training

##### Without Data Augmentation:
 [SuperLightMNIST No Data Aug Model](./best_model_cuda_no_data_aug.pth)


##### With Data Augmentation:
 [SuperLightMNIST With Data Aug Model](./best_model_cuda.pth)

#### PyTorch Model Files obtained from CPU Training
 [SuperLightMNIST With Data Aug Model In CPU](./best_model.pth)

Note: Only Final best model is saved in CPU.

---

## License

Distributed under the MIT License. See LICENSE for more information.


