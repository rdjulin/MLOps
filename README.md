# MLOps
MLOps Microcredential ISU October 2026

readme_content = """# MNIST MLOps Project

This project demonstrates how to train a Multi-Layer Perceptron (MLP) on the MNIST dataset using PyTorch, with full experiment tracking and post-training validation analysis integrated via Weights & Biases (W&B).

## Project Overview
- **Model**: A flexible Multi-Layer Perceptron (MLP) supporting customizable hidden dimensions and dropout layers.
- **Framework**: PyTorch & Torchvision.
- **Experiment Tracking**: Managed via [Weights & Biases (W&B)](https://wandb.ai/), tracking training losses, validation accuracies, learning rates, gradient norms, classification metrics (Macro Precision/Recall), and visual sample predictions.

## Local Setup and Running Instructions

### 1. Clone & Setup Environment
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install torch torchvision wandb scikit-learn numpy
```

### 2. Configure W&B Login
Authenticate your local terminal with Weights & Biases:
```bash
wandb login
```

### 3. Run Training
Run the training script or notebook cells. Training runs log directly to your `mnist-mlops` project. 
Upon completion, the final trained weights are saved in the root folder as `model.pt`:
```python
# PyTorch model weights saved locally
torch.save(model.state_dict(), "model.pt")
```

### 4. Run Post-Training Evaluation
The second stage evaluates the model performance and logs extra analysis (like gradient norms, macro metrics, and visual table projections of test images) under an `evaluation` job type.

## Weights & Biases Report
You can view the interactive project reports, charts, and hyperparameter sweeps here:
- **W&B Project Report**: https://wandb.ai/r-d-julin-iowa-state-university/mnist-mlops
"""

View project at: https://wandb.ai/r-d-julin-iowa-state-university/mnist-mlops

