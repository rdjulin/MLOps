# MLOps

Assignment 2:

MLOps Microcredential ISU October 2026

readme_content = MNIST MLOps Project

This project demonstrates how to train a Multi-Layer Perceptron (MLP) on the MNIST dataset using PyTorch, with full experiment tracking and post-training validation analysis integrated via Weights & Biases (W&B).

Project Overview


Weights & Biases Live Reports
All real-time monitoring loss dynamics, precision charts, and evaluation artifacts can be observed on your project dashboard:

👉 View Weights & Biases Project & Report Dashboard


From Assignment 1:
## Project Overview
- **Model**: A flexible Multi-Layer Perceptron (MLP) supporting customizable hidden dimensions and dropout layers.
- **Framework**: PyTorch & Torchvision.
- **Experiment Tracking**: Managed via [Weights & Biases (W&B)](https://wandb.ai/), tracking training losses, validation accuracies, learning rates, gradient norms, classification metrics (Macro Precision/Recall), and visual sample predictions.
- **Model Architecture:** Multi-Layer Perceptron (LogisticRegression) featuring flexible hidden dimensions, ReLU activations, and Dropout layer regularization built using PyTorch.
**Experiment Monitoring:** Fully integrated with Weights & Biases (W&B) to track hyperparameter sweeps, metric variations, confusion matrices, and unnormalized visual batch validation predictions.
**Production API Server:** Microservice backend powered by FastAPI to serve safe and asynchronous model inference over HTTP POST requests.
**Containerization Engine:** High-performance dockerized packaging utilizing lightweight Python slim baselines paired with CPU-only PyTorch wheels to optimize distribution footprints.
**CI/CD Architecture:** Complete GitHub Actions automation structure configuring validation pipelines via pytest, pushing containerized images to Docker Hub, and deploying live production containers on Heroku registries.

## Local Setup and Running Instructions

### 1. Clone & Setup Environment
Ensure Python 3.10+ is installed on your local host. Clone your workspace contents and install package dependencies:
```bash
pip install torch torchvision wandb scikit-learn numpy
```

pip install -r requirements.txt
pip install pytest httpx

### 2. Model Training
Run your training routine to generate and serialize the network weights locally into model.pt:

python train.py

### 3. Executing Unit & Integration Tests
Execute the testing suite via pytest to assert model layer tensor integrity and validate endpoint requests:

pytest test_deployment.py
4. Running the Local API Server
Launch your local FastAPI instance on host port 8000:

uvicorn app:app --host 0.0.0.0 --port 8000
Once started, navigate to http://localhost:8000/docs to interact with Swagger UI and trigger test payloads.

### Instructions for Building and Running the Docker Container
### 1. Build the Docker Image
Compile the application container locally under the tag mnist-fastapi:latest:

docker build -t mnist-fastapi:latest .
2. Run the Containerized App
Run the Docker image while binding port 8000 on your local machine to the container's internal port:

docker run -p 8000:8000 mnist-fastapi:latest
Verify by submitting image requests to http://localhost:8000/predict/.


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

