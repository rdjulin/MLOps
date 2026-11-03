# **Module 2 Assignment: Model Training and Experiment Tracking Overview**

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, dropout_rate=0.2):
        super(LogisticRegression, self).__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input images
        x = x.view(x.size(0), -1)
        return self.network(x)

def unnormalize(tensor, mean=0.1307, std=0.3081):
    """Reverses the MNIST normalization so images render correctly."""
    return tensor * std + mean

# Hyperparameters to experiment with
config = {
    "learning_rate": 0.01,
    "epochs": 10,
    "batch_size": 64,
    "hidden_layers": [128, 64],
    "dropout_rate": 0.2
}

# Initialize wandb
wandb.init(project="mnist-mlops", config=config)
config = wandb.config

# Load and preprocess data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = LogisticRegression(
    input_dim=28*28,
    output_dim=10,
    hidden_layers=config.hidden_layers,
    dropout_rate=config.dropout_rate
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Training loop
for epoch in range(config.epochs):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if batch_idx % 100 == 0:
            wandb.log({"batch_loss": loss.item()})

    avg_train_loss = epoch_loss / len(train_loader)
    wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total
    avg_val_loss = val_loss / len(test_loader)
    wandb.log({"val_loss": avg_val_loss, "val_accuracy": accuracy})
    print(f"Epoch {epoch+1}/{config.epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {accuracy:.2f}%")

# Log final sample validation predictions to W&B as requested
model.eval()
images, labels = next(iter(test_loader))
with torch.no_grad():
    preds = model(images[:16]).argmax(dim=1)

unnormalized_images = unnormalize(images[:16])

wandb.log({
    "val_predictions": [
        wandb.Image(img, caption=f"true={t.item()} pred={q.item()}")
        for img, t, q in zip(unnormalized_images, labels[:16], preds)
    ]
})

# Save the trained model weights
torch.save(model.state_dict(), "model.pt")

wandb.finish()

# Initialize a new run to log post-training validation analysis
wandb.init(project="mnist-mlops", job_type="evaluation")

# Watch the model
wandb.watch(model, log="all")

# Helper to calculate gradient norm
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# Log details (gradient norm, learning rate)
current_lr = optimizer.param_groups[0]['lr']
grad_norm = get_grad_norm(model)
wandb.log({
    "learning_rate": current_lr,
    "gradient_norm": grad_norm
})

# After validation, log more metrics (precision and recall)
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

precision, recall, _, _ = precision_recall_fscore_support(
    all_targets, all_preds, average='macro', zero_division=0
)

wandb.log({
    "precision_macro": precision,
    "recall_macro": recall
})

# Helper function to un-normalize MNIST images back to [0, 1] range
def unnormalize(tensor, mean=0.1307, std=0.3081):
    return tensor * std + mean

# Visualize sample predictions with un-normalization implemented
def log_predictions(model, data, target, num_samples=10):
    model.eval()
    with torch.no_grad():
        output = model(data)
        _, preds = torch.max(output, 1)

    # Un-normalize the batch of images first
    unnormalized_data = unnormalize(data)

    images = []
    for i in range(min(num_samples, len(data))):
        img = unnormalized_data[i].squeeze().cpu().numpy()
        pred = preds[i].item()
        truth = target[i].item()
        images.append(wandb.Image(
            img,
            caption=f"Pred: {pred}, Truth: {truth}"
        ))
    wandb.log({"sample_predictions": images})

# Call this function to visualize predictions
log_predictions(model, next(iter(test_loader))[0][:10], next(iter(test_loader))[1][:10])

wandb.finish()

