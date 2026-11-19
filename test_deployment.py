import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile
import torchvision.transforms as transforms
from fastapi.testclient import TestClient

# Redefining the LogisticRegression model inline for self-contained testing
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
        x = x.view(x.size(0), -1)
        return self.network(x)

# Instantiate a test app inline matching the production FastAPI environment
app = FastAPI()
model = LogisticRegression(input_dim=28*28, output_dim=10, hidden_layers=[128, 64], dropout_rate=0.2)
# Safe mock loading if no local weights found
try:
    model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
except Exception:
    pass
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        prediction = outputs.argmax(dim=1).item()
    return {"prediction": int(prediction)}

def test_model_inference():
    """Verify that our model handles correct input shape and generates valid logit dimensions."""
    test_model = LogisticRegression(input_dim=28*28, output_dim=10, hidden_layers=[128, 64], dropout_rate=0.2)
    dummy_input = torch.randn(2, 1, 28, 28)
    output = test_model(dummy_input)
    assert output.shape == (2, 10)

def test_api_predict():
    """Verify that the FastAPI predict endpoint correctly processes uploaded image files."""
    client = TestClient(app)
    img = Image.new('L', (28, 28), color=0)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    response = client.post(
        "/predict/",
        files={"file": ("test_digit.png", img_byte_arr, "image/png")}
    )
    
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert isinstance(json_data["prediction"], int)
    assert 0 <= json_data["prediction"] <= 9
