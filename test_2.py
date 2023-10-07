import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import mobilenet_v3_large

# Load model
model = mobilenet_v3_large(pretrained=True)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Assuming your input size is the same as required by the model: 3x224x224
# Generate a random tensor for testing 
input_tensor = torch.randn(1, 3, 512, 512).to(device)

model = nn.Sequential(*list(model.children())[:-2])

# Forward pass through the model
output = model(input_tensor)

# Print output tensor
print(output.shape)