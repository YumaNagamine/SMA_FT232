import torch
# import torchvision
# import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np

# image-based control

class AngleNet(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layer_size: np.ndarray):
        super().__init__()
        self.net1 = torch.nn.Linear(input_dim,hidden_layer_size[0])
        self.net2 = torch.nn.Linear(hidden_layer_size[0], hidden_layer_size[1])
        self.net3 = torch.nn.Linear(hidden_layer_size[1], output_dim)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.net1(x)
        x = torch.relu(x)
        x = self.net2(x)
        x = torch.relu(x)
        x = self.net3(x)
        x = torch.sigmoid(x)
        return x