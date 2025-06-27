import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    フルコネクションで構成したシンプルな MLP
    """
    def __init__(self, input_dim, output_dim, hidden_sizes=(256,256)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
