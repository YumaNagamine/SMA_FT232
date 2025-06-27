import torch
from src.agent.network import MLP

def test_mlp_output_shape():
    model = MLP(input_dim=8, output_dim=2, hidden_sizes=(16, 16))
    x = torch.randn(5, 8)
    y = model(x)
    assert y.shape == (5, 2)
