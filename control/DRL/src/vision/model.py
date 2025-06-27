import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    64x64 入力を想定した小型CNNエンコーダ
    """
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        # conv3 出力チャネル × (64/8)² = 64 × 8 × 8 = 4096
        self.fc = nn.Linear(4096, output_dim)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = h.view(h.size(0), -1)
        return self.fc(h)
