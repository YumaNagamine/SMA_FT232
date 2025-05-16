# Define nn.Module

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleJointNet(nn.Module):
    def __init__(self, input_ch=1, hidden_dim=256, output_dim=8):
        super().__init__()
        self.conv1 = nn.Conv2d(input_ch, 16, kernel_size=20, stride=4, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=20, stride=4, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=20, stride=8, padding=2)

        # 全結合層
        self.fc1 = nn.Linear(32 * 7 * 11, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        if x.dim() == 4 and x.size(1) == 3:
            r = x[:, 0:1, :, :]
            g = x[:, 1:2, :, :]
            b = x[:, 2:3, :, :]
            x = 0.299 * r + 0.587 * g + 0.114 * b  # -> (B,1,H,W)
            # x = (x > 100).float()

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    