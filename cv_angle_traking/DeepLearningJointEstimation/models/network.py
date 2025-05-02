# Define nn.Module

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleJointNet(nn.Module):
    def __init__(self, input_ch=3, hidden_dim=256, output_dim=8):
        super().__init__()
        # 畳み込み層でダウンサンプリング
        self.conv1 = nn.Conv2d(input_ch, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)

        # 全結合層
        # 出力サイズ: バッチ x (128 * (1200/8) * (1600/8))
        self.fc1 = nn.Linear(128 * 150 * 200, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)