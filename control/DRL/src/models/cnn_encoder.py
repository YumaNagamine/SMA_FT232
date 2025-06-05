import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """
    - シンプルな畳み込み層×数段 + 平坦化 + 全結合 で画像特徴を抽出
    - 目標: (3,H,W) を入力 → output_dim 次元のベクトルを返す
    - 現在画像と目標画像の両方に同一重みを適用 → 重み共有で FIFO
    """

    def __init__(self, output_dim=256, image_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),  # -> (32, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),              # -> (64, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # -> (128, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),            # -> (256, H/16, W/16)
            nn.ReLU(),
        )
        # conv 出力の空間サイズは (256, H/16, W/16) になるので
        # 平均プーリングかフラット化する
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (256,1,1)
        self.fc = nn.Linear(256, output_dim)               # -> (output_dim)

    def forward(self, x):
        """
        x: Tensor (B, 3, H, W), [0,1] float
        Returns: Tensor (B, output_dim)
        """
        h = self.conv(x)                     # (B,256, H/16, W/16)
        h = self.adaptive_pool(h)            # (B,256,1,1)
        h = h.view(h.size(0), -1)            # (B,256)
        out = self.fc(h)                     # (B, output_dim)
        return out                           # 例: (B,256)
