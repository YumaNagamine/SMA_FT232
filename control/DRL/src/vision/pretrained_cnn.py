import torch
import torch.nn as nn
from torchvision import models

class PretrainedCNN(nn.Module):
    """
    ImageNet pretrained ResNet18 をバックボーンに使い、
    最終特徴量を output_dim 次元に射影するクラス。
    """
    def __init__(self, output_dim: int, pretrained: bool = True):
        super().__init__()
        # ResNet18 の全層取得
        backbone = models.resnet18(pretrained=pretrained)
        # 最終の fc 層を除去 (バックボーン出力は 512 次元)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        # 必要に応じて微調整するための射影層
        self.projector = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)  前処理済み画像
        return: (B, output_dim) 特徴ベクトル
        """
        h = self.feature_extractor(x)        # → (B, 512, 1, 1)
        h = h.view(h.size(0), -1)            # → (B, 512)
        return self.projector(h)             # → (B, output_dim)
