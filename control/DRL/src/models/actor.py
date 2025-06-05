import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Actor(nn.Module):
    """
    - 状態特徴 s_feat (B, 512) を入力 (cnn(cur_img)+cnn(goal_img) を連結して512)
    - 出力: 6次元の μ と 6次元の logσ
    - サンプリング → tanh → [−1,1] → *0.5 + 0.5 → [0,1]
    """

    def __init__(self, state_dim=512, action_dim=6, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_layer = nn.Linear(hidden_dim, action_dim)       # μ
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)    # logσ

    def forward(self, s_feat):
        """
        s_feat: (B, state_dim)
        Returns: mean (B,6), log_std (B,6)
        """
        h = F.relu(self.fc1(s_feat))
        h = F.relu(self.fc2(h))
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, s_feat):
        """
        - サンプリングして (action, log_prob) を返す
        """
        mean, log_std = self.forward(s_feat)   # (B,6), (B,6)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()                 # reparameterization trick
        y_t = torch.tanh(x_t)                  # (B,6) in [-1,1]
        action = (y_t + 1) / 2                 # scale to [0,1]

        # 補正付き log_prob の計算
        log_prob = normal.log_prob(x_t)
        # tanh のヤコビアン補正: log_prob -= log(1 - y_t^2 + eps)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (B,1)

        mean_action = torch.tanh(mean) / 2 + 0.5
        return action, log_prob, mean_action

    def to(self, device):
        super().to(device)
        return self
