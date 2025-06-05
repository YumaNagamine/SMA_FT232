import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    - Critic は 2つの Q1, Q2 を持つ構造
    - 入力: s_feat (B,512) と action (B,6) → 結合して (B, 512+6)
    - 出力: Q 値スカラー (B,1)
    """

    def __init__(self, state_dim=512, action_dim=6, hidden_dim=256):
        super().__init__()
        # Q1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        # Q2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, s_feat, a):
        xu = torch.cat([s_feat, a], dim=-1)  # (B, 512+6)
        # Q1
        h1 = F.relu(self.q1_fc1(xu))
        h1 = F.relu(self.q1_fc2(h1))
        q1 = self.q1_out(h1)
        # Q2
        h2 = F.relu(self.q2_fc1(xu))
        h2 = F.relu(self.q2_fc2(h2))
        q2 = self.q2_out(h2)
        return q1, q2  # それぞれ (B,1)
