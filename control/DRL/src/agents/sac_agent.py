import os
import torch
import numpy as np
from torch.optim import Adam

from src.models.actor import Actor
from src.models.critic import QNetwork
from src.utils.replay_buffer import ReplayBuffer

class SACAgent:
    def __init__(self, config, device):
        self.device = device

        # --- 1) ネットワーク構築 ---
        # CNNEncoder は別プロセスか train_sac.py 内で使う想定
        # ここでは s_feat_dim = 256(cur) + 256(goal) = 512
        state_feat_dim = 512
        action_dim = 6
        hidden_dim = config.get("hidden_dim", 256)

        self.actor = Actor(
            state_dim=state_feat_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.critic = QNetwork(
            state_dim=state_feat_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.critic_target = QNetwork(
            state_dim=state_feat_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        # ターゲット初期化
        self.critic_target.load_state_dict(self.critic.state_dict())

        # --- 2) オプティマイザ ---
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config["actor_lr"])
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config["critic_lr"])

        # --- 3) リプレイバッファ ---
        self.buffer = ReplayBuffer(
            max_size=config["buffer_size"],
            state_dim=state_feat_dim,
            action_dim=action_dim,
            device=self.device
        )

        # --- 4) ハイパーパラメータ ---
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.alpha = config["alpha"]  # エントロピー係数 (固定)
        self.batch_size = config["batch_size"]
        self.start_train_threshold = config["start_train_threshold"]
        self.train_freq = config["train_freq"]
        self.policy_update_freq = config["policy_update_freq"]
        self.target_update_freq = config["target_update_freq"]
        self.total_it = 0

        # --- 5) 初期ポリシー (BC) のロード (オプション) ---
        bc_path = config.get("bc_pretrain_path", None)
        if bc_path is not None and os.path.isfile(bc_path):
            bc_weights = torch.load(bc_path, map_location=self.device)
            # Actor の μ を含む部分と互換がある前提でロード
            self.actor.load_state_dict(bc_weights, strict=False)
            print(f"[SACAgent] Loaded BC weights from {bc_path}")

    def select_action(self, s_feat, evaluate=False):
        """
        - s_feat: Torch Tensor (1,512) or (batch_size,512)
        - evaluate が True なら、μ のみを返す
        """
        s_feat = torch.FloatTensor(s_feat).to(self.device)
        if evaluate:
            with torch.no_grad():
                mean, log_std = self.actor(s_feat)
                y_t = torch.tanh(mean)
                action = (y_t + 1) / 2
                return action.cpu().numpy()
        else:
            with torch.no_grad():
                action, _, _ = self.actor.sample(s_feat)
                return action.cpu().numpy()

    def train_step(self, **kwargs):
        """
        - ここでは構造化していないので、
          train_sac.py のループ内で呼び出したほうが管理しやすい
        """
        pass  # train_sac.py 内で実装

    def update(self):
        """
        - リプレイバッファからサンプルを取得し、
          Critic と Actor を同時に更新
        """
        if self.buffer.size < self.start_train_threshold:
            return

        self.total_it += 1

        # --- 1) バッチサンプリング ---
        state, action, reward, next_state, not_done = self.buffer.sample(self.batch_size)
        # state, next_state: (B,512)
        # action: (B,6), reward: (B,1), not_done: (B,1)

        # --- 2) Critic 更新 ---
        with torch.no_grad():
            # next action & log_prob from target actor
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            # target Q1, Q2
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + not_done * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = (F.mse_loss(current_q1, target_q) +
                       F.mse_loss(current_q2, target_q)) * 0.5

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 3) Actor 更新 (一部のステップで) ---
        if self.total_it % self.policy_update_freq == 0:
            action_pi, log_pi, _ = self.actor.sample(state)
            q1_pi, q2_pi = self.critic(state, action_pi)
            min_q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (self.alpha * log_pi - min_q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # --- 4) ターゲットネットワーク更新 (一部のステップで) ---
        if self.total_it % self.target_update_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, checkpoint_dir, step):
        """
        - モデルとオプティマイザを保存
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(checkpoint_dir, f"actor_{step}.pth"))
        torch.save(self.critic.state_dict(), os.path.join(checkpoint_dir, f"critic_{step}.pth"))

    def load(self, checkpoint_dir, step):
        """
        - actor, critic をロード
        """
        actor_path = os.path.join(checkpoint_dir, f"actor_{step}.pth")
        critic_path = os.path.join(checkpoint_dir, f"critic_{step}.pth")
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        # ターゲットも同期
        self.critic_target.load_state_dict(self.critic.state_dict())
