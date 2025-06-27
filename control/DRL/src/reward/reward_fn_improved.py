import numpy as np

class RewardImproved:
    """
    距離ベース報酬＋時間ペナルティ＋成功ボーナス＋安全ペナルティ方式
    """
    def __init__(self, distance_coeff, time_penalty, success_bonus, lambda_time, epsilon, safety_penalty):
        self.distance_coeff = distance_coeff
        self.time_penalty = time_penalty
        self.success_bonus = success_bonus
        self.lambda_time = lambda_time
        self.epsilon = epsilon
        self.safety_penalty = safety_penalty

    def compute(self, dist, t, safety):
        # 安全違反時
        if safety:
            return self.safety_penalty, True
        # 毎ステップの距離ペナルティ＋時間ペナルティ
        r = - self.distance_coeff * dist - self.time_penalty
        # 成功時ボーナス
        if dist <= self.epsilon:
            r += self.success_bonus * np.exp(-self.lambda_time * t)
            return r, True
        return r, False
