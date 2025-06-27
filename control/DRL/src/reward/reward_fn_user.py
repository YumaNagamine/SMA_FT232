import numpy as np

class RewardUser:
    """
    ご提案のスパース報酬＋時間減衰＋安全ペナルティ方式
    """
    def __init__(self, threshold, lambda_time, success_reward, safety_penalty):
        self.threshold = threshold
        self.lambda_time = lambda_time
        self.success_reward = success_reward
        self.safety_penalty = safety_penalty

    def compute(self, dist, t, safety):
        # 安全違反時
        if safety:
            return self.safety_penalty, True
        # ターゲット到達時
        if dist <= self.threshold:
            reward = self.success_reward * np.exp(-self.lambda_time * t)
            return reward, True
        # 到達しない場合は報酬0
        return 0.0, False
