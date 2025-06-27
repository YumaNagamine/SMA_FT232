import numpy as np
from src.reward.reward_fn_user import RewardUser

def test_threshold_behavior():
    reward = RewardUser(threshold=0.1, lambda_time=0.5, success_reward=1.0, safety_penalty=-1.0)
    # 閾値以下なら exp(-λt) を返す
    r, done = reward.compute(dist=0.05, t=2.0, safety=False)
    assert np.isclose(r, np.exp(-0.5 * 2.0))
    assert done is True

def test_safety_violation():
    reward = RewardUser(threshold=0.1, lambda_time=0.5, success_reward=1.0, safety_penalty=-1.0)
    r, done = reward.compute(dist=0.2, t=1.0, safety=True)
    assert r == -1.0
    assert done is True
