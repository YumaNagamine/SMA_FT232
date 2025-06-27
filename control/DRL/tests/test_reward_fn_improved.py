import numpy as np
from src.reward.reward_fn_improved import RewardImproved

def test_distance_penalty():
    reward = RewardImproved(distance_coeff=1.0, time_penalty=0.01,
                             success_bonus=1.0, lambda_time=0.5,
                             epsilon=0.1, safety_penalty=-1.0)
    # 距離ベースのペナルティ
    r, done = reward.compute(dist=0.2, t=0.5, safety=False)
    assert np.isclose(r, -0.2 - 0.01)
    assert done is False

def test_success_bonus():
    reward = RewardImproved(distance_coeff=1.0, time_penalty=0.01,
                             success_bonus=1.0, lambda_time=0.5,
                             epsilon=0.1, safety_penalty=-1.0)
    # 成功時には bonus+exp 減衰
    r, done = reward.compute(dist=0.05, t=1.0, safety=False)
    assert r > 0
    assert done is True
