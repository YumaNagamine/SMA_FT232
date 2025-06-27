import numpy as np
from src.agent.replay_buffer import ReplayBuffer

def test_push_and_len():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.push(i, i*0.1, i*0.2, i*0.3, False)
    # capacity=3 以上は上書きされるので長さは3
    assert len(buf) == 3

def test_sample_shapes():
    buf = ReplayBuffer(capacity=10)
    for i in range(10):
        buf.push(np.array([i]), np.array([i]), i, np.array([i+1]), False)
    s, a, r, ns, d = buf.sample(batch_size=4)
    assert s.shape == (4, 1)
    assert a.shape == (4, 1)
    assert r.shape == (4,)
    assert ns.shape == (4, 1)
    assert d.shape == (4,)
