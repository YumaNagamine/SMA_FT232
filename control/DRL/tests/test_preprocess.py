import numpy as np
from src.vision.preprocess import preprocess

def test_preprocess_shape_and_range():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = preprocess(img, width=64, height=64)
    assert out.shape == (3, 64, 64)
    # 全部ゼロ入力なので出力も全ゼロになるはず
    assert np.allclose(out, 0.0)
