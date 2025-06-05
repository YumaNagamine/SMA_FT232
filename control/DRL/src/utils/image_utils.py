import cv2
import numpy as np

def preprocess_image(bgr_image, target_h, target_w):
    """
    - bgr_image: OpenCV が返す BGR uint8 画像 (H_orig×W_orig×3)
    - target_h, target_w: リサイズ先 (整数)
    Returns: RGB float32 [0,1], shape=(target_h, target_w, 3)
    """
    # 1) BGR→RGB
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # 2) リサイズ
    resized = cv2.resize(rgb, (target_w, target_h))
    # 3) 正規化
    out = resized.astype(np.float32) / 255.0
    return out  # (target_h, target_w, 3), float32, [0,1]
