import serial
import struct
import cv2
import numpy as np

class RobotInterface:
    """
    - シリアル通信経由で SMAワイヤ（あるいはモータ）に Duty比を送信
    - 同じくサイドカメラから OpenCV で画像を取得
    """

    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, cam_id=0):
        # シリアルポート初期化
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        # カメラ初期化
        self.cam = cv2.VideoCapture(cam_id)
        if not self.cam.isOpened():
            raise RuntimeError(f"Failed to open camera ID {cam_id}")

    def send_action(self, action_np):
        """
        action_np: np.ndarray shape=(6,), dtype=float32, 各成分 ∈ [0,1]
        → 通信プロトコルに応じてバイナリパケット化して送信
        例: 4bytes×6=24bytes の float32 6連パケット
        """
        # 必ず float32 としてパック (little-endian)
        packed = struct.pack("<6f", *(action_np.astype(np.float32).tolist()))
        # ヘッダなどが必要ならここで付与。今回は生データ送信と仮定
        self.ser.write(packed)

    def get_camera_image(self):
        """
        - OpenCV で BGR uint8 のフレームを1枚取得して返す
        - 呼び出すたびに新しいフレームを読んでいる想定
        """
        ret, frame = self.cam.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame  # BGR uint8

    def close(self):
        """
        - シリアルとカメラをクリーンに閉じる
        """
        if self.cam.isOpened():
            self.cam.release()
        if self.ser.is_open:
            self.ser.close()
