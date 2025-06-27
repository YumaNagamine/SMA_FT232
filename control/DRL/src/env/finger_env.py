import time
import numpy as np
from pathlib import Path
import torch
from src.vision.preprocess import preprocess
from src.vision.pretrained_cnn import PretrainedCNN

class FingerEnv:
    """
    Gym ライクな実機環境ラッパー (ImageNet-pretrained CNN版)
    """
    def __init__(self, hw_interface, goal_dir, cnn_output_dim, config, device='cpu'):
        self.hw = hw_interface
        self.goals = list(Path(goal_dir).glob("*.png"))
        self.config = config
        # pretrained CNN を読み込み
        self.cnn = PretrainedCNN(output_dim=cnn_output_dim, pretrained=True).to(device)
        self.device = device
        self.step_count = 0
        self.elapsed = 0.0

    def _load_image(self, path):
        import cv2
        return cv2.imread(str(path))

    def reset(self):
        self.step_count = 0
        self.elapsed = 0.0
        # ランダムに目標画像を選択
        goal_path = np.random.choice(self.goals)
        img = self._load_image(goal_path)
        x = torch.from_numpy(
            preprocess(img,
                       self.config['camera']['resolution']['width'],
                       self.config['camera']['resolution']['height'])
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.goal_feat = self.cnn(x).cpu().numpy().squeeze()
        # 最初の観測を返す
        return self._get_obs()

    def _get_obs(self):
        return self.last_feat

    def step(self, action):
        # Duty比を送信
        self.hw.send_duty(action)
        time.sleep(1.0 / self.config['camera']['fps'])
        # カメラ取得→前処理→CNN
        img = self._capture_image()
        x = torch.from_numpy(
            preprocess(img,
                       self.config['camera']['resolution']['width'],
                       self.config['camera']['resolution']['height'])
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.cnn(x).cpu().numpy().squeeze()
        self.last_feat = feat
        # 距離計算
        dist = np.linalg.norm(self.goal_feat - feat)
        # ステップ・時間更新
        self.step_count += 1
        self.elapsed += 1.0 / self.config['control_frequency']
        return self.last_feat, dist, self.step_count, self.elapsed

    def _capture_image(self):
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        return frame
