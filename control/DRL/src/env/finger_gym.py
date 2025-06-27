import time
import cv2
import gym
import numpy as np
from pathlib import Path
import torch
from src.vision.preprocess import preprocess
from src.vision.pretrained_cnn import PretrainedCNN
from src.env.hw_interface import Interface
from stable_baselines3.common.env_checker import check_env

COOLDOWN_TIME = 5
class FingerGymEnv:
    metadata = {'render.modes': []}

    def __init__(self, cfg, device='gpu'):
        super().__init__()
        # --- ハードウェア／CNN初期化 ---
        # self.hw = Interface(cfg['hw']['port'], cfg['hw'].get('baudrate', 115200))
        self.hw = Interface()
        self.cnn = PretrainedCNN(output_dim=cfg['env']['cnn_output_dim'], pretrained=True).to(device)
        self.device = device

        # --- 目標画像リスト ---
        self.goals = list(Path(cfg['env']['goal_dir']).glob("*.png"))

        # --- gym.spaces 定義 ---
        dim = cfg['env']['cnn_output_dim']
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32)

        # --- 報酬・制御パラメータ ---
        self.reward_cfg = cfg['reward']
        self.max_steps = int(cfg['env']['control_frequency'] * cfg['env']['episode_duration'])

        # --- 熱モデル用パラメータ ---
        th = cfg['thermal']
        # self.heat_coeff = th['heat_coeff']          # duty比×時間 の加熱係数
        self.heat_coeff = 1          # duty比×時間 の加熱係数
        self.cool_coeff = th['cool_coeff']          # duty=0 時の冷却係数
        self.temp_threshold = th['temp_threshold']  # 安全閾値
        # 各ワイヤの「温度相当値」
        self.heats = np.zeros(6, dtype=np.float32)

        self.cap = cv2.VideoCapture(0)
        self.counter = 0
        self.reset()

    def reset(self):
        self.step_count = 0
        self.elapsed = 0.0
        # 温度リセット
        self.heats.fill(0.0)
        
        # ランダムに目標画像を選択・特徴量計算

        # goal_path = np.random.choice(self.goals)
        # img = self._capture_image_from_file(goal_path)
        if self.counter % 20 == 0:
            goal_path = np.random.choice(self.goals)
            img = self._capture_image_from_file(goal_path)
            self.counter = 0
        self.counter += 1
        x = torch.from_numpy(preprocess(
            img,
            self.cfg['camera']['resolution']['width'],
            self.cfg['camera']['resolution']['height']
        )).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.goal_feat = self.cnn(x).gpu().numpy().squeeze()
            # self.goal_feat = self.cnn(x).cpu().numpy().squeeze()
        time.sleep(COOLDOWN_TIME)
        # 初期観測はゼロ埋めでも可
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action: np.ndarray):
        # --- 1) 制御出力 ---
        # self.hw.send_duty(action.tolist())
        self.hw.output_levels = np.array([x for x in action.tolist()])
        self.hw.apply_DR()
        time.sleep(1.0 / self.cfg['env']['camera']['fps'])

        # --- 2) 観測取得・特徴量計算 ---
        frame = self._capture_image()
        x = torch.from_numpy(preprocess(
            frame,
            self.cfg['env']['camera']['resolution']['width'],
            self.cfg['env']['camera']['resolution']['height']
        )).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # feat = self.cnn(x).cpu().numpy().squeeze()
            feat = self.cnn(x).gpu().numpy().squeeze()
        obs = feat.astype(np.float32)

        # --- 3) 熱モデル更新 ---
        dt = 1.0 / self.cfg['env']['control_frequency']
        # action[i] > 0 のとき加熱、0 のとき冷却
        # self.heats += action * (self.heat_coeff * dt)
        self.heats += action * dt
        cooling = (action == 0.0).astype(np.float32) * (self.cool_coeff * dt)
        self.heats -= cooling
        # 負の温度は 0 にクリップ
        self.heats = np.clip(self.heats, 0.0, None)

        # --- 4) 報酬・終端判定 ---
        dist = np.linalg.norm(self.goal_feat - feat)
        # 安全違反チェック
        if np.any(self.heats > self.temp_threshold):
            reward = self.reward_cfg['safety_penalty']
            done = True
        else:
            # 距離ベース報酬等
            r = - self.reward_cfg['distance_coeff'] * dist - self.reward_cfg['time_penalty']
            if dist <= self.reward_cfg['epsilon']:
                r += self.reward_cfg['success_bonus'] * np.exp(-self.reward_cfg['lambda_time'] * self.elapsed)
                done = True
            else:
                done = False
            reward = r

        # --- 5) カウンタ更新 ---
        self.step_count += 1
        self.elapsed += dt
        if self.step_count >= self.max_steps:
            done = True

        return obs, reward, done, {'temperatures': self.heats.copy()}

    def _capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        # self.cap.release()
        return frame

    def _capture_image_from_file(self, path):
        return cv2.imread(str(path))
    
    def release_camera(self):
        self.cap.release()

if __name__ == "__main___":
    import yaml
    with open('configs/env.yaml') as f:
        cfg = yaml.safe_load(f)
    env = FingerGymEnv(cfg, device='gpu')
    check_env(env, warn=True)

