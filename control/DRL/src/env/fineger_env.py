import numpy as np
import cv2
import time
from gym import spaces
from gym.utils import seeding
from src.utils.tracking_joint import @@@@

from .goal_manager import GoalManager
from src.utils.robot_interface import RobotInterface
from src.utils.image_utils import preprocess_image

class FingerEnv:
    """
    - 6本のワイヤにかける Duty比 (0~1) を action とする連続6次元アクション空間
    - 観測は (current_image, goal_image) の2枚の RGB 画像のみ
      → それぞれ preprocess_image() で (H×W×3 float32, [0,1]) に変換済み
    - 安全制約: Duty比>0 の時間累積 active_time が max_active_time を超えたら
      エピソード終了＋big_penalty ペナルティ
    """

    def __init__(self, config):
        # --- 設定読み込み ---
        self.image_height = config["image_height"]
        self.image_width  = config["image_width"]
        self.max_steps = config["max_steps_per_episode"]
        self.max_active_time = config["max_active_time"]
        self.step_time = config["step_time"]
        self.w_img = config["w_img"]
        self.w_time = config["w_time"]
        self.w_action = config["w_action"]
        self.big_penalty = config["big_penalty"]

        # GoalManager を初期化
        self.goal_manager = GoalManager(
            config.get("goal_schedule", None),
            config["goal_dir"],
            image_height=self.image_height,
            image_width=self.image_width
        )

        # 実機ロボットインターフェース (Python から直接制御)
        self.robot = RobotInterface(
            port=config["serial_port"], 
            baudrate=config["baudrate"]
        )

        # --- アクション空間: 6次元連続 [0,1] ---
        action_low  = np.zeros(6, dtype=np.float32)
        action_high = np.ones(6, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # --- 観測空間: 画像2枚, それぞれ [0,1] float32、shape=(H,W,3) ---
        #   2枚まとめて (2,H,W,3) にして Box(0,1) で定義
        obs_img_shape = (2, self.image_height, self.image_width, 3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_img_shape, dtype=np.float32
        )

        # --- 内部変数の初期化 ---
        self.current_step = 0
        self.active_time = 0.0        # Duty比>0 の累積時間
        self.current_goal_img = None  # np.ndarray (H,W,3) float32
        self.current_img = None       # 最新の環境画像
        self.seed()                   # 乱数シード
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        1エピソード開始時に呼ばれる
        - active_time, current_step をリセット
        - start step=0 なので goal_manager に is_reset=True でゴールを取得
        - カメラ画像を初期取得
        """
        self.current_step = 0
        self.active_time = 0.0
        # GoalManager から最初のゴール画像を取得
        self.current_goal_img = self.goal_manager.get_goal(0, is_reset=True)
        if self.current_goal_img is None:
            raise RuntimeError("Goal image cannot be None at reset()")

        # 実機を初期状態に戻す (必要なら実装)
        # 例: self.robot.reset_to_home()

        # カメラ画像を取得して前処理
        raw_img = self.robot.get_camera_image()  # BGR uint8
        self.current_img = preprocess_image(raw_img, self.image_height, self.image_width)

        # Stack して観測を返す (2,H,W,3)
        obs = np.stack([self.current_img, self.current_goal_img], axis=0)
        return obs

    def step(self, action):
        """
        action: np.ndarray shape=(6,), dtype=float32, in [0,1]
        Returns: (obs_next, reward, done, info)
        """
        # --- 1) アクション送信 ---
        # action は Duty比 → robot_interface.send_action() にそのまま渡す
        self.robot.send_action(action)

        # --- 2) 安全制約: Duty比>0 の時間累積 ---
        if np.any(action > 0.0):
            self.active_time += self.step_time

        # --- 3) 少し待って実機を動かす ---
        time.sleep(self.step_time)

        # --- 4) 次の観測画像取得 & 前処理 ---
        raw_img = self.robot.get_camera_image()
        next_img = preprocess_image(raw_img, self.image_height, self.image_width)

        # --- 5) current_step インクリメント & goal 更新 ---
        self.current_step += 1
        new_goal = self.goal_manager.get_goal(self.current_step, is_reset=False)
        if new_goal is not None:
            self.current_goal_img = new_goal

        # --- 6) 報酬計算 ---
        reward, done_safety = self._compute_reward_and_check_done_img(
            next_img, action
        )

        # エピソード終了判定
        done = False
        if self.current_step >= self.max_steps:
            done = True
        if done_safety:
            done = True

        # --- 7) 次の観測を返す ---
        obs_next = np.stack([next_img, self.current_goal_img], axis=0)
        info = {
            "step": self.current_step,
            "active_time": self.active_time
        }
        self.current_img = next_img
        return obs_next, reward, done, info

    def _compute_reward_and_check_done_img(self, cur_img, action, jointtracking=False):
        """
        - cur_img (H,W,3 float32), goal_img も (H,W,3 float32)
        - action: (6,) float32
        - active_time が max_active_time を超えたら done_safety=True で大ペナルティ
        Returns: (reward: float, done_safety: bool)
        """
        if not jointtracking:
            # 1) 画像誤差 (L2)
            diff = cur_img - self.current_goal_img
            img_dist = np.linalg.norm(diff)  # スカラー

            # 正規化: max_img_dist = sqrt(H×W×3)
            max_img_dist = np.sqrt(self.image_height * self.image_width * 3)
            norm_img = img_dist / max_img_dist  # ∈ [0, 1]

        elif jointtracking:
            # 1) joint pos extraction
            pass

        # 2) アクションノルム (L2) 正規化
        action_norm = np.linalg.norm(action)  # ∈ [0, sqrt(6)]
        norm_action = action_norm / np.sqrt(6)  # ∈ [0,1]

        # 3) 安全制約チェック
        done_safety = False
        if self.active_time >= self.max_active_time:
            done_safety = True

        # 4) 報酬を合成
        reward = - self.w_img * norm_img \
                 - self.w_action * norm_action

        # 安全違反時は大ペナルティを追加
        if done_safety:
            reward -= self.big_penalty

        # 5) active_time をペナルティ項に含めたい場合（連続項）
        #    reward -= self.w_time * (self.active_time / self.max_active_time)
        #    ただし、上で done_safety のときは big_penalty を使うので、
        #    連続的な penalize は optional。
        reward -= self.w_time * (self.active_time / self.max_active_time)

        return float(reward), done_safety


    def render(self, mode="human"):
        """
        - 実機環境なので簡易可視化：現在と目標画像を横並びで表示
        """
        img_cur = (self.current_img * 255).astype(np.uint8)
        img_goal = (self.current_goal_img * 255).astype(np.uint8)
        concat = np.concatenate([img_cur, img_goal], axis=1)  # (H,2W,3)
        cv2.imshow("Current | Goal", cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def close(self):
        """
        - 実機リソース解放 (必要なら)
        """
        self.robot.close()
        cv2.destroyAllWindows()
