import os
import cv2
import numpy as np

class GoalManager:
    """
    - goal_schedule_cfg: [
    #     { "step": 0, "image": "pose_A.png" },
    #     { "step": 200, "image": "pose_B.png" },
    #     ...
    #   ]
    - goal_dir: "data/goal_images"
    - したがって、各 step に対応する画像パスを組み立てて扱う
    - get_goal(step) を呼ぶと、もしそのステップが schedule に登録されていれば
      画像を読み込んで [0,1] float32 にリサイズして返す。登録なければ None 。
    """

    def __init__(self, goal_schedule_cfg, goal_dir, image_height=64, image_width=64):
        self.goal_dir = goal_dir
        self.image_height = image_height
        self.image_width = image_width

        # schedule_cfg が None なら、1エピソード開始時にランダム1枚読み込む仕組みに切り替え
        if goal_schedule_cfg is None:
            self.schedule = None
        else:
            self.schedule = []
            for item in goal_schedule_cfg:
                step = item["step"]
                img_name = item["image"]
                img_path = os.path.join(self.goal_dir, img_name)
                self.schedule.append((step, img_path))
            self.schedule.sort(key=lambda x: x[0])
        self.current_index = 0
        self.current_goal_img = None  # 最後に返した goal image をキャッシュ
        self.random_list = None       # ランダムロード用リスト
        if self.schedule is None and os.path.isdir(self.goal_dir):
            # goal_dir 以下の画像をすべてリストアップ
            all_files = os.listdir(self.goal_dir)
            self.random_list = [f for f in all_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    def get_goal(self, step, is_reset=False):
        """
        step: 現在ステップ数（int）
        is_reset: reset() から呼ばれたとき True, 以降 step 毎は False
        戻り値: 画像 (H×W×3 float32) または None
        """
        if self.schedule is None:
            # 1エピソード開始時のみランダム1枚選択して返す
            if is_reset:
                if not self.random_list:
                    raise FileNotFoundError(f"No images in {self.goal_dir}")
                selected = np.random.choice(self.random_list)
                img_path = os.path.join(self.goal_dir, selected)
                img = cv2.imread(img_path)
                if img is None:
                    raise FileNotFoundError(f"Failed to load image: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.image_width, self.image_height)).astype(np.float32) / 255.0
                self.current_goal_img = img
                return img
            else:
                # エピソード途中では新しいゴールを返さない
                return None
        else:
            # schedule ベース
            if self.current_index < len(self.schedule):
                sched_step, img_path = self.schedule[self.current_index]
                if step == sched_step:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise FileNotFoundError(f"Failed to load image: {img_path}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.image_width, self.image_height)).astype(np.float32) / 255.0
                    self.current_index += 1
                    self.current_goal_img = img
                    return img
            return None
