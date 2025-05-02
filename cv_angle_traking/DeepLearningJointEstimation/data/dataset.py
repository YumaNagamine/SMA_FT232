import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class MultiVideoDataset(Dataset):
    """
    root_dir/
      └── {video_name}/
           ├── frames/
           └── annotations.csv  (videoname,frame,...)
    config.yaml の data.video_names に列挙した複数動画をまとめて読む
    """
    def __init__(self, root_dir: str, video_name: list, transform=None):
        self.root_dir   = root_dir
        self.video_names = video_name
        self.transform  = transform or T.Compose([T.ToTensor()])

        # 各動画ごとに CSV を読み込んで結合
        dfs = []
        for vn in video_name:
            csv_path = os.path.join(root_dir, vn, "annotations.csv")
            df = pd.read_csv(csv_path)
            # 必要あればフィルタリング（例: vn と一致する行のみ）
            df = df[df["videoname"] == vn]
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True).sort_values(["videoname","frame"])

        # ラベル列
        self.label_cols = [
            "fingertip_x","fingertip_y",
            "DIP_x","DIP_y",
            "PIP_x","PIP_y",
            "MCP_x","MCP_y",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vn = row["videoname"]
        frame_idx = int(row["frame"])
        img_path = os.path.join(self.root_dir, vn, "frames", f"{frame_idx}.jpg")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        labels = row[self.label_cols].to_numpy(dtype=np.float32)
        labels = torch.from_numpy(labels)
        return image, labels