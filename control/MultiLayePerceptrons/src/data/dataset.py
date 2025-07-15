# for loading data 
import  os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import pandas as pd
import torch

# angle_based
class AngleDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        super().__init__()
        self.root_dir = root_dir
        

    def __init__(self, csv_path, angle_cols, duty_cols, transform=None):
        """
        Args:
            csv_path (str): summary.csv のパス
            angle_cols (list of str): ['angle0', 'angle1', 'angle2', 'angle_top']
            duty_cols (list of str): ['dutyratio0', …, 'dutyratio5']
            transform (callable, optional): 入力／出力に対する前処理
        """
        df = pd.read_csv(csv_path)
        # ファイル単位で initial と final をまとめる
        groups = df.groupby('file_name')
        samples = []
        for _, g in groups:
            # 両方揃っていることを仮定
            init = g[g.state == 'initial']
            final = g[g.state == 'final']
            if len(init) == 1 and len(final) == 1:
                # 角度部分をベクトル化
                x_init = init[angle_cols].values.flatten()
                x_final = final[angle_cols].values.flatten()
                x = torch.tensor(
                    list(x_init) + list(x_final), 
                    dtype=torch.float32
                )  # shape = (8,)
                
                # 出力は dutyratio（どちらの行を使っても同じはず）
                y = torch.tensor(
                    final[duty_cols].values.flatten(), 
                    dtype=torch.float32
                )  # shape = (6,)
                
                samples.append((x, y))
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.transform:
            x, y = self.transform(x, y)
        return x, y

# image_based
class ImageDataset(Dataset):
    pass

if __name__ == "__main__":
    is_angle_based = True
    if is_angle_based:
        pass
    else:
        pass

class AddGaussianNoise:
    def __init__(self, mean:float = 0.0, std: float = 2.0):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise, y
    