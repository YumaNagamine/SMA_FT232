# for loading data 
import  os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import pandas as pd
import torch

# angle_based
class AngleDataset(Dataset): # Load initial angles, final angles, and dutyratios from summary.csv
    def __init__(self, csv_path, noise_std = 1.0):
        """
        Args:
            csv_path (str): training data csv path
            noise_std (float): standard deviation of Gaussian noise to be added
        """
        self.df = pd.read_csv(csv_path)
        self.pairs = []
        for i in range(len(self.df) -1 ):
            if self.df.loc[i, 'state'] == 'initial' and self.df.loc[i+1, 'state']== 'final':
                angles_init = self.df.loc[i, ['angle0', 'angle1', 'angle2', 'angle_top']].values.astype(float)
                angles_final = self.df.loc[i+1, ['angle0', 'angle1', 'angle2', 'angle_top']].values.astype(float)
                x = torch.tensor(
                    list(angles_init) + list(angles_final), dtype=torch.float32
                )
                y = torch.tensor(
                    self.df.loc[i, ['duty_ratio0', 'duty_ratio1', 'duty_ratio2', 'duty_ratio3', 'duty_ratio4', 'duty_ratio5']].values.astype(float), 
                    dtype=torch.float32)
                self.pairs.append((x,y))
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        x, y = self.pairs[idx]
        noise = torch.randn_like(x) * self.noise_std
        x += noise
        return x, y

class AngleHistoryDataset(Dataset):
    def __init__(self):
        super().__init__()

# image_based
class ImageDataset(Dataset):
    pass
