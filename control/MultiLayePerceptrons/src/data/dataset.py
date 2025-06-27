# for loading data 
import  os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

# angle_based
class AngleDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        super().__init__()

    

# image_based
class ImageDataset(Dataset):