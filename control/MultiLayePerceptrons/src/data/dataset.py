# for loading data 
import  os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

# angle_based
class AngleDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        super().__init__()
        self.root_dir = root_dir
        

    

# image_based
class ImageDataset(Dataset):
    pass

if __name__ == "__main__":
    is_angle_based = True
    if is_angle_based:
        pass
    else:
        pass