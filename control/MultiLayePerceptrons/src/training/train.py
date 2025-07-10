import yaml
import torch
from torch.utils.data import dataloader
from MultiLayePerceptrons.models.AngleNet import AngleNet
from utils.logger import setup_logger
from utils.metrics import mse_loss


def main():
    with open("./MultiLayerPerceptrons/configs/configs.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train_ds = 