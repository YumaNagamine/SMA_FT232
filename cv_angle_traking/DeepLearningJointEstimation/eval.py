import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset import MultiVideoDataset
from models.network import SimpleJointNet
from utils.metrics import mse_loss


def evaluate():
    with open("./cv_angle_traking/DeepLearningJointEstimation/configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['train']['device'])

    # テストデータセット
    test_ds = MultiVideoDataset(
        root_dir=".",
        video_name=cfg['data']['video_name'],
        transform=None
    )
    test_loader = DataLoader(test_ds,
                             batch_size=cfg['data']['batch_size'],
                             shuffle=False,
                             num_workers=cfg['data']['num_workers'])

    # モデル & 重みロード
    model = SimpleJointNet(
        hidden_dim=cfg['model']['hidden_dim'],
        output_dim=cfg['model']['output_dim']
    ).to(device)
    model.load_state_dict(torch.load(cfg['eval']['checkpoint']))
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            total_loss += mse_loss(model(imgs), labels).item()
    print(f"Test MSE: {total_loss / len(test_loader):.4f}")


if __name__ == '__main__':
    evaluate()