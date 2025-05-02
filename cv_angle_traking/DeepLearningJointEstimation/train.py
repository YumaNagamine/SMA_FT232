import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset import MultiVideoDataset
from models.network import SimpleJointNet
from utils.logger import setup_logger
from utils.metrics import mse_loss


def main():
    # 設定読み込み
    with open("./cv_angle_traking/DeepLearningJointEstimation/configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # データローダー
    train_ds = MultiVideoDataset(
        root_dir="./sc01",
        video_name=cfg['data']['video_name'],
        transform=None
    )
    val_ds = MultiVideoDataset(
        root_dir="./sc01",
        video_name=cfg['data']['video_name'],
        transform=None
    )
    train_loader = DataLoader(train_ds,
                              batch_size=cfg['data']['batch_size'],
                              shuffle=True,
                              num_workers=cfg['data']['num_workers'])
    val_loader = DataLoader(val_ds,
                            batch_size=cfg['data']['batch_size'],
                            shuffle=False,
                            num_workers=cfg['data']['num_workers'])

    # モデル準備
    device = torch.device(cfg['train']['device'])
    model = SimpleJointNet(
        hidden_dim=cfg['model']['hidden_dim'],
        output_dim=cfg['model']['output_dim']
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg['train']['lr']),
        weight_decay=cfg['train']['weight_decay']
    )

    logger = setup_logger()

    total_batches = len(train_loader)

    # 学習ループ
    for epoch in range(cfg['train']['epochs']):
        model.train()
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = mse_loss(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"\rEpoch {epoch+1}/{cfg['train']['epochs']} "
                  f"[Batch {batch_idx+1}/{total_batches}] "
                  f"Loss: {loss.item():.4f}", end="")
            
        print()


        # バリデーション
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_loss += mse_loss(model(imgs), labels).item()
        val_loss /= len(val_loader)

        logger.info(f"Epoch {epoch+1}/{cfg['train']['epochs']} - "
                    f"Train loss: {loss.item():.4f}, Val loss: {val_loss:.4f}")

        # チェックポイント保存
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pt")


if __name__ == '__main__':
    import os, time
    os.makedirs('checkpoints', exist_ok=True)
    start = time.time()
    main()
    print("time:", (time.time()-start)/3600, " hours")