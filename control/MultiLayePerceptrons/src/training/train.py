import yaml
import torch
from torch.utils.data import DataLoader
from control.MultiLayePerceptrons.src.data.dataset import AngleDataset
from MultiLayePerceptrons.models.AngleNet import AngleNet
from utils.logger import setup_logger
from utils.metrics import mse_loss


def main():
    with open("./MultiLayerPerceptrons/configs/configs.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    csv_file_path = './sc01/multi_angle_tracking/FDP_training_data.csv'
    train_ds = AngleDataset(
        params
    )
    val_ds = AngleDataset(
        param
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=,
        shuffler =True,
        num_workers=
    )    
    val_loader = DataLoader(
        val_ds,
        batch_size=,
        shuffle=False,
        num_workers=
    )

    device = torch.device(cfg['train']['device'])
    model = AngleNet(
        input_dim = cfg['model']['input_dim'],
        output_dim = cfg['model']['output_dim'],
        hidden_layer_size = cfg['model']['hidden_layer_size']
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg['train']['lr']),
        weight_decay=cfg['train']['weight_decay']
    )
    logger = setup_logger()

    total_batches = len(train_loader)
    for epoch in range(cfg['train']['epochs']):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = mse_loss(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % cfg['train']['log_interval'] == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}/{total_batches}, Loss: {loss.item()}')
        # Validation
        model.eval()
        val_los = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_loss = mse_loss(preds, y)
                val_los += val_loss.item()
        val_loss /= len(val_loader)
        logger.info(f"Epoch: {epoch+1}/{cfg['train']['epochs']},-"
                     f"Train loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")
        
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pt")

if __name__ == "__main__":
    import os, time
    start = time.time()
    os.makedirs("checkpoints", exist_ok=True)
    main()
    print(f"time: {(time.time() - start)/60} mins")