import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from control.MultiLayePerceptrons.src.data.dataset import AngleDataset
from MultiLayePerceptrons.models.AngleNet import AngleNet
from utils.logger import setup_logger
from utils.metrics import mse_loss
import torch.optim as optim


def main():
    with open("./MultiLayerPerceptrons/configs/configs.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    csv_file_path = './sc01/multi_angle_tracking/FDP_training_data.csv'
    batch_size = 4
    epochs = 20
    learning_rate = 1e-3
    val_split = 0.2
    dataset = AngleDataset(csv_file_path, noise_std=1.0)
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_ds, val_ds = DataLoader(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AngleNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= n_train
    

#     train_ds = AngleDataset(
#         params
#     )
#     val_ds = AngleDataset(
#         param
#     )

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=,
#         shuffler =True,
#         num_workers=
#     )    
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=,
#         shuffle=False,
#         num_workers=
#     )

#     device = torch.device(cfg['train']['device'])
#     model = AngleNet(
#         input_dim = cfg['model']['input_dim'],
#         output_dim = cfg['model']['output_dim'],
#         hidden_layer_size = cfg['model']['hidden_layer_size']
#     ).to(device)
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=float(cfg['train']['lr']),
#         weight_decay=cfg['train']['weight_decay']
#     )
#     logger = setup_logger()

#     total_batches = len(train_loader)
#     for epoch in range(cfg['train']['epochs']):
#         model.train()
#         for batch_idx, (x, y) in enumerate(train_loader):
#             x, y = x.to(device), y.to(device)
#             preds = model(x)
#             loss = mse_loss(preds, y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if batch_idx % cfg['train']['log_interval'] == 0:
#                 logger.info(f'Epoch: {epoch}, Batch: {batch_idx}/{total_batches}, Loss: {loss.item()}')
#         # Validation
#         model.eval()
#         val_los = 0.0
#         with torch.no_grad():
#             for x, y in val_loader:
#                 x, y = x.to(device), y.to(device)
#                 preds = model(x)
#                 val_loss = mse_loss(preds, y)
#                 val_los += val_loss.item()
#         val_loss /= len(val_loader)
#         logger.info(f"Epoch: {epoch+1}/{cfg['train']['epochs']},-"
#                      f"Train loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")
        
#         torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pt")

if __name__ == "__main__":
    import os, time
    start = time.time()
    os.makedirs("checkpoints", exist_ok=True)
    main()
    print(f"time: {(time.time() - start)/60} mins")