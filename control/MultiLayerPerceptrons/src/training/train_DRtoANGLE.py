import yaml
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from control.MultiLayerPerceptrons.src.data.dataset import DutyratioToAngleDataset
from control.MultiLayerPerceptrons.models.AngleNet import DutyRatioNet
from control.MultiLayerPerceptrons.src.utils.logger import setup_logger
from control.MultiLayerPerceptrons.src.utils.metrics import mse_loss
import torch.optim as optim

# training input; duty ratios , output; 
def main():
    with open("./control/MultiLayerPerceptrons/configs/configs.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    csv_file_path = './sc01/multi_angle_tracking/FDP_training_data.csv'
    batch_size = 4
    epochs = 50
    learning_rate = 1e-3
    val_split = 0.2
    dataset = DutyratioToAngleDataset(csv_file_path, noise_std=0.0)
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DutyRatioNet(input_dim=cfg['model2']['input_dim'], 
                     output_dim=cfg['model2']['output_dim'], 
                     hidden_layer_size=[128, 64, 32])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= n_val

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, f'./control/MultiLayerPerceptrons/checkpoints_DRtoAngle/model_epoch_{epoch}.pth'
        )

        print(f"Epoch {epoch:2d}/{epochs} "
              f"Train loss: {train_loss:.4f}, "
              f"Val loss: {val_loss:.4f}")
    print('training completed!')


if __name__ == "__main__":
    import os, time
    start = time.time()
    os.makedirs("./control/MultiLayerPerceptrons/checkpoints_DRtoAngle", exist_ok=True)
    main()
    print(f"time: {(time.time() - start)/60} mins")