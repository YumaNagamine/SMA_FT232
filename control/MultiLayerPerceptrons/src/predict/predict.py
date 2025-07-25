import torch
from torch.utils.data import DataLoader
from control.MultiLayerPerceptrons.models.AngleNet import AngleNet
from control.MultiLayerPerceptrons.src.data.dataset import AngleDataset
# 1. モデル定義を同じ形で用意
model = AngleNet(input_dim=8, output_dim=6, hidden_layer_size=[32, 16])

# 2. 保存したパラメータをロード
checkpoint = torch.load("./control/MultiLayerPerceptrons/checkpoints/model_epoch_50.pth", map_location="cpu")
model.load_state_dict(checkpoint if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint
                      else checkpoint['model_state_dict'])

# 3. 推論モードに切り替え
model.eval()

# 4. 新しいデータ用の Dataset / DataLoader を用意（ノイズは不要なら noise_std=0 に）
predict_ds = AngleDataset("./control/MultiLayerPerceptrons/src/predict/FDP_training_data.csv", noise_std=0.0)
predict_loader = DataLoader(predict_ds, batch_size=1, shuffle=False)

# 5. 無勾配モードで予測
predictions = []
with torch.no_grad():
    for inputs, _ in predict_loader:   # ラベル y が不要なら "_" で受け取る
        outputs = model(inputs)        # (batch_size, 6) のテンソル
        predictions.append(outputs.numpy())

# 6. 結果をまとめて表示
import numpy as np
np.set_printoptions(suppress=True, precision=4)

predictions = np.vstack(predictions)  # shape = (N_samples, 6)
print("Predicted duty ratios:")
print(predictions)
