import yaml
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from models.network import SimpleJointNet

def predict(image_path: str, checkpoint: str):
    with open("./cv_angle_traking/DeepLearningJointEstimation/configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg['train']['device'])

    # モデルロード
    model = SimpleJointNet(
        hidden_dim=cfg['model']['hidden_dim'],
        output_dim=cfg['model']['output_dim']
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # 画像読み込み
    img = Image.open(image_path).convert("RGB")
    tensor = ToTensor()(img).unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        out = model(tensor)
    return out.squeeze(0).cpu().numpy()

def predict_frame(frame, checkpoint: str, model, device):
    tensor = ToTensor()(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
    return out.squeeze(0).cpu().numpy()

if __name__ == '__main__':
    import sys
    img_path = sys.argv[1]
    ckpt = sys.argv[2]
    preds = predict(img_path, ckpt)
    print("Predicted 8-dim vector:", preds)