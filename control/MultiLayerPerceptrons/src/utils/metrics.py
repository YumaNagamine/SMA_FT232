import torch.nn.functional as F

def mse_loss(preds, targets):
    return F.mse_loss(preds, targets)