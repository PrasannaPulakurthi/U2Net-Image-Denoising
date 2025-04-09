# loss.py
import torch.nn as nn
from pytorch_msssim import ssim

def ssim_loss(pred, target):
    return 1 - ssim(pred, target, data_range=1.0, size_average=True)

def combined_ssim_l1_loss(pred, target, alpha=0.5):
    l1 = nn.L1Loss()(pred, target)
    ssim_val = ssim_loss(pred, target)
    return alpha * ssim_val + (1 - alpha) * l1

def get_loss_function(loss_type: str):
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'ssim':
        return ssim_loss
    elif loss_type == 'ssim_l1':
        return combined_ssim_l1_loss
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}.")
