import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, eps=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2 * intersection + eps) / (union + eps)
        return 1 - dice

bce = nn.BCEWithLogitsLoss()
dice = DiceLoss()

def combined_loss(pred, mask):
    return 0.5*bce(pred, mask) + 0.5*dice(pred, mask)
