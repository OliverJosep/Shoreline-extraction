import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, weight: torch.Tensor = None, smooth: float = 1.0):
        """
        Binary Dice Loss with optional class weighting.

        Parameters
        weight (torch.Tensor): Weight factor to balance positive negative classes. Default is None.
        smooth (float): Smoothing factor to avoid division by zero. Default is 1.0.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds) # Transform it to probabilities
        preds = preds.view(-1) # Flatten
        targets = targets.view(-1) # Flatten

        intersection = (preds * targets).sum() # Intersecction between predictions and targets

        # Dice coefficient -> 2 * intersection / (preds + targets + smooth(to avoid division by zero))
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice  # Transform it to loss

        # TODO: Check if this is correct
        if self.weight is not None:
            # Weight factor to balance positive negative classes
            weight_factor = self.weight * targets + (1 - self.weight) * (1 - targets)
            dice_loss = (dice_loss * weight_factor).mean()
        
        return dice_loss