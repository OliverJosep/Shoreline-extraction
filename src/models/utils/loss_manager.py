import torch.nn as nn
import torch
from src.models.utils.loss_functions.dice_loss import DiceLoss


class LossManager:
    """Class to manage the loss functions used in the models."""

    @staticmethod
    def get_loss_function(loss_function_name: str, weight: torch.Tensor = None) -> nn.Module:
        """
        Get the loss function by name.

        Parameters:
        loss_function_name (str): The name of the loss function.
        kwargs: Additional parameters for the loss function.

        Raises:
        ValueError: If the loss function is unknown.

        Returns:
        nn.Module: The loss function.
        """

        loss_functions = {
            "CrossEntropy": nn.CrossEntropyLoss(),
            "BCELoss": nn.BCELoss(),
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(pos_weight=weight),
            "DiceLoss": DiceLoss(weight=weight), # Binary Dice Loss with optional class weighting
            # Add more loss functions here as needed
        }
        
        if loss_function_name not in loss_functions or loss_functions[loss_function_name] is None:
            raise ValueError(f"Unknown loss function: {loss_function_name}")
        
        return loss_functions[loss_function_name]