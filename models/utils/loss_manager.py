import torch.nn as nn

class LossManager:
    """Class to manage the loss functions used in the models."""

    @staticmethod
    def get_loss_function(loss_function_name: str) -> nn.Module:
        """
        Get the loss function by name.

        Parameters:
        loss_function_name (str): The name of the loss function.

        Raises:
        ValueError: If the loss function is unknown.

        Returns:
        nn.Module: The loss function.
        """

        loss_functions = {
            "CrossEntropy": nn.CrossEntropyLoss(),
            # TODO: Add more loss functions here
        }
        
        if loss_function_name not in loss_functions or loss_functions[loss_function_name] is None:
            raise ValueError(f"Unknown loss function: {loss_function_name}")
        
        return loss_functions[loss_function_name]