from typing import Iterable
import torch.nn as nn
import torch.optim as optim

class OptimizerManager:
    """Class to manage optimizers used in the models."""

    @staticmethod
    def get_optimizer(optimizer_name: str, model_params: Iterable[nn.Parameter], learning_rate: float = 0.01) -> optim.Optimizer:
        """
        Get the optimizer by name.

        Parameters:
        optimizer_name (str): The name of the optimizer.
        model_params (Iterable[nn.Parameter]): The parameters of the model.
        learning_rate (float, optional): The learning rate. Default is 0.01.

        Raises:
        ValueError: If the optimizer is unknown.

        Returns:
        optim.Optimizer: The optimizer instance.
        """

        # TODO: Fix the type of model_params
        optimizers = {
            "adam": optim.Adam(params=model_params, lr=learning_rate),
            # "sgd": optim.SGD(params=list(model_params), lr=learning_rate, momentum=0.9)
            # TODO: Add more optimizers if needed
        }

        # Transform the optimizer name to lowercase
        optimizer_name = optimizer_name.lower()

        if optimizer_name not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizers[optimizer_name]