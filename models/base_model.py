import os
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Union, Type
from torch.utils.data import Dataset
from models.data_management.data_loader import DataLoaderManager

class BaseModel(ABC, nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def load_data(self, data_source: Union[str, dict], formes_class: Type[Dataset], batch_size: int = 16) -> None:
        """
        The method to load the data from the given path.

        Parameters:
        data_source (Union[str, dict]): The path to the data or a dictionary containing the data.
        formes_class (Type[Dataset]): The class of the Formes dataset.
        batch_size (int, optional): The batch size to use. Default is 16.

        Raises:
        ValueError: If the data_source is not a string or a dictionary.

        Returns:
        None
        """

        self.data = DataLoaderManager.load_data(data_source)

        self.train_formes = DataLoaderManager.generate_formes(self.data["train"]["images"], self.data["train"]["masks"], formes_class)
        self.train_loader = DataLoaderManager.generate_data_loaders(self.train_formes, batch_size, shuffle=True)

        self.validation_formes = DataLoaderManager.generate_formes(self.data["validation"]["images"], self.data["validation"]["masks"], formes_class)
        self.validation_loader = DataLoaderManager.generate_data_loaders(self.validation_formes, batch_size, shuffle=False)

        if "test" in self.data:
            self.test_formes = DataLoaderManager.generate_formes(self.data["test"]["images"], self.data["test"]["masks"], formes_class)
            self.test_loader = DataLoaderManager.generate_data_loaders(self.test_formes, batch_size, shuffle=False)

    def train(self):
        # TODO
        pass