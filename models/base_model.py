import os
import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Type
from torch.utils.data import Dataset
from torch import Tensor
from src.models.data_management.data_loader import DataLoaderManager
from src.models.utils.loss_manager import LossManager
from src.models.utils.optimizer_manager import OptimizerManager
from src.models.metrics import Metrics
from src.models.utils.mlflow_manager import MLflowManager

class BaseModel(ABC):
    def __init__(self, model: nn.Module, classes: int = 0, experiment_name:str = "default_experiment", use_mlflow: bool = False) -> None:
        """
        Initializes the BaseModel object.

        Parameters:
        model (nn.Module): The model to use.
        classes (int): The number of classes in the dataset. Default is 0.
        experiment_name (str): The name of the experiment. Default: default_experiment
        use_mlflow (bool): If True, log metrics to MLflow. Default: False

        Returns:
        None
        """
        super(BaseModel, self).__init__()
        self.model = model
        self.classes = classes
        self.use_mlflow = use_mlflow

        if self.use_mlflow:
            self.mlflow_manager = MLflowManager(experiment_name=experiment_name)

    def save_model(self, path: str) -> None:
        """
        Save the model to the given path.

        Parameters:
        path (str): The path to save the model.

        Returns:
        None
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        """
        Load the model from the given path.

        Parameters:
        path (str): The path to load the model.

        Returns:
        None
        """
        self.model.load_state_dict(torch.load(path))

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

    def train(self, epochs: int = 100, loss_function_name: str = "CrossEntropy", optimizer_name: str = "Adam", learning_rate: float = 0.01, early_stopping: int = 25, run_name: str = None) -> None:
        """
        Trains the model using specified parameters.

        Parameters:
        epochs (int): The number of epochs (complete passes through the training dataset) to train the model. Default: 100
        loss_function_name (str): The name of the loss function to use during training. Default: "CrossEntropy"
        optimizer_name (str): The name of the optimizer to use for updating the model weights. Default: "Adam"
        learning_rate (float): The learning rate to control the step size during weight updates. Default: 0.01
        early_stopping (int): Number of epochs to wait for improvement before stopping the training early. Default: 25
        run_name (str): The name of the run. Default: None

        Raises:
        ValueError: If the data loaders have not been initialized.

        Returns:
        TODO: Metrics
        """

        # TODO: Generate a new folder for each run, with the timestamp as the name, and save the model, name of the model (like UNet), logs, and metrics there.

        if (self.train_loader is None) or (self.validation_loader is None):
            raise ValueError("Error: The data loaders have not been initialized. Please load the data before training the model.")
        
        # Start the MLflow run
        if self.use_mlflow:
            self.mlflow_manager.start_run(run_name)
            self.mlflow_manager.log_params({"epochs": epochs, "loss_function": loss_function_name, "optimizer": optimizer_name, "learning_rate": learning_rate, "early_stopping": early_stopping})


        try:
            # Load the loss function and optimizer
            loss_function = LossManager.get_loss_function(loss_function_name)
            optimizer = OptimizerManager.get_optimizer(optimizer_name, self.model.parameters(), learning_rate)

            # Move the model to the device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.model.to(self.device)

            # # Initialize the early stopping counter. TODO: Move this to a external function or class
            # early_stopping_counter = 0
            # best_validation_loss = float('inf')

            metrics_train = Metrics(phase='train', num_classes=self.classes, average='macro')
            metrics_validation = Metrics(phase='validation', num_classes=self.classes, average='macro')

            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")

                # Train
                self.model.train()
                for (input_image, target) in self.train_loader:
                    input_image = input_image.to(self.device)
                    target = target.to(self.device)

                    train_loss, preds = self.train_step(input_image, target, loss_function, optimizer)

                    metrics_train.update(preds, target, train_loss)

                metrics_train.compute(epoch)
                print(metrics_train.get_last_epoch_info())
                if self.use_mlflow:
                    self.mlflow_manager.log_metrics(metrics_train.get_last_epoch_info_dict(), epoch)

                # Validation    
                self.model.eval()

                with torch.no_grad():
                    for input_image, target in self.validation_loader:
                        input_image = input_image.to(self.device)
                        target = target.to(self.device)

                        val_loss, preds = self.validate_step(input_image, target, loss_function)
                        
                        metrics_validation.update(preds, target, val_loss)

                metrics_validation.compute(epoch)
                print(metrics_validation.get_last_epoch_info())
                if self.use_mlflow:
                    self.mlflow_manager.log_metrics(metrics_validation.get_last_epoch_info_dict(), epoch)

                # # Early stopping check. TODO: Implement this in a class.
                # if val_loss < best_validation_loss:
                #     best_validation_loss = val_loss
                #     early_stopping_counter = 0
                #     # TODO: Save the best model to a path, generate a unique name like "best_model" 
                # else:
                #     early_stopping_counter += 1
                #     if early_stopping_counter >= early_stopping:
                #         print("Early stopping triggered.")
                #         break
        except KeyboardInterrupt:
            print("Training interrupted by user. Closing MLflow run.")
            if self.use_mlflow:
                self.mlflow_manager.end_run(status="KILLED")
            raise

        except Exception as e:
            print(f"Training interrupted due to an error: {e}")
            if self.use_mlflow:
                self.mlflow_manager.end_run(status="FAILED")
            raise

        if self.use_mlflow:
            self.mlflow_manager.end_run()

    @abstractmethod
    def train_step(self, input_image, target, loss_function, optimizer, device) -> tuple[float, Tensor]:
        """Defines the forward and backward pass for a single training step."""
        pass

    @abstractmethod
    def validate_step(self, input_image, target, loss_function, device) ->  tuple[float, Tensor]:
        """Defines the forward pass for a single validation step."""
        pass