import os
import torch
import torch.nn as nn
import numpy as np
import tempfile

from abc import ABC, abstractmethod
from typing import Union, Type
from torch.utils.data import Dataset
from torch import Tensor
from src.models.data_management.data_loader import DataLoaderManager
from src.models.utils.loss_manager import LossManager
from src.models.utils.optimizer_manager import OptimizerManager
from src.models.metrics import Metrics
from src.models.utils.mlflow_manager import MLflowManager
from src.data_processing.patchify import Patchify
from src.data_processing.patch_reconstructor import PatchReconstructor
from datetime import datetime
from src.models.data_management.cnn_formes import CNNFormes

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

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
    
    def generate_folders_for_training(self, artifact_path: str, artifact_name: str = None) -> None:
        """
        Generates the folders for saving the logs, models, and metrics.

        Parameters:
        artifact_path (str): The path to save the artifacts.
        artifact_name (str): The name of the model. Default: None

        Returns:
        None
        """

        if not artifact_path:
            print("No artifact path provided. Skipping the creation of folders.")
            return

        if not artifact_name:
            artifact_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        full_path = os.path.join(artifact_path, artifact_name)
        print(f"Creating folders for the artifacts at {full_path}")
        if os.path.exists(full_path):
            # Throw an error if the path already exists
            raise FileExistsError(f"The path {artifact_path} already exists. Please provide a new path.")

        if not os.path.exists(artifact_path):
            os.makedirs(artifact_path, exist_ok=True)

        os.makedirs(full_path, exist_ok=True)
        os.makedirs(os.path.join(full_path, "logs"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "metrics"), exist_ok=True)

        self.artifact_path = full_path
        self.artifact_name = artifact_name

    def train(self, epochs: int = 100, loss_function_name: str = "CrossEntropy", optimizer_name: str = "Adam", learning_rate: float = 0.01, early_stopping: int = 25, artifact_path: str = None, run_name: str = None, run_description: str = None, weight: list = None) -> None:
        """
        Trains the model using specified parameters.

        Parameters:
        epochs (int): The number of epochs (complete passes through the training dataset) to train the model. Default: 100
        loss_function_name (str): The name of the loss function to use during training. Default: "CrossEntropy"
        optimizer_name (str): The name of the optimizer to use for updating the model weights. Default: "Adam"
        learning_rate (float): The learning rate to control the step size during weight updates. Default: 0.01
        early_stopping (int): Number of epochs to wait for improvement before stopping the training early. Default: 25
        artifact_path (str): The path to save the artifacts. Default: None
        run_name (str): The name of the run. Default: None
        run_description (str): The description of the run. Default: None
        weight (list): The positive class weights for the loss function. Default

        Raises:
        ValueError: If the data loaders have not been initialized.

        Returns:
        TODO: Metrics
        """

        if (self.train_loader is None) or (self.validation_loader is None):
            raise ValueError("Error: The data loaders have not been initialized. Please load the data before training the model.")
        
        # TODO: Generate a new folder for each run, with the timestamp as the name, and save the model, name of the model (like UNet), logs, and metrics there. For the logs, use the logging module.
        self.generate_folders_for_training(artifact_path=artifact_path, artifact_name=run_name)
        
        # Start the MLflow run
        if self.use_mlflow:
            self.mlflow_manager.start_run(self.artifact_name)
            self.mlflow_manager.log_params({"epochs": epochs, "loss_function": loss_function_name, "optimizer": optimizer_name, "learning_rate": learning_rate, "early_stopping": early_stopping, "weight": weight})
            if run_description:
                self.mlflow_manager.log_tag("mlflow.note.content", run_description)

        try:
            # Move the model to the device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.model.to(self.device)

            # Load the loss function and optimizer
            if weight:
                weight = torch.tensor(weight).to(self.device)
            loss_function = LossManager.get_loss_function(loss_function_name, weight=weight)
            optimizer = OptimizerManager.get_optimizer(optimizer_name, self.model.parameters(), learning_rate)

            # Initialize the early stopping counter. TODO: Move this to a external function or class
            early_stopping_counter = 0
            best_validation_loss = float('inf')

            metrics_train = Metrics(phase='train', num_classes=self.classes, average='macro', compute_loss=True)
            metrics_validation = Metrics(phase='validation', num_classes=self.classes, average='macro', compute_loss=True)

            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")

                # Train
                self.model.train()
                for (input_image, target) in self.train_loader:
                    input_image = input_image.to(self.device)
                    target = target.to(self.device)

                    train_loss, preds = self.train_step(input_image, target, loss_function, optimizer)

                    metrics_train.update_loss(train_loss)
                    metrics_train.update_metrics(preds, target)

                metrics_train.compute()
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
                        # metrics_validation.update(preds, target, val_loss)
                        metrics_validation.update_loss(val_loss)
                        metrics_validation.update_metrics(preds, target)

                metrics_validation.compute()
                print(metrics_validation.get_last_epoch_info())
                if self.use_mlflow:
                    self.mlflow_manager.log_metrics(metrics_validation.get_last_epoch_info_dict(), epoch)

                # Save the model at the end of each epoch
                if self.artifact_path:
                    self.save_model(os.path.join(self.artifact_path, "models", "last_epoch.pth"))

                # Early stopping check. TODO: Implement this in a class.
                val_loss = metrics_validation.get_last_loss()
                if val_loss < best_validation_loss:
                    early_stopping_counter = 0
                    print(f"Validation loss improved from {best_validation_loss:.6f} to {val_loss:.6f}. Saving the model. Early stopping counter: {early_stopping_counter}/{early_stopping}")
                    best_validation_loss = val_loss
                    # Save the model only if the validation loss has improved and the artifact path is provided
                    if self.artifact_path:
                        self.save_model(os.path.join(self.artifact_path, "models", "best_model.pth"))
                else:
                    print(f"Validation loss did not improve from {best_validation_loss:.46}, actual loss {val_loss:.6f}. Early stopping counter: {early_stopping_counter}/{early_stopping}")
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping:
                        print("Early stopping triggered.")
                        break

        except KeyboardInterrupt:
            print("Training interrupted by user. Closing MLflow run.")
            if self.use_mlflow:
                if self.artifact_path:
                    self.mlflow_manager.log_artifacts(self.artifact_path)
                self.mlflow_manager.end_run(status="KILLED")
            raise

        except Exception as e:
            print(f"Training interrupted due to an error: {e}")
            if self.use_mlflow:
                if self.artifact_path:
                    self.mlflow_manager.log_artifacts(self.artifact_path)
                self.mlflow_manager.end_run(status="FAILED")
            raise

        if self.use_mlflow:
            if self.artifact_path:
                self.mlflow_manager.log_artifacts(self.artifact_path)
            self.mlflow_manager.end_run()

    @abstractmethod
    def train_step(self, input_image, target, loss_function, optimizer, device) -> tuple[float, Tensor]:
        """Defines the forward and backward pass for a single training step."""
        pass

    @abstractmethod
    def validate_step(self, input_image, target, loss_function, device) ->  tuple[float, Tensor]:
        """Defines the forward pass for a single validation step."""
        pass

    @abstractmethod
    def predict(self, input_image: Tensor, raw_output = False) -> Tensor:
        """Predicts the output for a single input image."""
        pass

    def forward_pass(self, input_image: Tensor) -> Tensor:
        """
        Forward pass for the model.

        Parameters:
        input_image (Tensor): The input image.

        Returns:
        Tensor: The output of the model.
        """
        return self.model(input_image)

    def predict_patch(self, image_path: str, patch_size: int = 256, stride: int = 128, formes_class: Type[Dataset] = CNNFormes, combination: str = "avg") -> Tensor:
        """
        Predicts the output for an image by extracting patches and reconstructing the image.

        Parameters:
        image_path (str): The path to the image file.
        patch_size (int): The size of the patches. Default: 256
        stride (int): The stride for the patches. Default: 128
        formes_class (Type[Dataset]): The class of the Form. Default: CNNFormes
        combination (str): The method to combine the patches. Options: 'avg' or 'max'. Default: 'avg'

        Raises:
        ValueError: If the combination method is not 'avg' or 'max'.

        Returns:
        Tensor: The predicted output for the image.
        """

        # Create the patchify object
        patchify = Patchify(patch_size=patch_size, stride=stride)

        # Create a temporary directory to store the patches
        with tempfile.TemporaryDirectory() as temp_dir:
            result = patchify.extract_an_image_and_save_patches(image_path=image_path, output_image_dir=temp_dir)

            # list of patches of the tmp directory
            input_imgs = [f"{temp_dir}/{patch['image_path']}" for patch in result['patches']]

            # Predict the output for each patch
            output = torch.tensor([])
            for input_img in input_imgs:
                raw_output = self.predict(input_img, formes_class, raw_output = True)
                output = torch.cat((output, raw_output), dim=0)

        # Combine the patches
        reconstruded = PatchReconstructor.combine_patches(
            output = output, 
            n_classes = self.classes, 
            patches = result['patches'],
            padding = result['padding'],
            patch_size = result['options']['size'],
            method='max'
        )
        
        pred = torch.argmax(reconstruded, dim=0)

        return pred