import torch
from models.unet.architecture import UNet_architecture
from models.cnn_model import CNNModel

class UNet(CNNModel):
    """
    UNet model for image segmentation tasks.

    This class extends the CNNModel class and provides a specific implementation
    for the UNet architecture. It initializes the model with the
    UNet architecture and sets the number of output classes.
    """
    def __init__(self, num_classes: int = 2, experiment_name:str = "default_experiment", use_mlflow: bool = False):
        model: torch.nn.Module = UNet_architecture(in_channels=3, out_channels=num_classes)
        super().__init__(model=model, num_classes=num_classes, experiment_name=experiment_name, use_mlflow=use_mlflow)