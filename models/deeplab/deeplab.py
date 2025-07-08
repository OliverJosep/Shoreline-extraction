import torch

from models.cnn_model import CNNModel
from torch import Tensor

class DeepLabV3(CNNModel):
    """
    DeepLabV3 model for image segmentation tasks.

    This class extends the CNNModel class and provides a specific implementation
    for the DeepLabV3 architecture. It initializes the model with the
    DeepLabV3 architecture and sets the number of output classes.
    """
    def __init__(self, num_classes: int = 2, experiment_name:str = "default_experiment", use_mlflow: bool = False, pretrained: bool = False):
        # model: torch.nn.Module = DuckNetArchitecture(in_channels=3, out_channels=num_classes)
        model: torch.nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=pretrained)

        # Change the output layer to match the number of classes
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        
        super().__init__(model=model, num_classes=num_classes, experiment_name=experiment_name, use_mlflow=use_mlflow)

    def forward_pass(self, input_image: Tensor) -> Tensor:
        return self.model(input_image)['out']