import torch
from models.attention_unet.architecture import Attention_UNet_architecture
from models.cnn_model import CNNFormes

class Attention_UNet(CNNFormes):
    """
    TODO: Add description

    id we want to modify the prediction method or the train step, we can do it here.
    """
    def __init__(self, num_classes: int = 2, experiment_name:str = "default_experiment", use_mlflow: bool = False):
        model: torch.nn.Module = Attention_UNet_architecture(in_channels=3, out_channels=num_classes)
        super().__init__(model = model, classes=num_classes, experiment_name=experiment_name, use_mlflow=use_mlflow)