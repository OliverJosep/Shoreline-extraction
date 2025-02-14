import torch
from models.unet.architecture import UNet_architecture
from models.base_model import BaseModel
import numpy as np

class UNet(BaseModel):
    def __init__(self, num_classes: int = 2, experiment_name:str = "default_experiment", use_mlflow: bool = False):
        model = UNet_architecture(in_channels=3, out_channels=num_classes)
        super().__init__(model, classes=num_classes, experiment_name=experiment_name, use_mlflow=use_mlflow)

    def train_step(self, input_image, target, loss_function, optimizer): # TODO: Add types and descriptions
        # Forward pass
        output = self.model(input_image)

        # Unsqueeze target to match output shape (only if we have more than one class)
        if self.classes > 1:
            target = target.squeeze(1).long()  # [batch_size, height, width]
        else:
            target = target.unsqueeze(1).float()  # [batch_size, 1, height, width]

        # Compute loss
        loss = loss_function(output, target)

        # Compute predictions # TODO: This is for the metrics
        if self.classes > 1:
            preds = torch.argmax(output, dim=1)  # [batch_size, height, width]
        else:
            # Apply sigmoid to output to get probabilities
            probs = torch.sigmoid(output)  
            preds = (probs > 0.5).float()  

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), preds

    def validate_step(self, input_image, target, loss_function): # TODO: Add types and descriptions
        # Forward pass
        output = self.model(input_image)

        # Unsqueeze target to match output shape
        target = target.squeeze(1).long()  # [batch_size, height, width]

        # Compute loss
        loss = loss_function(output, target)

        # Compute predictions
        preds = torch.argmax(output, dim=1)  # [batch_size, height, width]

        return loss.item(), preds