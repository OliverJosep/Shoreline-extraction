import torch
from models.unet.architecture import UNet_architecture
from models.base_model import BaseModel

class UNet(BaseModel):
    def __init__(self, num_classes: int = 2):
        model = UNet_architecture(in_channels=3, out_channels=num_classes)
        super().__init__(model)

    def train_step(self, input_image, target, loss_function, optimizer): # TODO: Add types and descriptions
        # Forward pass
        output = self.model(input_image)

        # Unsqueeze target to match output shape
        target = target.squeeze(1).long()  # [batch_size, height, width]

        # Compute loss
        loss = loss_function(output, target)

        # Compute predictions # TODO: This is for the metrics
        preds = torch.argmax(output, dim=1)  # [batch_size, height, width]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def validate_step(self, input_image, target, loss_function): # TODO: Add types and descriptions
        # Forward pass
        output = self.model(input_image)

        # Unsqueeze target to match output shape
        target = target.squeeze(1).long()  # [batch_size, height, width]

        # Compute loss
        loss = loss_function(output, target)

        # Compute predictions
        preds = torch.argmax(output, dim=1)  # [batch_size, height, width]

        return loss.item()