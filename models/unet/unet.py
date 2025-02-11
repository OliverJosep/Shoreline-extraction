import torch
from models.unet.architecture import UNet_architecutre
from models.base_model import BaseModel

class UNetModel(BaseModel):
    def __init__(self, num_classes: int = 2):
        super(UNetModel, self).__init__()
        self.model = UNet_architecutre(in_channels=3, out_channels=num_classes)

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