import torch

from models.base_model import BaseModel
from models.bilstm.architecture import BiLSTM_architecture
from torch import Tensor

from typing import Type
from torch.utils.data import Dataset
from src.models.data_management.bilstm_formes import BiLSTMFormesDataset

class BiLSTM(BaseModel):
    def __init__(self, num_classes: int = 1, experiment_name: str = "default_experiment", use_mlflow: bool = False, pretrained: bool = False, hidden_units: int = 45):
        model: torch.nn.Module = BiLSTM_architecture(in_channels=3, out_channels=num_classes, hidden_units = hidden_units)        
        super().__init__(model=model, classes=num_classes, experiment_name=experiment_name, use_mlflow=use_mlflow)
        self.in_channels = 3

    def train_step(self, input_image, target, loss_function, optimizer): # TODO: Add types and descriptions
        
        # print(f"input_image shape: {input_image.shape}")
        input_image = input_image.reshape(-1, input_image.shape[2], self.in_channels)

        # Forward pass
        output = self.forward_pass(input_image)

        target = target.unsqueeze(dim=-1) 
        target = target.squeeze(0)
        target = target.float()

        # Compute loss
        sigmoid_output = torch.sigmoid(output)  # Apply sigmoid to output to get probabilities
        loss = loss_function(sigmoid_output, target)

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

        input_image = input_image.reshape(-1, input_image.shape[2], self.in_channels)

        # Forward pass
        output = self.forward_pass(input_image)

        target = target.unsqueeze(dim=-1) 
        target = target.squeeze(0)
        target = target.float()

        # Compute loss
        sigmoid_output = torch.sigmoid(output)  # Apply sigmoid to output to get probabilities
        loss = loss_function(sigmoid_output, target)

        # Compute predictions
        if self.classes > 1:
            preds = torch.argmax(output, dim=1)  # [batch_size, height, width]
        else:
            # Apply sigmoid to output to get probabilities
            probs = torch.sigmoid(output)  
            preds = (probs > 0.5).float()  

        return loss.item(), preds
    
    def predict(self, image_path, formes_class: Type[Dataset] = BiLSTMFormesDataset, raw_output = False): # TODO: Add types and descriptions
        formes = formes_class(imgs_path=[image_path])
        input_image = formes[0] # Get the first element of the list, we only have one image

        # Add the dimension of the batch
        input_image = input_image.unsqueeze(0)

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            input_image = input_image.to(self.device)
            input_image = input_image.reshape(-1, input_image.shape[2], self.in_channels)

            output = self.forward_pass(input_image)

        if raw_output:
            return output
        
        # Compute predictions
        if self.classes > 1:
            pred = torch.argmax(output, dim=1)
        else:
            # Apply sigmoid to output to get probabilities
            prob = torch.sigmoid(output)  
            pred = (prob > 0.5).float()
        
        return pred