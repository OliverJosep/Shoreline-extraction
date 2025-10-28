import torch

from models.base_model import BaseModel
from models.bilstm.architecture import BiLSTM_architecture
from torch import Tensor

from typing import Type
from torch.utils.data import Dataset
from src.models.data_management.bilstm_formes import BiLSTMFormesDataset

class BiLSTM(BaseModel):
    def __init__(self, num_classes: int = 1, experiment_name: str = "default_experiment", use_mlflow: bool = False, pretrained: bool = False, hidden_units: int = 45, oblique: bool = False):
        model: torch.nn.Module = BiLSTM_architecture(in_channels=3, out_channels=num_classes, hidden_units = hidden_units)        
        super().__init__(model=model, classes=num_classes, experiment_name=experiment_name, use_mlflow=use_mlflow)
        self.in_channels = 3
        self.H_THRESHOLD = 150
        self.oblique = oblique

    def train_step(self, input_image, target, loss_function, optimizer):
        optimizer.zero_grad()

        target = target.unsqueeze(dim=-1).squeeze(0).float()
        
        all_preds = []
        total_loss_value = 0.0

        total_loss_value, all_preds = self.process_recursive(input_image, target, loss_function, H_TOTAL=input_image.size(1))
        optimizer.step()

        return total_loss_value, all_preds

    def validate_step(self, input_image, target, loss_function):

        target = target.unsqueeze(dim=-1).squeeze(0).float()
        
        all_preds = []
        total_loss_value = 0.0

        total_loss_value, all_preds = self.process_recursive(input_image, target, loss_function, H_TOTAL=input_image.size(1), validation=True)

        return total_loss_value, all_preds

    def predict(self, image_path, formes_class: Type[Dataset] = BiLSTMFormesDataset, raw_output = False): # TODO: Check raw_output usage
        formes = formes_class(imgs_path=[image_path])
        input_image = formes[0] # Get the first element of the list, we only have one image
        input_image = input_image.unsqueeze(0)  # Add batch dimension

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            input_image = input_image.to(self.device)
            output = self.process_recursive(input_image, None, loss_function=None, H_TOTAL=input_image.size(1), validation=True)[1]

        return output

    def process_recursive(self, chunk_in, chunk_target, loss_function, H_TOTAL, validation: bool = False):
        current_h = chunk_in.size(1)

        if self.oblique and current_h > self.H_THRESHOLD:
            mid_point = current_h // 2
            
            chunk_in_1 = chunk_in[:, :mid_point, :]
            chunk_in_2 = chunk_in[:, mid_point:, :]
            
            if chunk_target is not None:
                chunk_target_1 = chunk_target[:mid_point]
                chunk_target_2 = chunk_target[mid_point:]
            else:
                chunk_target_1 = None
                chunk_target_2 = None

            loss_1, preds_1 = self.process_recursive(chunk_in_1, chunk_target_1, loss_function, H_TOTAL, validation)
            loss_2, preds_2 = self.process_recursive(chunk_in_2, chunk_target_2, loss_function, H_TOTAL, validation)

            return (loss_1 + loss_2), torch.cat((preds_1, preds_2), dim=0)
    
        chunk_reshaped = chunk_in.reshape(-1, chunk_in.shape[2], self.in_channels)

        output = self.forward_pass(chunk_reshaped)

        sigmoid_output = torch.sigmoid(output)
        if loss_function is not None:
            loss = loss_function(sigmoid_output, chunk_target)

        if self.classes > 1:
            preds = torch.argmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)  
            preds = (probs > 0.5).float()
        
        if loss_function is None:
            return 0.0, preds.detach()
        
        normalized_loss = loss * (current_h / H_TOTAL)
        if not validation:
            normalized_loss.backward()

        return normalized_loss.item(), preds.detach()
