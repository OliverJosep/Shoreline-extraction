from torcheval.metrics import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, 
    MulticlassRecall, MulticlassConfusionMatrix, BinaryAccuracy, 
    BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryConfusionMatrix
)
import torch
import numpy as np
import csv
import json

class Metrics():
    def __init__(self, phase: str, num_classes: int = 1, average: str = 'macro', compute_loss: bool = False, use_margin: bool = False, margin: int = 10, save_path: str = None):
        self.phase = phase
        self.num_classes = num_classes
        self.average = average

        self.save_path = save_path

        self.use_margin = use_margin
        self.margin = margin

        self.compute_loss = compute_loss

        metrics_classes = {
            f'{phase}_accuracy': (MulticlassAccuracy, BinaryAccuracy),
            f'{phase}_f1_score': (MulticlassF1Score, BinaryF1Score),
            f'{phase}_precision': (MulticlassPrecision, BinaryPrecision),
            f'{phase}_recall': (MulticlassRecall, BinaryRecall),
            f'{phase}_confusion_matrix': (MulticlassConfusionMatrix, BinaryConfusionMatrix)
        }

        is_multiclass = num_classes > 1
        index_class = 0 if is_multiclass else 1

        self.metrics = {}
        for name, metric_class_pair in metrics_classes.items():
            metric_class = metric_class_pair[index_class]

            posible_kwargs = {'num_classes': num_classes, 'average': average, "normalize": "true"}
            kwargs = {key: value for key, value in posible_kwargs.items() if key in metric_class.__init__.__code__.co_varnames}

            self.metrics[name] = metric_class(**kwargs)

        self.metrics_history = {name: [] for name in metrics_classes.keys()}

        # Add loss metric if needed
        if compute_loss:
            self.actual_loss = []
            self.loss_history = []

    def update_loss(self, loss: float):
        if self.compute_loss:
            self.actual_loss.append(loss)

    def get_last_loss(self):
        if not self.compute_loss:
            raise ValueError('Loss is not being computed')
        return self.loss_history[-1]

    def update_metrics(self, prediction, target):
        if self.use_margin:
            prediction, target = self.apply_roi(prediction, target, shoreline_value=3)

        prediction = prediction.flatten().to(torch.int64)
        target = target.flatten().to(torch.int64)

        for key, metric in self.metrics.items():
            metric.update(prediction.flatten(), target.flatten())

    def compute(self):
        for key, metric in self.metrics.items():
            value = metric.compute() # .item()
            self.metrics_history[key].append(value)
            metric.reset()
        
        if self.compute_loss:
            self.loss_history.append(np.mean(self.actual_loss))
            self.actual_loss = []
    
    def get_last_epoch_info(self):
        text = f'{self.phase} metrics: \n'
        if self.compute_loss:
            text += f"\t{self.phase}_loss: {self.get_last_loss()}\n"
        for key, _ in self.metrics.items():
            text += f'\t{key}: {self.metrics_history[key][-1]}' + '\n'
        return text
    
    def get_last_epoch_info_dict(self):
        metrics_dict = {
            name: self.metrics_history[name][-1]  # Agafa l'últim valor guardat
            for name in self.metrics_history
            if "confusion_matrix" not in name and self.metrics_history[name]  # Exclou matriu i evita llistes buides
        }
    
        if self.compute_loss and self.loss_history:
            metrics_dict[f"{self.phase}_loss"] = self.loss_history[-1]  # Últim valor de la pèrdua

        return metrics_dict
    
    def apply_roi(self, prediction, target, shoreline_value: int = 3):
        coastline_indices = np.where(target == shoreline_value)

        roi_mask = np.zeros_like(target)

        for row, col in zip(*coastline_indices):
            start_col = max(col - self.margin, 0)
            end_col = min(col + self.margin, target.shape[1])
        
            roi_mask[row, start_col:end_col] = 1

        prediction_roi = prediction[roi_mask == 1]
        target_roi = target[roi_mask == 1]

        return prediction_roi, target_roi
    
    def save_metrics_to_json(self):
        # Save the entire history of metrics to JSON
        if self.save_path is None:
            return # Do not save if no path is provided
        
        path = f"{self.save_path}/metrics/{self.phase}.json"
    
        class_names = [f'class_{i}' for i in range(self.num_classes)]

        # Function to convert confusion matrix to a dictionary with class names as keys
        def convert_confusion_matrix_to_dict(confusion_matrix):
            # Convert each confusion matrix row into a dictionary with class names as keys
            return {class_names[i]: row.tolist() for i, row in enumerate(confusion_matrix)}

        # Filter and process metrics history: 
        # If the key is related to confusion matrix, apply the conversion function
        metrics_history_filtered = {
            key: [item.item() if isinstance(item, torch.Tensor) else item for item in value]
            if "confusion_matrix" not in key else [convert_confusion_matrix_to_dict(cm) for cm in value]
            for key, value in self.metrics_history.items()
        }
    
        # Create a dictionary with the filtered metric history
        metrics_dict = {
            'metrics_history': metrics_history_filtered,
            'loss_history': [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in self.loss_history] if self.compute_loss else None
        }

        with open(path, 'w') as json_file:
            json.dump(metrics_dict, json_file, indent=4)