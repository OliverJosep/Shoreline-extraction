from torcheval.metrics import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, 
    MulticlassRecall, MulticlassConfusionMatrix, BinaryAccuracy, 
    BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryConfusionMatrix
)
import torch
import numpy as np

class Metrics():
    def __init__(self, phase: str, num_classes: int = 1, average: str = 'macro', compute_loss: bool = False, use_margin: bool = False, margin: int = 10):
        self.phase = phase
        self.num_classes = num_classes
        self.average = average

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

            posible_kwargs = {'num_classes': num_classes, 'average': average}
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
            prediction, target = self.apply_roi(prediction, target)

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
    
    def apply_roi(self, prediction, target):
        coastline_indices = np.where(target == 3) # TODO: Change this to the correct value and pass it as an argument

        roi_mask = np.zeros_like(target)

        for row, col in zip(*coastline_indices):
            start_col = max(col - self.margin, 0)
            end_col = min(col + self.margin, target.shape[1])
        
            roi_mask[row, start_col:end_col] = 1

        prediction_roi = prediction[roi_mask == 1]
        target_roi = target[roi_mask == 1]

        return prediction_roi, target_roi
    