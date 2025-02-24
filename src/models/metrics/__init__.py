from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
from src.models.metrics.torcheval_metrics import TorchevalMetric
from src.models.metrics.loss_metrics import LossMetric
# import binary metrics
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryConfusionMatrix

import numpy as np

class Metrics():
    def __init__(self, phase, num_classes = None, average = 'macro', loss = True, use_margin: bool = False, margin: int = 10):
        self.phase = phase
        self.num_classes = num_classes
        self.average = average


        self.use_margin = use_margin
        self.margin = margin

        if num_classes is not None and num_classes > 1:
            self.general_metrics = {
                'accuracy': TorchevalMetric(f'{phase}_accuracy', MulticlassAccuracy(num_classes=num_classes, average=average)),
                'f1_score': TorchevalMetric(f'{phase}_f1', MulticlassF1Score(num_classes=num_classes, average=average)),
                'precision': TorchevalMetric(f'{phase}_precision', MulticlassPrecision(num_classes=num_classes, average=average)),
                'recall': TorchevalMetric(f'{phase}_recall', MulticlassRecall(num_classes=num_classes, average=average)),
                'confusion_matrix': TorchevalMetric(f'{phase}_confusion_matrix', MulticlassConfusionMatrix(num_classes=num_classes, normalize="true"))
            }
        else:
            self.general_metrics = {
                'accuracy': TorchevalMetric(f'{phase}_accuracy', BinaryAccuracy()),
                'f1_score': TorchevalMetric(f'{phase}_f1', BinaryF1Score()),
                'precision': TorchevalMetric(f'{phase}_precision', BinaryPrecision()),
                'recall': TorchevalMetric(f'{phase}_recall', BinaryRecall()),
                'confusion_matrix': TorchevalMetric(f'{phase}_confusion_matrix', BinaryConfusionMatrix(normalize="true"))
            }

        self.loss_metric = {'loss': LossMetric(f'{phase}_loss')} if loss else {}

        self.metrics = {**self.general_metrics, **self.loss_metric}

    def compute(self):
        for key, metric in self.metrics.items():
            if key == 'confusion_matrix':
                data = metric.compute(False)
                continue
            else:
                data = metric.compute(False)

    def update(self, prediction, target, loss = None):
        prediction = prediction.flatten()
        target = target.flatten()
        for key, metric in self.metrics.items():
            if key == 'loss':
                metric.update(loss)
            else:
                metric.update(prediction, target)

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def print_metrics(self):
        for metric in self.metrics.values():
            metric.print_data()

    def get_epoch_info(self, epoch):
        text = f"Epoch {epoch + 1}:\n"
        for metric in self.metrics.values():
            text += f"\t{metric.get_epoch_info(epoch)}\n"
        return text
    
    def get_last_epoch_info(self):
        text = ""
        for metric in self.metrics.values():
            text += f"\t{metric.get_name()}: {metric.get_last_metric()}\n"
        return text
    
    def get_last_epoch_info_dict(self):
        return {metric.get_name(): metric.get_last_metric() for metric in self.metrics.values() if 'confusion_matrix' not in metric.get_name()}

    def get_last_loss(self):
        if 'loss' not in self.metrics:
            return None
        return self.metrics['loss'].data[-1]

    def get_best_metric(self, metric_name):
        return self.metrics[metric_name].get_best_metric()

    def get_best_accuracy(self):
        best_acc_index = 0
        best_acc = 0

        for index, acc in enumerate(self.metrics['accuracy'].data):
            if acc >= best_acc:
                best_acc_index = index
                best_acc = acc

        return best_acc_index, best_acc, self.metrics['loss'].data[best_acc_index]
    
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
    