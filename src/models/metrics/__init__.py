from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
from src.models.metrics.torcheval_metrics import TorchevalMetric
from src.models.metrics.loss_metrics import LossMetric

class Metrics():
    def __init__(self, phase, num_classes = None, average = 'macro'):
        self.phase = phase
        self.num_classes = num_classes
        self.average = average

        self.general_metrics = {
            'accuracy': TorchevalMetric(f'{phase}_accuracy', MulticlassAccuracy(num_classes=num_classes, average=average)),
            'f1_score': TorchevalMetric(f'{phase}_f1', MulticlassF1Score(num_classes=num_classes, average=average)),
            'precision': TorchevalMetric(f'{phase}_precision', MulticlassPrecision(num_classes=num_classes, average=average)),
            'recall': TorchevalMetric(f'{phase}_recall', MulticlassRecall(num_classes=num_classes, average=average)),
            'confusion_matrix': TorchevalMetric(f'{phase}_confusion_matrix', MulticlassConfusionMatrix(num_classes=num_classes, normalize="true")),
        }

        self.loss_metric = {
            'loss': LossMetric(f'{phase}_loss')
        }

        self.metrics = {**self.general_metrics, **self.loss_metric}

    def compute(self, epoch, mlflow=False):
        for key, metric in self.metrics.items():
            if key == 'confusion_matrix':
                data = metric.compute(False)
                continue
            else:
                data = metric.compute()
            if mlflow:
                metric.log_metric(data, epoch+1)

    def update(self, prediction, target, loss):
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

    def getLastMetric(self, metric_name):
        return self.metrics[metric_name].data[-1]

    def getMetricsData(self):
        return {key: value.data for key, value in self.metrics.items()}
    

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