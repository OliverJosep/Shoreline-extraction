import torch
import numpy as np
import torcheval.metrics

class TorchevalMetric:
    def __init__(self, name: str, metric: torcheval.metrics):
        self.metric = metric
        self.data = []
        self.name = name

    def update(self, prediction, target):
        prediction = prediction.to(torch.int64)
        target = target.to(torch.int64)
        self.metric.update(prediction, target)

    def reset(self):
        self.metric.reset()

    def compute(self, format = True):
        data = round(self.metric.compute().item(), 8) if format else self.metric.compute()

        self.data.append(data)

        return data

    def get_name(self):
        return self.name

    def get_last_metric(self):
        return self.data[-1]

    def get_epoch_info(self, epoch):
        return f'{self.name}: {self.data[epoch]}'
    
    def get_best_metric(self):
        return np.max(self.data)
    
    def __str__(self):
        return f'{type(self.metric)}: {self.compute()}'