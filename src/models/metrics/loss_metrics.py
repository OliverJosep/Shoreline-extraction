import numpy as np

class LossMetric:
    def __init__(self, name):
        self.actual_loss = []
        self.data = []
        self.name = name

    def update(self, loss):
        self.actual_loss.append(loss)

    def reset(self):
        self.actual_loss = []

    def compute(self):
        data = sum(self.actual_loss) / len(self.actual_loss)
        self.data.append(data)
        return data

    # def log_metric(self, data, epoch):
    #     mlflow.log_metric(self.name, data, step=epoch)
      
    def get_name(self):
        return self.name
      
    def get_last_metric(self):
        return self.data[-1]
    
    def get_epoch_info(self, epoch):
        return f'{self.name}: {self.data[epoch]}'
    
    def get_best_metric(self):
        return np.min(self.data)

    def __str__(self):
        return f'{type(self)}: {self.compute()}'