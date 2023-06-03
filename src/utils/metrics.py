from collections import defaultdict
from easydict import EasyDict
import numpy as np

class BatchMetric:
    def __init__(self) -> None:
        self.metrics = defaultdict(list)

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            if not isinstance(v, list):
                v = [v]
                self.metrics[k].extend(v)
    
    def compute(self, kind='mean'):
        output = EasyDict()
        if kind =='mean':
            for k, v in self.metrics.items():
                output[k] = np.mean(v)

        return output
    

class BatchMeter(object):
    def __init__(self, name, summary_type=None) -> None:
        self.name = name
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.value = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
    
    def get_value(self):
        if self.summary_type == 'mean':
            return self.avg
        elif self.summary_type == 'sum':
            return self.sum
        else:
            raise Exception(f'{self.summary_type} must be mean/sum')
        

if __name__ == "__main__":
    metric = BatchMeter('box_loss', 'sum')
    for _ in range(1000):
        metric.update(np.random.randint(1, 10e3))
        print(metric.get_value())
    print(metric.get_value())