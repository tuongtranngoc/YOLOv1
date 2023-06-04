import numpy as np


class BatchMeter(object):
    def __init__(self, name) -> None:
        self.name = name
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
    
    def get_value(self, summary_type=None):
        if summary_type == 'mean':
            return self.avg
        elif summary_type == 'sum':
            return self.sum
        else:
           return self.value
        

if __name__ == "__main__":
    metric = BatchMeter('box_loss', 'sum')
    for _ in range(1000):
        metric.update(np.random.randint(1, 10e3))
        print(metric.get_value())
    print(metric.get_value())