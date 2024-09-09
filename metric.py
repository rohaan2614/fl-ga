"""
This module defines a `Metric` class that is used to track and compute the 
average of a metric (e.g., accuracy, loss) over multiple updates. 
"""

import random
import torch

class Metric:
    """
    Class to track and compute the average of a metric over multiple updates.
    """
    def __init__(self, name):
        """
        Initialize the metric with a given name and set initial values for sum and count.
        
        Args:
            name (str): Name of the metric (e.g., accuracy, loss).
        """
        self.name = name
        self.sum = 0
        self.n = 0

    def update(self, val):
        """
        Update the metric with a new value and increment the count.

        Args:
            val (float or torch.Tensor): New value to be added. If a tensor is provided, 
                                         it will be converted to a CPU float.
        """
        if isinstance(val, torch.Tensor):
            self.sum += val.detach().cpu()
        else:
            self.sum += val
        self.n += 1

    @property
    def avg(self):
        """
        Compute the average of the metric.

        Returns:
            float: The average value of the metric.
        """
        return self.sum / self.n
    
if __name__ == '__main__':
    metric = Metric(name='Accuracy')
    print("Metric:", metric.name)
    print("Generating random accuracies...\n", end='', flush=True)
    
    # Generate and update with random accuracy values
    random_numbers = [round(random.uniform(0.5, 1), 3) for _ in range(4)]
    for acc in random_numbers:
        print("\t-> ", acc)
        metric.update(acc)
    
    # Print the average accuracy
    print("\nAverage:", metric.avg)
