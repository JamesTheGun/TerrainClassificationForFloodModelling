from typing import Iterator
import torch
import random

class DataWithLabels:
    data: torch.Tensor
    labels: torch.Tensor

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        assert data.shape == labels.shape, "labels' shape do not match the given dataset's shape"
        self.data = data
        self.labels = labels

    def get_iterable(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return zip(self.data, self.labels)
    
    def get_hacky_fold_iterable(self, fold_size = 20):
        window_start = random.randint(0, len(self.data))
        window_end = window_start + fold_size
        return zip(self.data[window_start:window_end], self.labels[window_start:window_end])

