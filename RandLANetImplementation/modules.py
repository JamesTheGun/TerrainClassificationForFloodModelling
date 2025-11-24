from dataclasses import dataclass
from data import modelData

import pandas as pd
from RandLANetImplementation.constants import TRAIN_PER_PER_TEST_FOLD, HYPER_PARAMETERS
from RandLANetImplementation.data import modelData
import torch

def generate_neigbor_tensor(neighbor_tensor: torch.Tensor):
    pass

def _get_cordinate_features():
    pass

def _get_other_feature():
    pass

class LocSE():
    def __init__(self, in_tensor: torch.Tensor, neighbor_tensor: torch.Tensor):
        pass

class RandLANet:
    def __init__(self):
        self.data = modelData()
    def train():
        pass