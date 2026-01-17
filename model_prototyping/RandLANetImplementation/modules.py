from dataclasses import dataclass
from RandLANetImplementation.data import modelData

import pandas as pd
from RandLANetImplementation.constants import HYPER_PARAMETERS, NIEGBBOUR_COUNT
from RandLANetImplementation.data import modelData
import torch
import faiss

def random_sample(points: torch.tensor, percentage_to_sample: float):
    point_count = points.shape[0]
    num_to_sample = int(point_count*percentage_to_sample)
    idx = torch.randperm(point_count)[:num_to_sample]
    return points[idx]

import torch


def get_knn_points(points_downsampled: torch.Tensor,
                   all_points: torch.Tensor,
                   number_of_neighbours: int):
    
    if not points_downsampled.device.type == "cpu":
        points_downsampled = points_downsampled.cpu()
    if not all_points.device.type == "cpu":
        all_points = all_points.cpu()

    Xq_t = points_downsampled[:, :2].float().contiguous()
    Xb_t = all_points[:, :2].float().contiguous()

    Xq = Xq_t.numpy()
    Xb = Xb_t.numpy()

    d = Xb.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(Xb)

    _, idx = index.search(Xq, number_of_neighbours)

    idx = torch.from_numpy(idx).long()

    return all_points[idx]


def _get_cordinate_features():
    pass

def _get_other_feature():
    pass

class LocSE():
    def __init__(self, in_tensor: torch.Tensor, neighbor_tensor: torch.Tensor):
        self.in_tensor = in_tensor
        self.neighbor_tensor = neighbor_tensor

class RandLANet:
    def __init__(self):
        self.data = modelData()
    def train():
        pass