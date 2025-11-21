from dataclasses import dataclass
import pandas as pd
from constants import HYPER_PARAMETERS, TRAIN_TO_HOLDOUT_RATIO, STRIDE, WINDOW_SIZE
from typing import List
import torch


def load_df():
    pass


def build_lidar_tensor(lidar_data_pd_df: pd.DataFrame) -> torch.Tensor:
    pass


def build_lidar_window_fold(
    points: torch.Tensor, window_size=WINDOW_SIZE, stride=STRIDE
):
    N, D = points.shape
    starts = torch.arange(0, N, stride)
    windows_idxs = (starts.unsqueeze(1) + torch.arange(window_size)) % N  # 2d
    windows = points[windows_idxs]
    return windows


@dataclass
class modelData:
    """prepare and store data with methods for retreiving train/test split"""

    data: pd.DataFrame = None
    train_set: pd.DataFrame = None
    test_set: pd.DataFrame = None
    hyper_params: dict = None
    folds_train: List[torch.Tensor] = None
    folds_test: List[torch.Tensor] = None

    def prepare_hyper_params(self):
        self.hyper_params = HYPER_PARAMETERS

    def prepare_train_test(self, train_to_test_ratio: float = None):
        if not train_to_test_ratio:
            train_to_test_ratio = TRAIN_TO_HOLDOUT_RATIO
        self.train_set, self.test_set = self._get_train_test(
            self.data, train_to_test_ratio
        )

    def new_split(self, train_to_test_ratio: float = None):
        self.prepare_hyper_params()
        self.prepare_data()
        self.prepare_train_test(train_to_test_ratio)

    @staticmethod
    def _get_train_test(data: pd.DataFrame, train_to_test_ratio: float):
        pass

    @staticmethod
    def _get_folds(data: pd.DataFrame, ratio: float):
        pass

    def init_folds(self):
        folds = self._get_fold(self.data, TRAIN_TO_HOLDOUT_RATIO)

    def next_fold():
        pass

    def get_folds():
        pass

    def test_test_lebels(test_labels: pd.Series):
        pass
