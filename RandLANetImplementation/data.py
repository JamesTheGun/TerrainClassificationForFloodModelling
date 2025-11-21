from dataclasses import dataclass
import pandas as pd
from constants import HYPER_PARAMETERS, TRAIN_TO_TEST_RATIO
from typing import List
import torch


def load_df():
    pass


def build_lidar_tensor(lidar_data_pd: pd.DataFrame) -> torch.Tensor:
    pass


def build_lidar_window_fold(points: torch.Tensor):
    N, D = points.shape


@dataclass
class Fold:
    train_idx: pd.Series
    val_idx: pd.Series

    def get_train(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.iloc[self.train_idx]

    def get_val(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.iloc[self.val_idx]


@dataclass
class modelData:
    """prepare and store data with methods for retreiving train/test split"""

    data: pd.DataFrame = None
    train_set: pd.DataFrame = None
    test_set: pd.DataFrame = None
    hyper_params: dict = None
    fold: List[Fold]

    def prepare_hyper_params(self):
        self.hyper_params = HYPER_PARAMETERS

    def prepare_train_test(self, train_to_test_ratio: float = None):
        if not train_to_test_ratio:
            train_to_test_ratio = TRAIN_TO_TEST_RATIO
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
    def _get_fold(data: pd.DataFrame, ratio: float):
        pass

    def init_folds():
        pass

    def next_fold():
        pass

    def get_folds():
        pass

    def test_test_lebels(test_labels: pd.Series):
        pass
