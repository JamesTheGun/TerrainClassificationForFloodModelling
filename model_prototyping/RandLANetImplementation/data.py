from dataclasses import dataclass
import pandas as pd
import numpy as np
from RandLANetImplementation.constants import HYPER_PARAMETERS, TRAIN_PER_HOLDOUT, TRAIN_PER_TEST_FOLD
from common.data_loading import load_combined_pos_neg_df, las_split_kmeans, build_lidar_tensor
from typing import List, Tuple
import torch
from sklearn.cluster import KMeans
import laspy

@dataclass
class modelData:
    """prepare and store data with methods for retreiving train/test split"""

    data: pd.DataFrame = None
    train_set: torch.Tensor = None
    test_set: torch.Tensor = None
    hyper_params: dict = None
    folds_train: List[torch.Tensor] = None
    folds_test: List[torch.Tensor] = None
    fold_index: int = 0

    def prepare_data(self):
        self.data = load_combined_pos_neg_df()

    def prepare_hyper_params(self):
        self.hyper_params = HYPER_PARAMETERS

    def prepare_train_test(self, train_per_holdout: float = None):
        if not train_per_holdout:
            train_per_holdout = TRAIN_PER_HOLDOUT
        self.train_set, self.test_set = self._get_train_test_tensors(
            self.data, train_per_holdout
        )

    def prepare_folds(self, train_per_test_fold: int = None):
        if not train_per_test_fold:
            train_per_test_fold = TRAIN_PER_TEST_FOLD
        print(type(self.train_set))
        self.folds = self._get_folds(self.train_set, train_per_test_fold)

    def new_split(self, train_to_hold_out: int = None, train_test_ratio_fold: int = None):
        if (self.data == None):
            self.prepare_data()
        self.prepare_hyper_params()
        self.prepare_train_test(train_to_hold_out)
        self.prepare_folds(train_test_ratio_fold)

    @classmethod
    def _get_train_test_tensors(cls, data: pd.DataFrame, train_per_holdout: int):
        las_tensor = build_lidar_tensor(data)
        return cls._get_train_test_sets(las_tensor, train_per_holdout)

    @staticmethod
    def _get_folds(data: torch.Tensor, train_per_holdout: int) -> List[torch.Tensor]:
        return las_split_kmeans(data, train_per_holdout)
    
    @staticmethod
    def _get_train_test_sets(data: torch.Tensor, test_per_train: int = 7) -> Tuple[torch.Tensor]:
        splits = las_split_kmeans(data, test_per_train + 1)
        test_idxs = splits[0]
        train_idxs = torch.cat(splits[1:], dim=0)
        train = data[train_idxs]
        test = data[test_idxs]
        return train, test

    def next_fold(self) -> torch.Tensor | None:
        if self.fold_index > len(self.folds):
            return False
        fold = self.folds[self.fold_index]
        self.fold_index+=1
        return fold

    def get_folds(self) -> List[torch.Tensor]:
        return self.folds
    
    def get_data_from_fold(self, fold_index: int):
        return self.test_set[self.folds[fold_index]]
