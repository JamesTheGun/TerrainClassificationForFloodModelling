from dataclasses import dataclass
import pandas as pd
from RandLANetImplementation.constants import HYPER_PARAMETERS, TRAIN_PER_HOLDOUT, POSITIVE_CLASS_DIR, NEGATIVE_CLASS_DIR, TRAIN_PER_TEST_FOLD
from typing import List, Tuple
import torch
from sklearn.cluster import KMeans
import laspy

def load_combined_pos_neg_df() -> pd.DataFrame:
    pos_class = laspy.read(POSITIVE_CLASS_DIR)
    neg_class = laspy.read(NEGATIVE_CLASS_DIR)
    pos_class = pd.DataFrame({dim.name: pos_class[dim.name] 
                   for dim in pos_class.point_format.dimensions})
    neg_class = pd.DataFrame({dim.name: neg_class[dim.name] 
                   for dim in neg_class.point_format.dimensions})
    pos_class["label"] = 1
    neg_class["label"] = 0
    combined = pd.concat([pos_class, neg_class])
    #TO DO... unfuck this coercsion thing, like we need to make
    #sure we are actually doing this right... for now just fucking
    #stuff everything into numeric...
    combined = combined.apply(pd.to_numeric, errors = 'coerce')
    return combined

def build_lidar_tensor(las_df: pd.DataFrame) -> torch.Tensor:
    #consider using mixed procision for some vars in future...
    tensor = torch.tensor(las_df.to_numpy(), dtype=torch.float32)
    return tensor

def las_split_kmeans(points: torch.Tensor, split_count=30) -> List[torch.Tensor]:
    #build folds using kmeans
    coords = points[:, :2].cpu().numpy() #split using x and y
    labels = KMeans(n_clusters=split_count).fit_predict(coords)

    labels = torch.from_numpy(labels).long()

    splits = []
    for f in range(split_count):
        idx = torch.nonzero(labels == f, as_tuple=False).squeeze(1)
        splits.append(idx) 
    return splits

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
        if not self.data:
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
        print(test_idxs)
        print(train_idxs)
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
