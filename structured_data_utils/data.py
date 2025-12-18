from typing import List, Tuple
from dataclasses import dataclass
from structuring import get_negative_geotiff_tensor, get_positive_geotiff_tensor, get_combined_geotiff_tensor
import pandas as pd
import torch
import subprocess

def standardise_geotiffs():
    import subprocess
    subprocess.run([
        "gdalwarp",
        "-t_srs", "EPSG:XXXX",
        "-tr", "1", "1",
        "-r", "bilinear",
        "-overwrite",
        "in.tif",
        "out.tif",
    ], check=True)

def load_combined_pos_neg_df_structured() -> pd.DataFrame:
    positive = get_positive_geotiff_tensor()
    combined = get_combined_geotiff_tensor()
    label = torch.full_like(combined[:1], 0)
    combined = torch.cat([label,combined], dim=0)
    

@dataclass
class modelData:
    """prepare and store data with methods for retreiving train/test split"""
    data: pd.DataFrame = None
    train_set: torch.Tensor = None
    test_set: torch.Tensor = None
    hyper_params: dict = None
    folds_train: List[torch.Tensor] = None
    folds_test: List[torch.Tensor] = None
    folds: list[torch.Tensor] = None
    fold_index: int = 0

    def prepare_data(self):
        self.data = load_combined_pos_neg_df_structured()

    def set_hyper_params(self, HYPER_PARAMETERS):
        self.hyper_params = HYPER_PARAMETERS

    def prepare_train_test(self, train_per_holdout: float = None):
        self.train_set, self.test_set = self._get_train_test_tensors(
            self.data, train_per_holdout
        )

    def prepare_folds(train_test_ratio_fold):
        pass

    def new_split(self, hyper_params: dict, train_to_hold_out_ratio: int = None, train_test_ratio_fold: int = None):
        if (self.data == None):
            self.prepare_data()
        self.set_hyper_params(hyper_params)
        self.prepare_train_test(train_to_hold_out_ratio)
        self.prepare_folds(train_test_ratio_fold)

    @classmethod
    def _get_train_test_tensors(cls, data: pd.DataFrame, train_per_holdout: int):
        las_tensor = build_lidar_tensor(data)
        return cls._get_train_test_sets(las_tensor, train_per_holdout)

    @staticmethod
    def _get_folds(data: torch.Tensor, train_per_holdout: int) -> List[torch.Tensor]:
        pass
    
    @staticmethod
    def _get_train_test_sets(data: torch.Tensor, test_per_train: int = 7) -> Tuple[torch.Tensor]:
        pass

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
