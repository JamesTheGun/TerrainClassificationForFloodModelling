from typing import List, Tuple
from dataclasses import dataclass
import subprocess

import pandas as pd
import torch
import torch.nn.functional as F

from common.data_managment import DataWithLabels, SegmentedDataWithLabels
from structured_data_utils.config.constants import ESPSG, RES, EMPTY_VAL
from structured_data_utils.structured_data_interfacing import get_segments_with_sliding_window, remove_empty_segments, load_data_with_labels, put_nans_in_neggative_positions, remove_segments_missing_positive, infer_nans_segmented, splice_tensors, bloat_positives

@dataclass
class ModelData:
    """prepare and store data with methods for retreiving train/test split"""
    data_with_labels: DataWithLabels = None 
    segmented_data_with_labels: SegmentedDataWithLabels = None
    hyper_params: dict = None
    train_set: DataWithLabels = None
    test_set: DataWithLabels = None

    def prepare_data(self, folder_name: str, sliding_window_size = 300, stride = 300):
        self.data_with_labels = load_data_with_labels(folder_name)
        self.data_with_labels.data = put_nans_in_neggative_positions(self.data_with_labels.data)
        self.segmented_data_with_labels = get_segments_with_sliding_window(self.data_with_labels, window_size = sliding_window_size, stride = stride)
        self.segmented_data_with_labels = remove_empty_segments(self.segmented_data_with_labels)
        self.segmented_data_with_labels = infer_nans_segmented(self.segmented_data_with_labels)
        self.segmented_data_with_labels = bloat_positives(self.segmented_data_with_labels)
        #self.segmented_data_with_labels = infer_nans(self.segmented_data_with_labels)
        if torch.isnan(self.segmented_data_with_labels.data).any():
            print("infering nans failed...")
        self.segmented_data_with_labels = remove_segments_missing_positive(self.segmented_data_with_labels)

    def set_hyper_params(self, HYPER_PARAMETERS: dict):
        self.hyper_params = HYPER_PARAMETERS

    def prepare_train_test(self, train_per_holdout: float = 0.6):
        self.train_set, self.test_set = self._get_train_test_tensors(
            train_per_holdout
        )

    def prepare_folds(train_test_ratio_fold):
        pass

    def new_split(self, hyper_params: dict, train_percent: int = None, train_test_ratio_fold: int = None):
        if train_percent is None:
            train_percent = 0.6
        print(train_percent)
        self.prepare_data()
        self.set_hyper_params(hyper_params)
        self.prepare_train_test(train_percent)
        self.prepare_folds(train_test_ratio_fold)

    def _get_train_test_tensors(self, train_percent: float) -> Tuple[SegmentedDataWithLabels, SegmentedDataWithLabels]:
        shape = self.segmented_data_with_labels.data.shape
        first_half_data = self.segmented_data_with_labels.data[int(shape[0]*train_percent) : ]
        second_half_data = self.segmented_data_with_labels.data[: int(shape[0]*train_percent)]
        first_half_labels = self.segmented_data_with_labels.labels[int(shape[0]*train_percent) : ]
        second_half_labels = self.segmented_data_with_labels.labels[: int(shape[0]*train_percent)]
        first_half_data_and_labels = SegmentedDataWithLabels(first_half_data, first_half_labels)
        second_half_data_and_labels = SegmentedDataWithLabels(second_half_data, second_half_labels)
        return first_half_data_and_labels, second_half_data_and_labels
    
    @staticmethod
    def splice_model_data(model_datas: List["ModelData"]) -> SegmentedDataWithLabels:

        segmented_data_tensors = [model_data.segmented_data_with_labels.data for model_data in model_datas]
        segmented_labels_tensors = [model_data.segmented_data_with_labels.labels for model_data in model_datas]

        segmented_data_spliced = splice_tensors(segmented_data_tensors)
        segmented_labels_spliced = splice_tensors(segmented_labels_tensors)

        segmented_labeled_data_spliced = SegmentedDataWithLabels(segmented_data_spliced, segmented_labels_spliced)

        spliced_model_data = ModelData(data_with_labels=None, segmented_data_with_labels=segmented_labeled_data_spliced)

        return spliced_model_data

        
    @staticmethod
    def _get_folds(data: torch.Tensor, train_per_holdout: int) -> List[torch.Tensor]:
        pass

    def next_fold(self) -> torch.Tensor | None:
        pass

    def get_folds(self) -> List[torch.Tensor]:
        pass

    def get_data_from_fold(self, fold_index: int):
        pass
