from typing import List, Tuple
from dataclasses import dataclass
import subprocess

import pandas as pd
import torch
import torch.nn.functional as F

from common.data_managment import DataWithLabelsGeoTethered, SegmentedDataWithLabels
from structured_data_utils.config.constants import ESPSG, RES, EMPTY_VAL
from structured_data_utils.structured_data_interfacing import (
    standardise_dataset,
    get_segments_with_sliding_window,
    remove_empty_segments,
    load_data_with_labels,
    put_nans_in_neggative_positions,
    infer_nans_segmented,
    determanistic_splice_tensors,
    normalise_dwl_local,
)


@dataclass
class ModelData:
    """prepare and store data with methods for retreiving train/test split"""

    data_with_labels: DataWithLabelsGeoTethered = None
    segmented_data_with_labels: SegmentedDataWithLabels = None
    train_set: DataWithLabelsGeoTethered = None
    test_set: DataWithLabelsGeoTethered = None

    def prepare_data(
        self,
        folder_name: str,
        sliding_window_size=300,
        stride=300,
        force_standardise: bool = False,
    ):
        standardise_dataset(folder_name, force=force_standardise)
        self.data_with_labels = load_data_with_labels(folder_name)
        self.data_with_labels.data = put_nans_in_neggative_positions(
            self.data_with_labels.data
        )
        self.data_with_labels = normalise_dwl_local(self.data_with_labels)
        self.segmented_data_with_labels = get_segments_with_sliding_window(
            self.data_with_labels,
            base_window_size=sliding_window_size,
            stride=stride,
        )
        self.segmented_data_with_labels = infer_nans_segmented(
            self.segmented_data_with_labels
        )
        # self.segmented_data_with_labels = infer_nans(self.segmented_data_with_labels)
        if torch.isnan(self.segmented_data_with_labels.data).any():
            print("infering nans failed...")

    @staticmethod
    def splice_model_data(model_datas: List["ModelData"]) -> "ModelData":
        segmented_data_tensors = [
            model_data.segmented_data_with_labels.data for model_data in model_datas
        ]
        segmented_labels_tensors = [
            model_data.segmented_data_with_labels.labels for model_data in model_datas
        ]

        segmented_data_spliced = determanistic_splice_tensors(segmented_data_tensors)
        segmented_labels_spliced = determanistic_splice_tensors(
            segmented_labels_tensors
        )

        segmented_labeled_data_spliced = SegmentedDataWithLabels(
            segmented_data_spliced, segmented_labels_spliced
        )

        spliced_model_data = ModelData(
            data_with_labels=None,
            segmented_data_with_labels=segmented_labeled_data_spliced,
        )

        return spliced_model_data
