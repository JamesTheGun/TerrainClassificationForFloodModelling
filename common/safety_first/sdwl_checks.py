import torch


def check_data_sdwl(data: torch.Tensor):
    assert isinstance(
        data, torch.Tensor
    ), f"data should be a torch.Tensor but got {type(data)}"
    assert (
        data.ndim == 4
    ), f"data should be in the format (num_segments, channels, height, width), but got {data.shape}"


def check_labels_sdwl(labels: torch.Tensor):
    assert isinstance(
        labels, torch.Tensor
    ), f"labels should be a torch.Tensor but got {type(labels)}"
    assert (
        labels.ndim == 4
    ), f"labels should be in the format (num_segments, channels, height, width), but got {labels.shape}"


def check_params_sdwl(data: torch.Tensor, labels: torch.Tensor):
    check_data_sdwl(data)
    check_labels_sdwl(labels)
    assert (
        data.shape[0] == labels.shape[0]
    ), "labels' number of segments do not match the given dataset's number of segments"
    assert (
        data.shape[2:] == labels.shape[2:]
    ), "labels' height and width do not match the given dataset's height and width"
