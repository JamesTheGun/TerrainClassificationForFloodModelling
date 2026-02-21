import torch


def check_data_dwl(data: torch.Tensor):
    assert isinstance(
        data, torch.Tensor
    ), f"data should be a torch.Tensor but got {type(data)}"
    assert (
        data.ndim == 3
    ), f"data should be in the format (channels, height, width), but got {data.shape}"


def check_labels_dwl(labels: torch.Tensor):
    assert isinstance(
        labels, torch.Tensor
    ), f"labels should be a torch.Tensor but got {type(labels)}"
    assert (
        labels.ndim == 3
    ), f"labels should be in the format (channels, height, width), but got {labels.shape}"


def check_params_dwl(data: torch.Tensor, labels: torch.Tensor):
    check_data_dwl(data)
    check_labels_dwl(labels)
    assert (
        data.shape[1:] == labels.shape[1:]
    ), "labels' height and width do not match the given dataset's height and width"
