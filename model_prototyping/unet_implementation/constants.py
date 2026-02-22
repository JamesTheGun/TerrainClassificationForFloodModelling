"""
UNet Block Architecture Configurations

Each config describes a drop-in variant of a UNet convolutional block.
These are *descriptive configs* (not executable by themselves) intended
to be consumed by a block builder / factory.
"""

# ---------------------------------------------------------------------
# 0) Baseline: Conv → GN → ReLU → Conv → GN → ReLU
# ---------------------------------------------------------------------
UNET_BLOCK_BASELINE = {
    "conv_layers": [
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
    ],
    "norm_type": "GroupNorm",
    "norm_params": {"num_groups": "min(groups, out_ch)"},
    "activation": "ReLU",
    "activation_params": {"inplace": True},
}

UNET_BLOCK_CONFIG = UNET_BLOCK_BASELINE

# ---------------------------------------------------------------------
# 1) Pre-activation block (GN → ReLU → Conv)
# ---------------------------------------------------------------------
UNET_BLOCK_PREACT = {
    "pre_activation": True,
    "conv_layers": [
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
    ],
    "norm_type": "GroupNorm",
    "norm_params": {"num_groups": "min(groups, channels)"},
    "activation": "ReLU",
    "activation_params": {"inplace": True},
}

# ---------------------------------------------------------------------
# 2) Dropout-regularised block
# ---------------------------------------------------------------------
UNET_BLOCK_DROPOUT = {
    "conv_layers": [
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
    ],
    "norm_type": "GroupNorm",
    "norm_params": {"num_groups": "min(groups, out_ch)"},
    "activation": "ReLU",
    "activation_params": {"inplace": True},
    "dropout": {
        "type": "Dropout2d",
        "p": 0.1,
        "after_each_conv": True,
    },
}

# ---------------------------------------------------------------------
# 3) Residual UNet block
# ---------------------------------------------------------------------
UNET_BLOCK_RESIDUAL = {
    "residual": True,
    "skip_connection": {
        "type": "identity_or_1x1_conv",
        "bias": False,
    },
    "conv_layers": [
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
    ],
    "norm_type": "GroupNorm",
    "norm_params": {"num_groups": "min(groups, out_ch)"},
    "activation": "ReLU",
    "activation_params": {"inplace": True},
    "post_residual_activation": True,
}

# ---------------------------------------------------------------------
# 4) SiLU / Swish activation block
# ---------------------------------------------------------------------
UNET_BLOCK_SILU = {
    "conv_layers": [
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
    ],
    "norm_type": "GroupNorm",
    "norm_params": {"num_groups": "min(groups, out_ch)"},
    "activation": "SiLU",
    "activation_params": {"inplace": True},
}

# ---------------------------------------------------------------------
# 5) Dilated receptive-field expansion block
# ---------------------------------------------------------------------
UNET_BLOCK_DILATED = {
    "conv_layers": [
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "dilation": 1,
            "bias": False,
        },
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 2,
            "dilation": 2,
            "bias": False,
        },
    ],
    "norm_type": "GroupNorm",
    "norm_params": {"num_groups": "min(groups, out_ch)"},
    "activation": "ReLU",
    "activation_params": {"inplace": True},
}

# ---------------------------------------------------------------------
# 6) Depthwise-separable convolution block
# ---------------------------------------------------------------------
UNET_BLOCK_DEPTHWISE = {
    "conv_layers": [
        {
            "type": "depthwise",
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
        {
            "type": "pointwise",
            "kernel_size": 1,
            "padding": 0,
            "bias": False,
        },
        {
            "type": "depthwise",
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
        {
            "type": "pointwise",
            "kernel_size": 1,
            "padding": 0,
            "bias": False,
        },
    ],
    "norm_type": "GroupNorm",
    "norm_params": {"num_groups": "min(groups, out_ch)"},
    "activation": "ReLU",
    "activation_params": {"inplace": True},
}

# ---------------------------------------------------------------------
# 7) Single-activation (edge-preserving) block
# ---------------------------------------------------------------------
UNET_BLOCK_SINGLE_ACTIVATION = {
    "conv_layers": [
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
        {
            "in_channels": None,
            "out_channels": None,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
    ],
    "norm_type": "GroupNorm",
    "norm_params": {"num_groups": "min(groups, out_ch)"},
    "activation": "ReLU",
    "activation_params": {"inplace": True},
    "apply_activation_after_last_conv": False,
}

# ---------------------------------------------------------------------
# Registry (optional convenience)
# ---------------------------------------------------------------------
UNET_BLOCK_REGISTRY = {
    "baseline": UNET_BLOCK_BASELINE,
    "preact": UNET_BLOCK_PREACT,
    "dropout": UNET_BLOCK_DROPOUT,
    "residual": UNET_BLOCK_RESIDUAL,
    "silu": UNET_BLOCK_SILU,
    "dilated": UNET_BLOCK_DILATED,
    "depthwise": UNET_BLOCK_DEPTHWISE,
    "single_activation": UNET_BLOCK_SINGLE_ACTIVATION,
}
