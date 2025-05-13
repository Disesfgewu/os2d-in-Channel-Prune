import os
import math
import random
import errno
from PIL import Image
from io import BytesIO
import base64
import numpy as np

import torch
from os2d.utils.logger import setup_logger
# logger = setup_logger("OS2D")

def get_data_path():
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    data_path = os.path.expanduser(os.path.abspath(data_path))
    return data_path


def get_trainable_parameters(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def count_model_parameters(net):
    num_params = 0
    num_param_groups = 0
    for p in get_trainable_parameters(net):
        num_param_groups += 1
        num_params += p.numel()
    return num_params, num_param_groups


def get_image_size_after_resize_preserving_aspect_ratio(h, w, target_size):
    aspect_ratio_h_to_w = float(h) / w
    w = int(target_size / math.sqrt(aspect_ratio_h_to_w))
    h = int(target_size * math.sqrt(aspect_ratio_h_to_w))
    h, w = (1 if s <= 0 else s for s in (h, w))  # filter out crazy one pixel images
    return h, w


def masked_select_or_fill_constant(a, mask, constant=0):
    constant_tensor = torch.tensor([constant], dtype=a.dtype, device=a.device)
    return torch.where(mask, a, constant_tensor)


def set_random_seed(random_seed, cuda=False):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if cuda:
        torch.cuda.manual_seed_all(random_seed)


def mkdir(path):
    """From https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/miscellaneous.py
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_image(image_path):
    with open(image_path, "rb") as f:
        img = Image.open(f)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.load()
    return img


def ceildiv(a, b):
    return -(-a // b)

def decode_base64_to_image(base64_str):
    return Image.open(BytesIO(base64.b64decode(base64_str)))


import torch
import torch.nn as nn
def prune_conv_layer(conv_layer, keep_indices):
    """
    Prune a Conv2d layer by keeping only the output channels in keep_indices.

    Args:
        conv_layer (nn.Conv2d): The original conv layer.
        keep_indices (list or tensor): Indices of output channels to keep.

    Returns:
        nn.Conv2d: New conv layer with reduced channels.
    """
    device = conv_layer.weight.device
    keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=device)

    new_out_channels = len(keep_indices)
    new_conv = nn.Conv2d(
        in_channels=conv_layer.in_channels,
        out_channels=new_out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=(conv_layer.bias is not None)
    ).to(device)

    # Copy weights
    new_conv.weight.data = conv_layer.weight.data[keep_indices].clone()

    # Copy bias if exists
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data[keep_indices].clone()
    # logger.info(f"Pruned conv layer from {conv_layer.out_channels} to {new_conv.out_channels} channels.")
    return new_conv


def prune_batchnorm_layer(bn_layer, keep_indices):
    device = bn_layer.weight.device
    keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=device)
    new_num_features = len(keep_indices)
    new_bn = nn.BatchNorm2d(
        num_features=new_num_features,
        eps=bn_layer.eps,
        momentum=bn_layer.momentum,
        affine=bn_layer.affine,
        track_running_stats=bn_layer.track_running_stats
    ).to(device)
    # 複製參數
    new_bn.weight.data = bn_layer.weight.data[keep_indices].clone()
    new_bn.bias.data = bn_layer.bias.data[keep_indices].clone()
    new_bn.running_mean.data = bn_layer.running_mean.data[keep_indices].clone()
    new_bn.running_var.data = bn_layer.running_var.data[keep_indices].clone()
    return new_bn

def prune_conv_in_channels(conv_layer, keep_indices):
    device = conv_layer.weight.device
    keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=device)
    new_in_channels = len(keep_indices)
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=(conv_layer.bias is not None)
    ).to(device)
    # slice in_channel
    new_conv.weight.data = conv_layer.weight.data[:, keep_indices, :, :].clone()
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data.clone()
    return new_conv
