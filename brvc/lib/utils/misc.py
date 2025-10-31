from torch import nn
import torch
from typing import Optional, Tuple


def get_padding(kernel_size: int, dilation: int = 1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean: float = 0.0, std: float = 0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None):
    if max_length is None:
        max_length = int(length.max().item())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

