from typing import Optional, Tuple
import torch


def slice_segments2(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4
) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def slice_segments(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4
) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, segment_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, d, t = x.size()
    if x_lengths is None:
        # x_lengths = t
        x_lengths = torch.ones([b], device=x.device).to(dtype=torch.long) * t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str
