from typing import Optional, Tuple
import torch


def slice_segments2(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4
) -> torch.Tensor:
    """Slice 1D temporal segments from a (batch, time) tensor.

    Parameters
    ----------
    x
        Tensor of shape (batch, time).
    ids_str
        1D tensor of start indices with length equal to batch size. Each
        value specifies the starting time index for the corresponding
        batch element.
    segment_size
        Length of the temporal segment to extract.

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch, segment_size) containing the extracted
        slices. The returned tensor shares the same dtype and device as
        the input `x`.
    """
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def slice_segments(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4
) -> torch.Tensor:
    """Slice temporal segments from a (batch, channel, time) tensor.

    Parameters
    ----------
    x
        Tensor of shape (batch, channel, time).
    ids_str
        1D tensor of start indices with length equal to batch size.
    segment_size
        Length of the temporal segment to extract.

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch, channel, segment_size) containing the
        extracted slices. The returned tensor uses the same dtype and
        device as `x`.
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: Optional[torch.Tensor | int] = None,
    segment_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample fixed-length temporal segments from a batch.

    This function selects a random start index for each batch element and
    returns the corresponding slice of length ``segment_size`` together
    with the sampled start indices.

    Parameters
    ----------
    x
        Tensor of shape (batch, channel, time).
    x_lengths
        Optional 1D tensor of per-example valid lengths (<= time). If
        provided, sampling ensures the segment is fully contained within
        the valid length. If ``None``, the full time dimension is used
        for every example.
    segment_size
        Length of the temporal segment to extract.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - extracted segments as a tensor of shape
          (batch, channel, segment_size)
        - 1D tensor of sampled start indices (length batch)

    Notes
    -----
    - If ``x_lengths`` is provided it must be a 1D tensor with length
      equal to the batch size. Values should be >= ``segment_size``.
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
        # x_lengths = torch.ones([b], device=x.device).to(dtype=torch.long) * t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str
