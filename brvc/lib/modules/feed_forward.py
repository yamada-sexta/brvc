from typing import Optional
from torch import nn
import torch
from torch.nn import functional as F


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: Optional[str] = None,
        causal: bool = False,
    ) -> None:
        super(FeedForwardNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.is_activation = True if activation == "gelu" else False

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def padding(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        if self.causal:
            padding = self._causal_padding(x * x_mask)
        else:
            padding = self._same_padding(x * x_mask)
        return padding

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(self.padding(x, x_mask))
        if self.is_activation:
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)

        x = self.conv_2(self.padding(x, x_mask))
        return x * x_mask

    def _causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        pad_l: int = self.kernel_size - 1
        pad_r: int = 0
        x = F.pad(
            x,
            [pad_l, pad_r, 0, 0, 0, 0],
        )
        return x

    def _same_padding(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        pad_l: int = (self.kernel_size - 1) // 2
        pad_r: int = self.kernel_size // 2
        x = F.pad(
            x,
            [pad_l, pad_r, 0, 0, 0, 0],
        )
        return x
