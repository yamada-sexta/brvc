import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

from lib.modules.attentions_enc import Encoder
from lib.utils.misc import sequence_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        f0: bool = True,
        lrelu_slope: float = 0.1,
    ):
        super(TextEncoder, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(lrelu_slope, inplace=True)

        if f0 == True:
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256

        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        phone: torch.Tensor,
        pitch: torch.Tensor,
        lengths: torch.Tensor,
        skip_head: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # print(f"Phone tensor stats: min={phone.min()}, max={phone.max()}, shape={phone.shape}")
        # print(f"Pitch tensor stats: min={pitch.min()}, max={pitch.max()}, shape={pitch.shape}")
        # print(f"Phone embedding layer: {self.emb_phone}")
        # print(f"Pitch embedding layer: {self.emb_pitch}")

        if pitch == None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        
        if skip_head is not None:
            assert isinstance(skip_head, torch.Tensor)
            head = int(skip_head.item())
            x = x[:, :, head:]
            x_mask = x_mask[:, :, head:]
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask
