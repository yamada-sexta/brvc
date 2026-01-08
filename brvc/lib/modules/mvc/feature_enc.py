from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from lib.modules.multihead_attention import MultiHeadAttention
from lib.utils.misc import sequence_mask


class TextEncoderParams(Protocol):
    """Protocol for text encoder parameters."""

    n_feats: int
    n_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = torch.nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.conv_2 = torch.nn.Conv1d(
            filter_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))

            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x: Input tensor of shape [Batch, Channels, Time]
            x_mask: Mask tensor of shape [Batch, 1, Time]

        Returns:
            Encoded tensor of shape [Batch, Channels, Time]
        """
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        for i in range(self.n_layers):
            x = x * x_mask

            # --- Attention Block ---
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)

            # LayerNorm (Requires Channel-Last)
            # x is [B, C, T] -> transpose to [B, T, C]
            res = x + y
            res = res.transpose(1, 2)
            res = self.norm_layers_1[i](res)
            x = res.transpose(1, 2)  # Back to [B, C, T]

            # --- FFN Block ---
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)

            # LayerNorm (Requires Channel-Last)
            res = x + y
            res = res.transpose(1, 2)
            res = self.norm_layers_2[i](res)
            x = res.transpose(1, 2)

        x = x * x_mask
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        heads_share: bool = True,
        p_dropout: float = 0.0,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ) -> None:
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)

        # from https://nn.labml.ai/transformers/rope/index.html
        self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
        self.key_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)

        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            if self.conv_k.bias is not None and self.conv_q.bias is not None:
                self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)

        query = self.query_rotary_pe(query)
        key = self.key_rotary_pe(key)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    @staticmethod
    def _attention_bias_proximal(length: int) -> torch.Tensor:
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        encoder_params: TextEncoderParams,
        n_feats: int,  # Target dimensions (e.g., 80 for Mel)
        hubert_dim: int = 768,  # Input dimensions from HuBERT
    ) -> None:
        super().__init__()
        self.n_feats = n_feats
        self.n_channels = encoder_params.n_channels

        # 1. Linear projection to map HuBERT features to the encoder's hidden dimension
        self.feature_proj = torch.nn.Conv1d(hubert_dim, self.n_channels, 1)

        # 2. Transformer Encoder
        # Now uses a fixed internal dimension since there is no speaker conditioning
        self.encoder = Encoder(
            self.n_channels,
            encoder_params.filter_channels,
            encoder_params.n_heads,
            encoder_params.n_layers,
            encoder_params.kernel_size,
            encoder_params.p_dropout,
        )

        # 3. Final projection to the target feature space (mu)
        self.proj_m = torch.nn.Conv1d(self.n_channels, self.n_feats, 1)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): HuBERT features
                shape: (batch_size, hubert_dim, seq_length)
            x_lengths (torch.Tensor): lengths of the sequences
                shape: (batch_size,)

        Returns:
            mu (torch.Tensor): Predicted mean for the Flow decoder
                shape: (batch_size, n_feats, seq_length)
            x_mask (torch.Tensor): Mask based on input lengths
        """
        # Create mask: [Batch, 1, Time]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        # Project input to model dimension and apply mask
        x = self.feature_proj(x) * x_mask

        # Pass through Transformer encoder layers
        x = self.encoder(x, x_mask)

        # Project to target feature space
        mu = self.proj_m(x) * x_mask

        return mu, x_mask
