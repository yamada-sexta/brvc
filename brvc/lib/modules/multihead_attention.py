from typing import Optional
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention as PyTorchMultiheadAttention

class MultiHeadAttention(nn.Module):
    """
    Wrapper around PyTorch's MultiheadAttention with Conv1d projections.
    
    Maintains backward compatibility with the original custom implementation
    while leveraging PyTorch's optimized MultiheadAttention layer.
    
    Args:
        channels: Input channel dimension
        out_channels: Output channel dimension
        n_heads: Number of attention heads
        p_dropout: Dropout probability (default: 0.0)
        window_size: Window size for relative attention (not used in wrapper, for backward compatibility)
        heads_share: Whether heads share relative embeddings (not used, for backward compatibility)
        block_length: Block length for local attention (not used, for backward compatibility)
        proximal_bias: Add proximal bias (not used, for backward compatibility)
        proximal_init: Initialize K from Q weights (default: False)
    """
    
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        heads_share: bool = True,
        block_length: Optional[int] = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        assert channels % n_heads == 0, "channels must be divisible by n_heads"
        
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        
        self.k_channels = channels // n_heads
        
        # Conv1d projection layers
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        
        # PyTorch's optimized MultiheadAttention
        self.mha = PyTorchMultiheadAttention(
            embed_dim=channels,
            num_heads=n_heads,
            dropout=p_dropout,
            bias=True,
            batch_first=False,  # We'll use (seq_len, batch, embed_dim) format
        )
        
        # Dropout layer
        self.drop = nn.Dropout(p_dropout)
        
        # Initialize Conv1d weights
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                if self.conv_k.bias is not None and self.conv_q.bias is not None:
                    self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Query tensor of shape [batch, channels, time]
            c: Key/Value context tensor of shape [batch, channels, time]
            attn_mask: Optional attention mask of shape [batch, time_q, time_kv]
        
        Returns:
            Output tensor of shape [batch, out_channels, time]
        """
        # Project Q, K, V using Conv1d
        q = self.conv_q(x)  # [batch, channels, time_q]
        k = self.conv_k(c)  # [batch, channels, time_kv]
        v = self.conv_v(c)  # [batch, channels, time_kv]
        
        # Reshape from [batch, channels, time] to [time, batch, channels] for MultiheadAttention
        # This is the (seq_len, batch, embed_dim) format that MultiheadAttention expects
        q = q.transpose(1, 2)  # [batch, time_q, channels] -> transpose needed first
        k = k.transpose(1, 2)  # [batch, time_kv, channels]
        v = v.transpose(1, 2)  # [batch, time_kv, channels]
        
        q = q.transpose(0, 1)  # [time_q, batch, channels]
        k = k.transpose(0, 1)  # [time_kv, batch, channels]
        v = v.transpose(0, 1)  # [time_kv, batch, channels]
        
        # Handle attention mask format
        # Input mask is [batch, time_q, time_kv] where 0 means mask out
        # MultiheadAttention expects mask where True/1.0 means mask out
        if attn_mask is not None:
            # Accept several input formats for attn_mask coming from upstream modules:
            #  - [batch, time_q, time_kv]
            #  - [batch, 1, time_q, time_kv]
            #  - [batch, n_heads, time_q, time_kv]
            # Upstream uses 0 to mean "mask out". PyTorch's MultiheadAttention expects a
            # boolean mask where True indicates positions that should be masked. We'll
            # normalize to a boolean tensor of shape [batch*n_heads, time_q, time_kv].

            # If numeric, convert to boolean mask where True means masked (attn_mask==0)
            mask = attn_mask
            # If mask is float/int, compare to zero to produce boolean
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask, device=q.device)

            # Normalize to CPU/GPU/device and boolean
            mask = (mask == 0)

            if mask.dim() == 3:
                # [batch, time_q, time_kv] -> expand over heads
                batch_size = mask.size(0)
                mask = mask.unsqueeze(1)  # [batch, 1, time_q, time_kv]
                mask = mask.expand(batch_size, self.n_heads, -1, -1)
                mask = mask.reshape(batch_size * self.n_heads, mask.size(2), mask.size(3))
                attn_mask = mask
            elif mask.dim() == 4:
                # Could be [batch, 1, time_q, time_kv] or [batch, n_heads, time_q, time_kv]
                batch_size = mask.size(0)
                if mask.size(1) == 1:
                    # squeeze the singleton head dim and expand across heads
                    mask = mask.squeeze(1)  # [batch, time_q, time_kv]
                    mask = mask.unsqueeze(1).expand(batch_size, self.n_heads, -1, -1)
                    mask = mask.reshape(batch_size * self.n_heads, mask.size(2), mask.size(3))
                    attn_mask = mask
                elif mask.size(1) == self.n_heads:
                    # Already has head dim: reshape to [batch*n_heads, time_q, time_kv]
                    attn_mask = mask.reshape(batch_size * self.n_heads, mask.size(2), mask.size(3))
                else:
                    # Unexpected head dimension size: try to squeeze any singleton dims
                    # and fall back to treating as [batch, time_q, time_kv]
                    squeezed = mask.squeeze()
                    if squeezed.dim() == 3:
                        batch_size = squeezed.size(0)
                        attn_mask = squeezed.unsqueeze(1).expand(batch_size, self.n_heads, -1, -1)
                        attn_mask = attn_mask.reshape(batch_size * self.n_heads, attn_mask.size(2), attn_mask.size(3))
                    else:
                        raise ValueError(f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}")
            else:
                raise ValueError(f"Unsupported attn_mask number of dims: {attn_mask.dim()}")
        
        # Apply MultiheadAttention
        attn_output, _ = self.mha(q, k, v, attn_mask=attn_mask, need_weights=False)
        
        # Reshape back to [batch, channels, time_q]
        attn_output = attn_output.transpose(0, 1)  # [batch, time_q, channels]
        attn_output = attn_output.transpose(1, 2)  # [batch, channels, time_q]
        
        # Output projection
        output = self.conv_o(attn_output)  # [batch, out_channels, time_q]
        
        return output


def test():
    batch_size = 2
    channels = 16
    time = 10
    n_heads = 4

    x = torch.randn(batch_size, channels, time)
    c = torch.randn(batch_size, channels, time)
    attn_mask = torch.randint(0, 2, (batch_size, time, time))

    mha = MultiHeadAttention(channels, channels, n_heads)
    output = mha(x, c, attn_mask)
    print("Output shape:", output.shape)
    # Expected shape: [batch_size, channels, time]
    print("Expected shape:", (batch_size, channels, time))

if __name__ == "__main__":
    test()