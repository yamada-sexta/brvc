import torch
import torch.nn as nn
from typing import Optional, Tuple


class Flip(nn.Module):
    # torch.jit.script() Compiled functions \
    # can't take variable number of arguments or \
    # use keyword-only arguments with defaults
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x, torch.zeros([1], device=x.device)
