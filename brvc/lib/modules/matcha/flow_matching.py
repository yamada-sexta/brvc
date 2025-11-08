from abc import ABC
from typing import Optional, TypedDict
import torch
import torch.nn.functional as F
import logging
# from matcha.models.components.decoder import Decoder
# from matcha.utils.pylogger import get_pylogger

# log = get_pylogger(__name__)

log = logging.getLogger(__name__)

class CFMParameters(TypedDict):
    solver: str
    sigma_min: Optional[float]


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats: int,
        cfm_params: CFMParameters,
        n_spks: int = 1,
        spk_emb_dim: int = 128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min if cfm_params.sigma_min else 1e-4
        
        self.estimator = None

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(
            z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond
        )

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: Optional[torch.Tensor],
        cond: Optional[torch.Tensor],
    ):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(
        self,
        x1: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        spks: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(
            self.estimator(y, mask, mu, t.squeeze(), spks), u, reduction="sum"
        ) / (torch.sum(mask) * u.shape[1])
        return loss, y




class CFM(BASECFM):
    def __init__(
        self,
        in_channels: int,
        out_channel: int,
        cfm_params: CFMParameters,
        decoder_params: dict,
        n_spks: int = 1,
        spk_emb_dim: int = 64,
    ):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
        self.estimator = Decoder(
            in_channels=in_channels, out_channels=out_channel, **decoder_params
        )
