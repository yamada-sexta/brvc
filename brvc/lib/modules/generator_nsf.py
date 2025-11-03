import logging
import math
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ConvTranspose1d, Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm
from lib.modules.source_module_hn_nsf import SourceModuleHnNSF
from lib.modules.res_block import RES_BLOCK_VERSION, ResBlock1, ResBlock2
from lib.utils.misc import init_weights


class GeneratorNSF(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock: RES_BLOCK_VERSION,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[tuple[int, int, int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int,
        sr: int,
        # is_half: bool,
        lrelu_slope: float,
    ):
        super(GeneratorNSF, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.f0_upsamp = nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr,
            harmonic_num=0,
            sine_amp=0.1,
            add_noise_std=0.003,
            voiced_threshod=0,
            # is_half=is_half,
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = Conv1d(
            # initial_channel, upsample_initial_channel, 7, 1, padding=3
            in_channels=initial_channel,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        # resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2
        # res_block = ResBlock2 if resblock_version == "2" else ResBlock1
        if isinstance(resblock, str):
            res_block = ResBlock2 if resblock == "2" else ResBlock1
        else:
            res_block = resblock
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_prev: int = upsample_initial_channel // (2**i)
            c_cur: int = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        in_channels=c_prev,
                        out_channels=c_cur,
                        kernel_size=k,
                        stride=u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    Conv1d(
                        in_channels=1,
                        out_channels=c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(
                    Conv1d(
                        in_channels=1,
                        out_channels=c_cur,
                        kernel_size=1,
                    )
                )
        self.resblocks = nn.ModuleList()

        ch: int = upsample_initial_channel  # Fix type checker error
        for i in range(len(self.ups)):
            ch: int = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    res_block(
                        channels=ch, kernel_size=k, dilation=d, lrelu_slope=lrelu_slope
                    )
                )

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = lrelu_slope

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        n_res: Optional[torch.Tensor] = None,
    ):
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            if xs is None:
                raise RuntimeError("xs is None")
            x = xs / self.num_kernels
            
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
