import logging
import math
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ConvTranspose1d, Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm
from lib.modules.source_module_hn_nsf import SourceModuleHnNSF
from lib.modules.res_block import RES_BLOCK_VERSION, ResBlock1
from lib.utils.misc import init_weights


class GeneratorNSF(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock_version: RES_BLOCK_VERSION,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[tuple[int, int, int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int,
        sr: int,
        is_half: bool,
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
            is_half=is_half,
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
        res_block = ResBlock1
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_prev: int = upsample_initial_channel // (2**i)
            c_cur: int = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        # upsample_initial_channel // (2**i),
                        # upsample_initial_channel // (2 ** (i + 1)),
                        # k,
                        # u,
                        # padding=(k - u) // 2,
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
                        # 1,
                        # c_cur,
                        # kernel_size=stride_f0 * 2,
                        # stride=stride_f0,
                        # padding=stride_f0 // 2,
                        in_channels=1,
                        out_channels=c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                # self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
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
        x,
        f0,
        g: Optional[torch.Tensor] = None,
        n_res: Optional[torch.Tensor] = None,
    ):
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        if n_res is not None:
            assert isinstance(n_res, torch.Tensor)
            n = int(n_res.item())
            if n * self.upp != har_source.shape[-1]:
                har_source = F.interpolate(har_source, size=n * self.upp, mode="linear")
            if n != x.shape[-1]:
                x = F.interpolate(x, size=n, mode="linear")
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        # torch.jit.script() does not support direct indexing of torch modules
        # That's why I wrote this
        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            if i < self.num_upsamples:
                x = F.leaky_relu(x, self.lrelu_slope)
                x = ups(x)
                x_source = noise_convs(har_source)
                x = x + x_source
                xs: Optional[torch.Tensor] = None
                l = [i * self.num_kernels + j for j in range(self.num_kernels)]
                for j, resblock in enumerate(self.resblocks):
                    if j in l:
                        if xs is None:
                            xs = resblock(x)
                        else:
                            xs += resblock(x)
                # This assertion cannot be ignored! \
                # If ignored, it will cause torch.jit.script() compilation errors
                assert isinstance(xs, torch.Tensor)
                x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            # l.remove_weight_norm()
            logging.debug("Make sure it is a module with remove_weight_norm method")
            # Disable type checker for the next line
            l.remove_weight_norm()  # type: ignore

    def __prepare_scriptable__(self):
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                # The hook we want to remove is an instance of WeightNorm class, so
                # normally we would do `if isinstance(...)` but this class is not accessible
                # because of shadowing, so we check the module name directly.
                # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        for l in self.resblocks:
            for hook in self.resblocks._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)
        return self
