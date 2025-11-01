from collections import OrderedDict
import traceback

import torch


def save_final(
    ckpt: dict,
    sr: int,
    filter_length: int,
    inner_channels: int,
    hidden_channels: int,
    filter_channels: int,
    n_heads: int,
    n_layers: int,
    kernel_size: int,
    p_dropout: float,
    resblock: str,
    resblock_kernel_sizes: list,
    resblock_dilation_sizes: list,
    upsample_rates: list,
    upsample_initial_channel: int,
    upsample_kernel_sizes: list,
    spk_embed_dim: int,
    gin_channels: int,
    sampling_rate: int,
    if_f0: bool,
    name: str,
    epoch: int,
    version: str,
):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = [
            filter_length // 2 + 1,
            32,
            inner_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sampling_rate,
        ]
        opt["info"] = "%sepoch" % epoch
        opt["sr"] = sr
        opt["f0"] = if_f0
        opt["version"] = version
        torch.save(opt, f"assets/weights/{name}.pth")
        return "Success."
    except:
        return traceback.format_exc()
