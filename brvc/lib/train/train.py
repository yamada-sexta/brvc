import json
import os
import sys
import logging

from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid
logger = logging.getLogger(__name__)
now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))
import datetime
from random import randint, shuffle
import torch

from tap import Tap
from accelerate import Accelerator

default_config = {
  "train": {
    "log_interval": 200,
    "seed": 1234,
    "epochs": 20000,
    "learning_rate": 1e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 4,
    "fp16_run": True,
    "lr_decay": 0.999875,
    "segment_size": 17280,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0
  },
  "data": {
    "max_wav_value": 32768.0,
    "sampling_rate": 48000,
    "filter_length": 2048,
    "hop_length": 480,
    "win_length": 2048,
    "n_mel_channels": 128,
    "mel_fmin": 0.0,
    "mel_fmax": None
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0,
    "resblock": "1",
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [12,10,2,2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [24,20,4,4],
    "use_spectral_norm": False,
    "gin_channels": 256,
    "spk_embed_dim": 109
  }
}



from lib.train.utils.data import TextAudioCollateMultiNSFsid, TextAudioLoaderMultiNSFsid
from torch.utils.data import DataLoader
class TrainArgs(Tap):
    """Training arguments."""
    ngpu: int = 1 # Number of GPUs to use
    seed: int = 1234 # Random seed
    sample_rate: int = 48000 # Sampling rate
    train_filelist: str = "filelists/train.txt" # Path to training filelist
    hop_length: int = 480 # Hop length
    win_length: int = 2048 # Window length
    max_text_len: int = 5000 # Maximum text length
    min_text_len: int = 1 # Minimum text length
    max_wav_value: float = 32768.0 # Maximum waveform value
    filter_length: int = 2048 # Filter length
    
def run_train(
    args: TrainArgs,
):
    accelerator = Accelerator()
    
    train_dataset = TextAudioLoaderMultiNSFsid(
        audiopaths_and_text=args.train_filelist,
        max_wav_value=args.max_wav_value,
        sampling_rate=args.sample_rate,
        filter_length=args.filter_length,
        hop_length=args.hop_length,
        win_length=args.win_length,
        max_text_len=args.max_text_len,
        min_text_len=args.min_text_len,
    )
    
    logging.info(f"Training dataset size: {len(train_dataset)}")
    collate_fn = TextAudioCollateMultiNSFsid()
    
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        # batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    m = default_config["model"]
    
    net_g = SynthesizerTrnMsNSFsid(
        spec_channels=args.filter_length // 2 + 1,
        segment_size=default_config["train"]["segment_size"] // args.hop_length,
        inter_channels=m["inter_channels"],
        hidden_channels=m["hidden_channels"],
        filter_channels=m["filter_channels"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        kernel_size=m["kernel_size"],
        p_dropout=m["p_dropout"],
        resblock_version=m["resblock"],
        resblock_kernel_sizes=m["resblock_kernel_sizes"],
        resblock_dilation_sizes=m["resblock_dilation_sizes"],
        upsample_rates=m["upsample_rates"],
        upsample_initial_channel=m["upsample_initial_channel"],
        upsample_kernel_sizes=m["upsample_kernel_sizes"],
        spk_embed_dim=m["spk_embed_dim"],
        gin_channels=m["gin_channels"],
        sr=default_config["data"]["sampling_rate"],
        is_half=default_config["train"]["fp16_run"],
        txt_channels=768,
    )
    
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
    
    )

    train_sampler = torch.utils.data.DistributedSampler(train_dataset)