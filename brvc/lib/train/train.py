import json
import os
from pathlib import Path
import sys
from accelerate.utils import set_seed
from tqdm import tqdm
from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid
from accelerate.logging import get_logger

from lib.train.loss import discriminator_loss, feature_loss, generator_loss, kl_loss
from lib.train.utils.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from lib.utils.misc import clip_grad_value_
from lib.utils.slice import slice_segments

import torch.nn.functional as F

logger = get_logger(__name__)
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
        "c_kl": 1.0,
    },
    "data": {
        "max_wav_value": 32768.0,
        "sampling_rate": 48000,
        "filter_length": 2048,
        "hop_length": 480,
        "win_length": 2048,
        "n_mel_channels": 128,
        "mel_fmin": 0.0,
        "mel_fmax": None,
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
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [12, 10, 2, 2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [24, 20, 4, 4],
        "use_spectral_norm": False,
        "gin_channels": 256,
        "spk_embed_dim": 109,
    },
}


from lib.train.utils.data import TextAudioCollateMultiNSFsid, TextAudioLoaderMultiNSFsid
from torch.utils.data import DataLoader
from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid
from lib.modules.discriminators import MultiPeriodDiscriminatorV2


# class TrainArgs(Tap):
#     """Training arguments."""

#     # Required
#     train_filelist: str = "filelists/train.txt"  # Path to training filelist
#     model_dir: str = "logs/model"  # Directory to save checkpoints

#     # Training
#     epochs: int = 20000  # Number of epochs
#     batch_size: int = 4  # Batch size per GPU
#     learning_rate: float = 1e-4  # Learning rate
#     lr_decay: float = 0.999875  # Learning rate decay
#     seed: int = 1234  # Random seed

#     # Data
#     sample_rate: int = 48000  # Sampling rate
#     hop_length: int = 480  # Hop length
#     win_length: int = 2048  # Window length
#     max_text_len: int = 5000  # Maximum text length
#     min_text_len: int = 1  # Minimum text length
#     max_wav_value: float = 32768.0  # Maximum waveform value
#     filter_length: int = 2048  # Filter length

#     # Optimizer
#     eps: float = 1e-9  # Epsilon for optimizer
#     betas: tuple = (0.8, 0.99)  # Betas for optimizer

#     # Checkpointing
#     save_every_epoch: int = 10  # Save checkpoint every N epochs
#     log_interval: int = 200  # Log every N steps
#     pretrain_g: str = ""  # Pretrained generator path
#     pretrain_d: str = ""  # Pretrained discriminator path

#     # Data loading
#     num_workers: int = 4
#     prefetch_factor: int = 8


def save_checkpoint(
    accelerator: Accelerator,
    net_g: torch.nn.Module,
    net_d: torch.nn.Module,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    model_dir: str,
):
    """Save checkpoint."""
    if accelerator.is_main_process:
        save_dir = Path(model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_g = accelerator.unwrap_model(net_g)
        unwrapped_d = accelerator.unwrap_model(net_d)

        torch.save(
            {
                "model": unwrapped_g.state_dict(),
                "optimizer": optim_g.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            save_dir / f"G_{global_step}.pth",
        )

        torch.save(
            {
                "model": unwrapped_d.state_dict(),
                "optimizer": optim_d.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            save_dir / f"D_{global_step}.pth",
        )

        logger.info(f"✓ Saved checkpoint at step {global_step}")


def load_pretrained(
    model: torch.nn.Module, path: str, accelerator: Accelerator
) -> None:
    """Load pretrained weights."""
    if not path or not os.path.exists(path):
        logger.warning(f"Pretrained path {path} does not exist. Skipping load.")
        return

    # accelerator.print(f"Loading pretrained from {path}")
    logger.info(f"Loading pretrained from {path}", main_process_only=True)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.load_state_dict(state_dict)


def run_train(
    # args: TrainArgs,
    train_filelist: str = "filelists/train.txt",
    model_dir: str = "logs/model",
    epochs: int = 20000,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    lr_decay: float = 0.999875,
    seed: int = 1234,
    sample_rate: int = 48000,
    hop_length: int = 480,
    win_length: int = 2048,
    max_text_len: int = 5000,
    min_text_len: int = 1,
    max_wav_value: float = 32768.0,
    filter_length: int = 2048,
    eps: float = 1e-9,
    betas: tuple = (0.8, 0.99),
    save_every_epoch: int = 10,
    log_interval: int = 200,
    pretrain_g: str = "",
    pretrain_d: str = "",
):
    """Main training function."""

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if default_config["train"]["fp16_run"] else "no",
        gradient_accumulation_steps=1,
    )

    set_seed(seed)

    # Create model directory
    if accelerator.is_main_process:
        os.makedirs(model_dir, exist_ok=True)

    # Dataset
    train_dataset = TextAudioLoaderMultiNSFsid(
        audiopaths_and_text=train_filelist,
        max_wav_value=max_wav_value,
        sampling_rate=sample_rate,
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        max_text_len=max_text_len,
        min_text_len=min_text_len,
    )

    logger.info(f"Training samples: {len(train_dataset)}", main_process_only=True)

    collate_fn = TextAudioCollateMultiNSFsid()

    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=8,
    )

    # Models
    m = default_config["model"]
    net_g = SynthesizerTrnMsNSFsid(
        spec_channels=filter_length // 2 + 1,
        segment_size=default_config["train"]["segment_size"] // hop_length,
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

    net_d = MultiPeriodDiscriminatorV2(use_spectral_norm=False, lrelu_slope=0.1)

    # Optimizers
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    # Schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay)

    # Prepare with accelerator
    net_g, net_d, optim_g, optim_d, train_loader, scheduler_g, scheduler_d = (
        accelerator.prepare(
            net_g, net_d, optim_g, optim_d, train_loader, scheduler_g, scheduler_d
        )
    )

    # Load pretrained
    if pretrain_g:
        load_pretrained(net_g, pretrain_g, accelerator)
    if pretrain_d:
        load_pretrained(net_d, pretrain_d, accelerator)
    # Training loop
    global_step = 0
    logger.info(f"Starting training for {epochs} epochs")
    for epoch in range(1, epochs + 1):
        net_g.train()
        net_d.train()

        progress_bar = tqdm(
            train_loader,
            disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch}/{epochs}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = batch

            # Forward generator
            with accelerator.autocast():
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (
                    net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                )

                # Compute mel
                mel = spec_to_mel_torch(
                    spec,
                    default_config["data"]["filter_length"],
                    default_config["data"]["n_mel_channels"],
                    default_config["data"]["sampling_rate"],
                    default_config["data"]["mel_fmin"],
                    default_config["data"]["mel_fmax"],
                )

                y_mel = slice_segments(
                    mel,
                    ids_slice,
                    default_config["train"]["segment_size"] // hop_length,
                )

                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    default_config["data"]["filter_length"],
                    default_config["data"]["n_mel_channels"],
                    default_config["data"]["sampling_rate"],
                    hop_length,
                    win_length,
                    default_config["data"]["mel_fmin"],
                    default_config["data"]["mel_fmax"],
                )

                if default_config["train"]["fp16_run"]:
                    y_hat_mel = y_hat_mel.half()

                wave = slice_segments(
                    wave,
                    ids_slice * hop_length,
                    default_config["train"]["segment_size"],
                )

                # Train Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
            optim_d.zero_grad()
            accelerator.backward(loss_disc)
            grad_norm_d = clip_grad_value_(net_d.parameters(), None)
            optim_d.step()

            with accelerator.autocast():
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)

                loss_mel = (
                    F.l1_loss(y_mel, y_hat_mel) * default_config["train"]["c_mel"]
                )
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * 1.0
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            optim_g.zero_grad()
            accelerator.backward(loss_gen_all)
            grad_norm_g = clip_grad_value_(net_g.parameters(), None)
            optim_g.step()
            global_step += 1

            # Logging
            if global_step % log_interval == 0 and accelerator.is_main_process:
                lr = optim_g.param_groups[0]["lr"]

                # Clamp extreme values for logging
                loss_mel_log = min(loss_mel.item(), 75)
                loss_kl_log = min(loss_kl.item(), 9)

                log_msg = (
                    f"Step {global_step} | Epoch {epoch} [{100.0 * batch_idx / len(train_loader):.0f}%] | "
                    f"LR: {lr:.6f} | "
                    f"D: {loss_disc.item():.3f} | "
                    f"G: {loss_gen.item():.3f} | "
                    f"FM: {loss_fm.item():.3f} | "
                    f"Mel: {loss_mel_log:.3f} | "
                    f"KL: {loss_kl_log:.3f}"
                )
                # accelerator.print(log_msg)
                logger.info(log_msg, main_process_only=True)

                # JSON metrics for parsing
                metrics = {
                    "step": global_step,
                    "epoch": epoch,
                    "lr": lr,
                    "loss_disc": loss_disc.item(),
                    "loss_gen": loss_gen.item(),
                    "loss_gen_all": loss_gen_all.item(),
                    "loss_fm": loss_fm.item(),
                    "loss_mel": loss_mel.item(),
                    "loss_kl": loss_kl.item(),
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                logger.info(f"METRICS: {json.dumps(metrics)}", main_process_only=True)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "D": f"{loss_disc.item():.2f}",
                    "G": f"{loss_gen.item():.2f}",
                    "Mel": f"{loss_mel.item():.2f}",
                }
            )

        # Step schedulers
        scheduler_g.step()
        scheduler_d.step()

        # Save checkpoint
        if epoch % save_every_epoch == 0:
            save_checkpoint(
                accelerator,
                net_g,
                net_d,
                optim_g,
                optim_d,
                epoch,
                global_step,
                model_dir,
            )

        accelerator.wait_for_everyone()

        # Final save
    if accelerator.is_main_process:
        accelerator.print("Training completed!")
        save_checkpoint(
            accelerator,
            net_g,
            net_d,
            optim_g,
            optim_d,
            epochs,
            global_step,
            model_dir,
        )


def main():
    # args = TrainArgs().parse_args()
    # run_train(args)
    from tap import tapify

    tapify(run_train)()


if __name__ == "__main__":
    main()
