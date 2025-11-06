import json
import os
from pathlib import Path
import sys
from typing import List, Literal, Optional, Tuple, Union
from accelerate.utils import set_seed
from tqdm import tqdm
from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid
from accelerate.logging import get_logger

from lib.train.loss import discriminator_loss, feature_loss, generator_loss, kl_loss
from lib.train.utils.collect import TextAudioCollateMultiNSFsid
from lib.train.utils.dataset import TextAudioLoaderMultiNSFsid
from lib.train.utils.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from lib.utils.misc import clip_grad_value_
from lib.utils.slice import slice_segments
import shutil
import torch.nn.functional as F

logger = get_logger(__name__)
now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))
import torch
from accelerate import Accelerator

from torch.utils.data import DataLoader
from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid
from lib.modules.discriminators import MultiPeriodDiscriminatorV2
from lib.config.v2_config import ConfigV2, default_config
from safetensors.torch import load_file, save_file, save_model


def save_checkpoint(
    accelerator: Accelerator,
    net_g: SynthesizerTrnMsNSFsid,
    net_d: MultiPeriodDiscriminatorV2,
    optim_g: torch.optim.AdamW,
    optim_d: torch.optim.AdamW,
    epoch: int,
    global_step: int,
    model_dir: Path,
):
    """Save checkpoint."""
    if accelerator.is_main_process:
        save_dir = model_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_g: SynthesizerTrnMsNSFsid = accelerator.unwrap_model(net_g)
        unwrapped_d: MultiPeriodDiscriminatorV2 = accelerator.unwrap_model(net_d)

        # Save model weights as safetensors (only tensors allowed).
        g_safetensors_path = save_dir / f"G_{epoch}.safetensors"
        g_safetensors_latest = save_dir / f"G_latest.safetensors"
        d_safetensors_path = save_dir / f"D_{epoch}.safetensors"
        d_safetensors_latest = save_dir / f"D_latest.safetensors"
        optimizers_path = save_dir / f"O_{epoch}.pth"
        optim_latest_path = save_dir / f"O_latest.pth"

        save_model(
            unwrapped_g,
            str(g_safetensors_path),
        )
        
        save_model(
            unwrapped_d,
            str(d_safetensors_path),
        )

        torch.save(
            {
                "optimizer_g": optim_g.state_dict(),
                "optimizer_d": optim_d.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            optimizers_path,
        )
        # Also save latest COPY
        shutil.copy(g_safetensors_path, g_safetensors_latest, follow_symlinks=False)
        shutil.copy(d_safetensors_path, d_safetensors_latest, follow_symlinks=False)
        shutil.copy(optimizers_path, optim_latest_path, follow_symlinks=False)
        logger.info(f"[success] Saved checkpoint at step {global_step}")


def load_pretrained(
    model: torch.nn.Module, path: str, accelerator: Accelerator
) -> None:
    """Load pretrained weights."""
    if not path or not os.path.exists(path):
        logger.warning(f"Pretrained path {path} does not exist. Skipping load.")
        return

    # accelerator.print(f"Loading pretrained from {path}")
    logger.info(f"Loading pretrained from {path}", main_process_only=True)
    # Support both torch .pth-style checkpoints and safetensors files.
    unwrapped = accelerator.unwrap_model(model)
    if str(path).endswith(".safetensors"):
        # safetensors returns a dict[str, tensor]
        state_dict = load_file(path)
        # load_state_dict accepts the mapping directly
        unwrapped.load_state_dict(state_dict)
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        unwrapped.load_state_dict(state_dict)


def train_model(
    train_files: List[Tuple[Path, Path, Path]],
    cache_dir: Path,
    save_dir: Path,
    epochs: int = 200,
    batch_size: int = 1,
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
    save_every_epoch: Optional[int] = None,
    # log_interval: int = 200,
    pretrain_g: Union[Path, Literal["base", "last"], None] = "base",
    pretrain_d: Union[Path, Literal["base", "last"], None] = "base",
    opt_state: Optional[Path | Literal["last"]] = None,
    # is_half: bool = False,
    accelerator: Accelerator = Accelerator(),
):
    """Main training function."""
    set_seed(seed)

    # Create model directory
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    # Dataset
    train_dataset = TextAudioLoaderMultiNSFsid(
        audio_and_text_path=train_files,
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
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )

    # Models
    M = ConfigV2.Model
    net_g = SynthesizerTrnMsNSFsid(
        spec_channels=filter_length // 2 + 1,
        segment_size=ConfigV2.Train.segment_size // hop_length,
        inter_channels=M.inter_channels,
        hidden_channels=M.hidden_channels,
        filter_channels=M.filter_channels,
        n_heads=M.n_heads,
        n_layers=M.n_layers,
        kernel_size=M.kernel_size,
        p_dropout=M.p_dropout,
        resblock=M.resblock,
        resblock_kernel_sizes=M.resblock_kernel_sizes,
        resblock_dilation_sizes=M.resblock_dilation_sizes,
        upsample_rates=M.upsample_rates,
        upsample_initial_channel=M.upsample_initial_channel,
        upsample_kernel_sizes=M.upsample_kernel_sizes,
        spk_embed_dim=M.spk_embed_dim,
        gin_channels=M.gin_channels,
        sr=sample_rate,
        lrelu_slope=0.1,
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

    # Load pretrained if specified
    if pretrain_g == "last":
        if (save_dir / "G_latest.safetensors").exists():
            pretrain_g = save_dir / "G_latest.safetensors"
        else:
            logger.warning(
                f"Chosen to load last pretrained generator, but no checkpoint found at {save_dir / 'G_latest.safetensors'}", main_process_only=True
            )
    if pretrain_d == "last":
        if (save_dir / "D_latest.safetensors").exists():
            pretrain_d = save_dir / "D_latest.safetensors"
        else:
            logger.warning(
                f"Chosen to load last pretrained discriminator, but no checkpoint found at {save_dir / 'D_latest.safetensors'}", main_process_only=True
            )
    if opt_state == "last":
        if (save_dir / "O_latest.pth").exists():
            opt_state = save_dir / "O_latest.pth"
        else:
            logger.warning(
                f"Chosen to load last optimizer state, but no checkpoint found at {save_dir / 'O_latest.pth'}", main_process_only=True
            )
    if pretrain_g == "base":
        pretrain_g = Path("assets/pretrained_v2/f0G48k.pth")
        # Check if file exists
        if not pretrain_g.exists():
            from huggingface_hub import hf_hub_download

            logger.warning(
                f"Pretrained generator not found at {pretrain_g}. Downloading..."
            )
            # Download f0G48k.pth from the VoiceConversionWebUI repo
            # https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/pretrained_v2
            downloaded_model_path = hf_hub_download(
                repo_id="lj1995/VoiceConversionWebUI",
                filename="pretrained_v2/f0G48k.pth",
                repo_type="model",
            )
            logger.info(
                f"Downloaded model to {downloaded_model_path}", main_process_only=True
            )
            # Move to local path for future use
            os.makedirs(os.path.dirname(pretrain_g), exist_ok=True)
            # Copy to local path for future use
            shutil.copy(downloaded_model_path, pretrain_g)
            # Delete the downloaded file
            os.remove(downloaded_model_path)
            pretrain_g = pretrain_g
            logger.info(f"Moved model to {pretrain_g}", main_process_only=True)

    if pretrain_d == "base":
        pretrain_d = Path("assets/pretrained_v2/f0D48k.pth")
        # Check if file exists
        if not pretrain_d.exists():
            logger.warning(
                f"Pretrained discriminator not found at {pretrain_d}. Downloading..."
            )
            from huggingface_hub import hf_hub_download

            # Download f0D48k.pth from the VoiceConversionWebUI repo
            downloaded_model_path = hf_hub_download(
                repo_id="lj1995/VoiceConversionWebUI",
                filename="pretrained_v2/f0D48k.pth",
                repo_type="model",
            )
            logger.info(
                f"Downloaded model to {downloaded_model_path}", main_process_only=True
            )
            # Move to local path for future use
            os.makedirs(os.path.dirname(pretrain_d), exist_ok=True)
            shutil.copy(downloaded_model_path, pretrain_d)
            pretrain_d = pretrain_d
            logger.info(f"Copied model to {pretrain_d}", main_process_only=True)
            # Delete the downloaded file
            os.remove(downloaded_model_path)
    # Load pretrained
    if pretrain_g is not None:
        logger.info(
            f"Loading generator pretrained from {pretrain_g}", main_process_only=True
        )
        load_pretrained(net_g, str(pretrain_g), accelerator)
    else:
        logger.info(
            "No pretrained generator specified, training from scratch.",
            main_process_only=True,
        )
    if pretrain_d is not None:
        logger.info(
            f"Loading discriminator pretrained from {pretrain_d}",
            main_process_only=True,
        )
        load_pretrained(net_d, str(pretrain_d), accelerator)
    else:
        logger.info(
            "No pretrained discriminator specified, training from scratch.",
            main_process_only=True,
        )
    if opt_state is not None:
        logger.info(f"Loading optimizer state from {opt_state}", main_process_only=True)
        # Load optimizer states
        ckpt = torch.load(opt_state, map_location="cpu")
        optim_g.load_state_dict(ckpt["optimizer_g"])
        optim_d.load_state_dict(ckpt["optimizer_d"])
        logger.info(
            f"Loaded optimizer states from epoch {ckpt.get('epoch', 'N/A')}, step {ckpt.get('global_step', 'N/A')}",
            main_process_only=True,
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

    # Training loop
    global_step = 0
    logger.info(f"Starting training for {epochs} epochs")
    epoch_bar = tqdm(
        range(1, epochs + 1),
        disable=not accelerator.is_main_process,
        desc=f"Overall",
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        net_g.train()
        net_d.train()

        progress_bar = tqdm(
            train_loader,
            disable=not accelerator.is_main_process,
            desc=f"Epoch {epoch}/{epochs}",
            leave=False,
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
        if save_every_epoch is not None and epoch % save_every_epoch == 0:
            logger.info(f"Saving checkpoint for epoch {epoch}")
            save_checkpoint(
                accelerator,
                net_g,
                net_d,
                optim_g,
                optim_d,
                epoch,
                global_step,
                save_dir,
            )

        accelerator.wait_for_everyone()

        # Final save
    if accelerator.is_main_process:
        logger.info("Training completed!")
        save_checkpoint(
            accelerator,
            net_g,
            net_d,
            optim_g,
            optim_d,
            epochs,
            global_step,
            save_dir,
        )


def main():
    from tap import tapify

    # setup logging
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    tapify(train_model)


if __name__ == "__main__":
    main()
