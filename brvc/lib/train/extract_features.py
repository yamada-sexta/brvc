import os
import shutil
import sys
import logging
import traceback
from pathlib import Path
from typing import Literal
from omegaconf import DictConfig
from tqdm import tqdm
from tap import Tap

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from accelerate import Accelerator
from accelerate.logging import get_logger

from typing import TYPE_CHECKING

from lib.features.emb.hubert import load_hubert_model


logger = get_logger(__name__)


def read_wave(path: Path, normalize: bool = False) -> torch.Tensor:
    """Load a mono 16kHz waveform and optionally normalize."""
    wav, sr = sf.read(path)
    assert sr == 16000, f"Expected 16kHz, got {sr}Hz"
    feats = torch.from_numpy(wav).float()
    if feats.ndim == 2:
        feats = feats.mean(-1)
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats.view(1, -1)


def extract_feature(
    file: Path,
    out_file: Path,
    model,
    saved_cfg: DictConfig,
    accelerator: Accelerator,
    version: str,
):
    """Extract HuBERT features for a single file."""
    wav: torch.Tensor = read_wave(file, normalize=saved_cfg.task.normalize)
    padding_mask = torch.BoolTensor(wav.shape).fill_(False)

    inputs = {
        "source": wav.to(accelerator.device),
        "padding_mask": padding_mask.to(accelerator.device),
        "output_layer": 9 if version == "v1" else 12,
    }

    with torch.no_grad():
        logits = model.extract_features(**inputs)
        feats_tensor: torch.Tensor = (
            model.final_proj(logits[0]) if version == "v1" else logits[0]
        )

    feats = feats_tensor.squeeze(0).float().cpu().numpy()
    if not np.isnan(feats).any():
        np.save(out_file, feats, allow_pickle=False)
    else:
        logger.warning(f"{file.name} contains NaNs, skipped.")


def extract_features(
    exp_dir: Path,
    # output_dir: Path,
    version: Literal["v1", "v2"] = "v2",
    # is_half: bool = False,
    # model_path: Path = Path("assets/hubert/hubert_base.pt"),
):
    """
    Extract features from audio files using a pre-trained HuBERT model.
    Parameters
    ----------
    exp_dir : Path
        Path to the experiment directory containing wav files.
    output_dir : Path
        Path to save the extracted features.
    version : Literal["v1", "v2"], optional
        Version of HuBERT model to use ('v1' or 'v2'), by default "v2".
    is_half : bool, optional
        Whether to use half precision (fp16), by default False.
    model_path : str, optional
        Path to the HuBERT model checkpoint, by default "assets/hubert/hubert_base.pt".
    """
    # Initialize accelerator for multi-GPU, mixed precision, etc.
    accelerator = Accelerator()
    logger.info(f"Using device: {accelerator.device}")

    model, saved_cfg = load_hubert_model(accelerator)

    wav_dir = exp_dir / "1_16k_wavs"
    # out_dir = Path(output_dir)
    out_dir = exp_dir / f"3_feature768"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(wav_dir.glob("*.wav"))
    # wav_files = wav_files[args.i_part :: args.n_part]

    if not wav_files:
        logger.warning("No .wav files found to process.")
        return

    logger.info(f"Processing {len(wav_files)} files...")

    for file in tqdm(
        wav_files,
        desc="Extracting features",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    ):
        try:
            out_file = out_dir / file.with_suffix(".npy").name
            if out_file.exists():
                continue
            extract_feature(file, out_file, model, saved_cfg, accelerator, version)
        except Exception:
            logger.error(f"Error processing {file.name}:\n{traceback.format_exc()}")

    logger.info("[DONE] Feature extraction completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # args = FeatureExtractArgs().parse_args()
    # main(args)
    from tap import tapify

    tapify(extract_features)
