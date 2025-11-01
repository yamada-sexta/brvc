#!/usr/bin/env python3
"""
Functional F0 extraction pipeline using CRePE (torchcrepe).
- CPU-only
- No CUDA, no half-precision
- Fully functional design
- Uses TAP for CLI, tqdm for progress, pathlib for paths
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import soundfile as sf
from tap import Tap
from tqdm import tqdm
from lib.features.pitch.crepe import CRePE
from lib.features.pitch.pitch_predictor import PitchPredictor
from lib.utils.audio import load_audio

logger = logging.getLogger(__name__)

def mel_scale(f0: np.ndarray) -> np.ndarray:
    """Convert linear frequency (Hz) to Mel scale."""
    return 1127 * np.log1p(f0 / 700)

def coarse_f0(
    f0: np.ndarray,
    f0_bin: int = 256,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
) -> np.ndarray:
    """Quantize continuous F0 into discrete coarse Mel bins."""
    f0_mel = mel_scale(f0)
    f0_mel_min = mel_scale(np.array([f0_min]))[0]
    f0_mel_max = mel_scale(np.array([f0_max]))[0]

    mask = f0_mel > 0
    f0_mel[mask] = (f0_mel[mask] - f0_mel_min) * (f0_bin - 2) / (
        f0_mel_max - f0_mel_min
    ) + 1

    f0_mel = np.clip(f0_mel, 1, f0_bin - 1)
    return np.rint(f0_mel).astype(int)


def extract_f0_pair(
    inp_path: Path,
    opt_path1: Path,
    opt_path2: Path,
    pitch_extractor: PitchPredictor,
) -> None:
    """Extract F0 and coarse F0, and save them as .npy."""
    if opt_path1.exists() and opt_path2.exists():
        return
    try:
        wav = load_audio(inp_path, sr=pitch_extractor.sampling_rate)
        f0 = pitch_extractor.compute_f0(wav)
        np.save(opt_path2, f0, allow_pickle=False)
        np.save(opt_path1, coarse_f0(f0), allow_pickle=False)
    except Exception as e:
        logger.error(f"F0 extraction failed for {inp_path}: {e}")
        logger.debug(traceback.format_exc())


def collect_audio_paths(exp_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """Collect all (input, coarse_output, fine_output) triplets."""
    inp_root = exp_dir / "1_16k_wavs"
    opt_root1 = exp_dir / "2a_f0"
    opt_root2 = exp_dir / "2b-f0nsf"

    opt_root1.mkdir(parents=True, exist_ok=True)
    opt_root2.mkdir(parents=True, exist_ok=True)

    return [
        (inp, opt_root1 / f"{inp.stem}.npy", opt_root2 / f"{inp.stem}.npy")
        for inp in sorted(inp_root.iterdir())
        if inp.suffix.lower() in {".wav", ".flac", ".mp3"} and "spec" not in inp.name
    ]

def partition_paths(paths: List, n_part: int, i_part: int) -> List:
    """Split list of paths into `n_part` partitions."""
    return paths[i_part::n_part]

# ---------------------------------------------------------------------
# Main functional runner
# ---------------------------------------------------------------------
def run_f0_extraction(
                      n_part: int,
                      i_part: int,
                      exp_dir: Path
                      ) -> None:
    """Main F0 extraction logic using CRePE."""
    paths = collect_audio_paths(exp_dir)
    subset = partition_paths(paths, n_part, i_part)

    logger.info(f"Processing {len(subset)} of {len(paths)} files using CRePE (CPU)")

    pitch_extractor = CRePE(device="cpu", sampling_rate=44100)

    for inp, opt1, opt2 in tqdm(subset, desc="Extracting F0", unit="file"):
        extract_f0_pair(inp, opt1, opt2, pitch_extractor)

    logger.info("✅ All F0 features extracted successfully.")


def main() -> None:
    # args = Args().parse_args()
    # run_f0_extraction(args)
    from tap import tapify
    tapify(run_f0_extraction)


if __name__ == "__main__":
    main()
