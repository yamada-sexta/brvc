import traceback
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from tqdm import tqdm
from lib.features.pitch.crepe import CRePE
from lib.features.pitch.pitch_predictor import PitchExtractor
from lib.train.config import F0_DIR, RESAMPLED_16K_DIR
from lib.utils.audio import load_audio
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset
from numba import njit
from safetensors.torch import save_file

logger = get_logger(__name__)


@njit
def mel_scale(f0: np.ndarray) -> np.ndarray:
    """Convert linear frequency (Hz) to Mel scale."""
    return 1127 * np.log1p(f0 / 700)


@njit
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
    return np.rint(f0_mel).astype(np.int32)


@torch.no_grad()
def extract_f0_pair(
    inp_path: Path,
    output_path: Path,
    pitch_extractor: PitchExtractor,
) -> None:
    """Extract F0 and coarse F0, and save them."""
    if output_path.exists():
        logger.debug(f"F0 file {output_path} already exists, skipping.")
        return
    try:
        wav = load_audio(inp_path, resample_rate=pitch_extractor.sr)
        f0 = pitch_extractor.extract_pitch(wav)
        save_file(
            {"f0": torch.from_numpy(f0), "coarse_f0": torch.from_numpy(coarse_f0(f0))},
            output_path,
        )
    except Exception as e:
        logger.error(f"F0 extraction failed for {inp_path}: {e}")
        logger.debug(traceback.format_exc())


def collect_audio_paths(exp_dir: Path) -> List[Tuple[Path, Path]]:
    """Collect all (input, coarse_output, fine_output) triplets."""
    in_dir = exp_dir / RESAMPLED_16K_DIR
    out_dir = exp_dir / F0_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    res: List[Tuple[Path, Path]] = []

    for f in sorted(in_dir.iterdir()):
        if f.suffix.lower() == ".wav":
            out_path = out_dir / f"{f.stem}.safetensors"
            if not out_path.exists():
                res.append((f, out_path))
                logger.debug(f"Will process F0 for {f.name}")
    return res


class AudioDataset(Dataset):
    def __init__(self, paths: List[Tuple[Path, Path]]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx) -> Tuple[Path, Path]:
        return self.paths[idx]


@torch.no_grad()
def extract_f0(exp_dir: Path, accelerator: Accelerator = Accelerator()) -> None:
    device = accelerator.device
    paths = collect_audio_paths(exp_dir)
    dataset = AudioDataset(paths)
    dataloader = DataLoader(dataset, batch_size=None)
    dataloader = accelerator.prepare(dataloader)

    from lib.features.pitch.swift import Swift

    pitch_extractor = Swift()
    logger.info(
        f"Processing {len(paths)} files using Swift ({device})", main_process_only=True
    )

    for i, o in tqdm(
        dataloader,
        disable=not accelerator.is_main_process,
        desc="Extracting F0",
        unit="file",
        dynamic_ncols=True,
    ):
        extract_f0_pair(i, o, pitch_extractor)

    accelerator.wait_for_everyone()
    logger.info("All F0 features extracted successfully.", main_process_only=True)


def main() -> None:
    from tap import tapify
    import logging

    logging.basicConfig(level=logging.INFO)
    tapify(extract_f0)


if __name__ == "__main__":
    main()
