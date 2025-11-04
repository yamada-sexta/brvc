import traceback
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from lib.features.pitch.crepe import CRePE
from lib.features.pitch.pitch_predictor import PitchExtractor
from lib.utils.audio import load_audio
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset

logger = get_logger(__name__)

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
    pitch_extractor: PitchExtractor,
) -> None:
    """Extract F0 and coarse F0, and save them as .npy."""
    if opt_path1.exists() and opt_path2.exists():
        return
    try:
        wav = load_audio(inp_path, resample_rate=pitch_extractor.sr)
        f0 = pitch_extractor.extract_pitch(wav)
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


class AudioDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


def extract_f0(
    exp_dir: Path, 
    sample_rate: int = 48000, 
    accelerator: Accelerator = Accelerator()
) -> None:
    # accelerator = Accelerator()
    device = accelerator.device
    # logger.info(f"Using device: {device}", main_process_only=True)

    paths = collect_audio_paths(exp_dir)
    dataset = AudioDataset(paths)
    dataloader = DataLoader(dataset, batch_size=None)
    dataloader = accelerator.prepare(dataloader)

    pitch_extractor = CRePE(device=device, sample_rate=sample_rate)

    logger.info(
        f"Processing {len(paths)} files using CRePE ({device})", main_process_only=True
    )

    for inp, opt1, opt2 in tqdm(
        dataloader,
        disable=not accelerator.is_main_process,
        desc="Extracting F0",
        unit="file",
        dynamic_ncols=True,
    ):
        extract_f0_pair(inp, opt1, opt2, pitch_extractor)

    accelerator.wait_for_everyone()
    logger.info("All F0 features extracted successfully.", main_process_only=True)


def main() -> None:
    from tap import tapify
    import logging

    logging.basicConfig(level=logging.INFO)

    tapify(extract_f0)


if __name__ == "__main__":
    main()
