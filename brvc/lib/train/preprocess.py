import traceback
import logging
from typing import List, Optional
from tqdm import tqdm
import numpy as np
import librosa
from scipy import signal
from scipy.io import wavfile
from numpy.typing import NDArray
from typing import Union
from lib.utils.audio import load_audio
from lib.utils.slicer import Slicer
from pathlib import Path
from numba import njit, jit

logger = logging.getLogger(__name__)


@njit
def create_muted_audio(
    sample_rate: int, length: Union[int, float] = 3
) -> NDArray[np.float32]:
    """Create a muted audio segment."""
    return np.zeros(int(sample_rate * length), dtype=np.float32)


@njit
def normalize_audio(
    audio: NDArray[np.float32], max_amp: float, alpha: float
) -> Optional[NDArray[np.float32]]:
    """Normalize audio amplitude."""
    tmp_max = np.abs(audio).max()
    if tmp_max > 2.5:
        # logger.warning(f"Skipping segment with extreme amplitude: {tmp_max}")
        return None
    return (audio / tmp_max * (max_amp * alpha)) + (1 - alpha) * audio


def save_audio(
    audio: Optional[NDArray[np.float32]],
    sr: int,
    path: Path,
    resample_sr: Optional[int] = None,
) -> None:
    """Save audio at original or resampled rate."""
    if audio is None:
        logger.warning(f"Skipping saving audio to {path} because it is None")
        return

    if resample_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=resample_sr)
        sr = resample_sr

    wavfile.write(str(path), sr, audio.astype(np.float32))


def process_file(
    path: Path,
    idx: int,
    sr: int,
    slicer: Slicer,
    filter_coeffs: tuple[NDArray, NDArray],
    gt_wavs_dir: Path,
    wavs16k_dir: Path,
    per: float,
    overlap: float,
    max_amp: float,
    alpha: float,
) -> None:
    """Process a single audio file."""
    try:
        b, a = filter_coeffs
        audio = load_audio(file=path, resample_rate=sr)

        if audio is None:
            logger.error(f"Failed to load audio: {path}")
            return

        # audio: NDArray[np.float32] = signal.lfilter(b, a, audio)
        # Use numpy's lfilter to avoid type issues
        audio: NDArray[np.float32] = np.asarray(
            signal.lfilter(b, a, audio), dtype=np.float32
        )

        idx1 = 0
        for sliced_audio in slicer.slice(audio):
            i = 0
            while True:
                start = int(sr * (per - overlap) * i)
                i += 1
                if len(sliced_audio[start:]) > (per + overlap) * sr:
                    tmp_audio = sliced_audio[start : start + int(per * sr)]
                    norm_audio = normalize_audio(tmp_audio, max_amp, alpha)
                    if norm_audio is not None:
                        basename = f"{idx}_{idx1}.wav"
                        save_audio(norm_audio, sr, gt_wavs_dir / basename)
                        save_audio(norm_audio, sr, wavs16k_dir / basename, 16000)
                    else:
                        logger.warning(
                            f"Skipping segment {idx}_{idx1} from file {path} due to amplitude issues."
                        )
                    idx1 += 1
                else:
                    tmp_audio = sliced_audio[start:]
                    norm_audio = normalize_audio(tmp_audio, max_amp, alpha)
                    if norm_audio is not None:
                        basename = f"{idx}_{idx1}.wav"
                        save_audio(norm_audio, sr, gt_wavs_dir / basename)
                        save_audio(norm_audio, sr, wavs16k_dir / basename, 16000)
                    break

        logger.debug(f"Processed: {path}")

    except Exception:
        logger.error(f"Failed to process {path}\n{traceback.format_exc()}")


def preprocess_dataset(
    audio_dir: Path,
    exp_dir: Optional[Path] = None,
    sample_rate: int = 48000,
    per: float = 3.7,
    overlap: float = 0.3,
    max_amp: float = 0.9,
    alpha: float = 0.75,
    recursive: bool = False,
    # log_level: str = "INFO",
) -> None:
    """Main preprocessing pipeline.

    Parameters
    ----------
    input_root : str
        Path to the input data root directory.
    output_root : str
        Path to the output data root directory.
    sample_rate : int, optional
        Input sample rate.
    per : float, optional
        Segment length in seconds
    overlap : float, optional
        Overlap in seconds.
    max_amp : float, optional
        Max amplitude for normalization.
    alpha : float, optional
        Mixing factor for normalization.
    recursive : bool, optional
        Whether to search audio files recursively.
    """
    logger.info("Starting preprocessing...")

    if exp_dir is None:
        exp_dir = Path("experiments") / audio_dir.name
        logger.info(f"No exp_dir provided. Using default: {exp_dir}")

    slicer = Slicer(
        sr=sample_rate,
        threshold=-42,
        min_length=1500,
        min_interval=400,
        hop_size=15,
        max_sil_kept=500,
    )
    res: tuple[NDArray, NDArray] = signal.butter(N=5, Wn=48, btype="high", fs=sample_rate)  # type: ignore
    if isinstance(res, tuple) and len(res) == 2:
        b, a = res
    else:
        raise ValueError("Unexpected result from signal.butter")

    gt_wavs_dir = exp_dir / "0_gt_wavs"
    wavs16k_dir = exp_dir / "1_16k_wavs"
    gt_wavs_dir.mkdir(parents=True, exist_ok=True)
    wavs16k_dir.mkdir(parents=True, exist_ok=True)

    audio_exts = {".wav", ".flac", ".mp3", ".ogg"}
    if recursive:
        files = [p for p in audio_dir.rglob("*") if p.suffix.lower() in audio_exts]
    else:
        files = [p for p in audio_dir.iterdir() if p.suffix.lower() in audio_exts]

    files = sorted(files)

    for idx, file in enumerate(
        tqdm(files, dynamic_ncols=True, desc="Processing audio files")
    ):
        process_file(
            path=file,
            idx=idx,
            sr=sample_rate,
            slicer=slicer,
            filter_coeffs=(b, a),
            gt_wavs_dir=gt_wavs_dir,
            wavs16k_dir=wavs16k_dir,
            per=per,
            overlap=overlap,
            max_amp=max_amp,
            alpha=alpha,
        )

    # Save a muted audio segment for padding or other uses
    muted = create_muted_audio(sample_rate, length=per)
    save_audio(muted, sample_rate, gt_wavs_dir / "muted.wav")
    save_audio(muted, sample_rate, wavs16k_dir / "muted.wav", resample_sr=16000)

    logger.info("Finished preprocessing!")


if __name__ == "__main__":
    from tap import tapify

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    tapify(preprocess_dataset)
