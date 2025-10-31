import os
import traceback
import logging
from typing import List, Optional
from tap import Tap
from tqdm import tqdm
import numpy as np
import librosa
from scipy import signal
from scipy.io import wavfile
from numpy.typing import NDArray

from tap import Tap

from lib.utils.audio import load_audio
from lib.utils.slicer import Slicer

logger = logging.getLogger(__name__)


class Args(Tap):
    """
    Preprocessing arguments
    """

    input_root: str  # Path to the input data root directory
    output_root: str  # Path to the output data root directory
    sample_rate: int = 48000  # Input sample rate
    per: float = 3.7  # Segment length in seconds
    overlap: float = 0.3  # Overlap in seconds
    max_amp: float = 0.9  # Max amplitude for normalization
    alpha: float = 0.75  # Mixing factor for normalization


def init_dirs(output_root: str) -> tuple[str, str]:
    """Create required output directories."""
    gt_wavs_dir = os.path.join(output_root, "0_gt_wavs")
    wavs16k_dir = os.path.join(output_root, "1_16k_wavs")
    os.makedirs(gt_wavs_dir, exist_ok=True)
    os.makedirs(wavs16k_dir, exist_ok=True)
    return gt_wavs_dir, wavs16k_dir


def normalize_audio(audio: NDArray, max_amp: float, alpha: float) -> Optional[NDArray]:
    """Normalize audio amplitude."""
    tmp_max = np.abs(audio).max()
    if tmp_max > 2.5:
        logger.warning(f"Skipping segment with extreme amplitude: {tmp_max}")
        return None
    return (audio / tmp_max * (max_amp * alpha)) + (1 - alpha) * audio


def save_audio(
    audio: Optional[NDArray], sr: int, path: str, resample_sr: Optional[int] = None
) -> None:
    """Save audio at original or resampled rate."""
    if audio is None:
        logger.warning(f"Skipping saving audio to {path} because it is None")
        return

    if resample_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=resample_sr)
        sr = resample_sr
        # wavfile.write(path, resample_sr, audio.astype(np.float32))
    # else:
        # wavfile.write(path, sr, audio.astype(np.float32))
    wavfile.write(path, sr, audio.astype(np.float32))

def process_file(
    path: str,
    idx: int,
    sr: int,
    slicer: Slicer,
    filter_coeffs: tuple[NDArray, NDArray],
    gt_wavs_dir: str,
    wavs16k_dir: str,
    per: float,
    overlap: float,
    max_amp: float,
    alpha: float,
) -> None:
    """Process a single audio file."""
    try:
        b, a = filter_coeffs
        audio = load_audio(file=path, sr=sr)

        if audio is None:
            logging.error(f"Failed to load audio: {path}")
            return

        audio: NDArray = signal.lfilter(b, a, audio)

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
                        save_audio(norm_audio, sr, os.path.join(gt_wavs_dir, basename))
                        save_audio(
                            norm_audio, sr, os.path.join(wavs16k_dir, basename), 16000
                        )
                    idx1 += 1
                else:
                    tmp_audio = sliced_audio[start:]
                    norm_audio = normalize_audio(tmp_audio, max_amp, alpha)
                    if norm_audio is not None:
                        basename = f"{idx}_{idx1}.wav"
                        save_audio(norm_audio, sr, os.path.join(gt_wavs_dir, basename))
                        save_audio(
                            norm_audio, sr, os.path.join(wavs16k_dir, basename), 16000
                        )
                    break

        logging.info(f"Processed: {path}")

    except Exception:
        logging.error(f"Failed to process {path}\n{traceback.format_exc()}")


def preprocess_dataset(args: Args) -> None:
    """Main preprocessing pipeline."""
    logging.info("Starting preprocessing...")

    slicer = Slicer(
        sr=args.sample_rate,
        threshold=-42,
        min_length=1500,
        min_interval=400,
        hop_size=15,
        max_sil_kept=500,
    )
    res = signal.butter(N=5, Wn=48, btype="high", fs=args.sample_rate)
    if isinstance(res, tuple) and len(res) == 2:
        b, a = res
    else:
        raise ValueError("Unexpected result from signal.butter")
    
    gt_wavs_dir, wavs16k_dir = init_dirs(args.output_root)
    files = sorted(os.listdir(args.input_root))

    for idx, fname in enumerate(tqdm(files, desc="Processing audio files")):
        path = os.path.join(args.input_root, fname)
        process_file(
            path=path,
            idx=idx,
            sr=args.sample_rate,
            slicer=slicer,
            filter_coeffs=(b, a),
            gt_wavs_dir=gt_wavs_dir,
            wavs16k_dir=wavs16k_dir,
            per=args.per,
            overlap=args.overlap,
            max_amp=args.max_amp,
            alpha=args.alpha,
        )

    logging.info("Finished preprocessing!")


if __name__ == "__main__":
    args = Args().parse_args()
    preprocess_dataset(args)
