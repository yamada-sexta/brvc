from itertools import islice
import traceback
import logging
from typing import List, Optional
from gradio import Dataset
from typing_extensions import Literal
from tqdm import tqdm
import numpy as np
import librosa
from scipy import signal
from scipy.io import wavfile
from numpy.typing import NDArray
from typing import Union
from lib.train.config import GT_DIR, RESAMPLED_16K_DIR
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


# def process_file(
#     path: Path,
#     idx: int,
#     sr: int,
#     slicer: Slicer,
#     filter_coeffs: tuple[NDArray, NDArray],
#     gt_wavs_dir: Path,
#     wavs16k_dir: Path,
#     per: float,
#     overlap: float,
#     max_amp: float,
#     alpha: float,
# ) -> None:
#     """Process a single audio file."""
#     try:
#         b, a = filter_coeffs
#         audio = load_audio(file=path, resample_rate=sr)

#         if audio is None:
#             logger.error(f"Failed to load audio: {path}")
#             return

#         # audio: NDArray[np.float32] = signal.lfilter(b, a, audio)
#         # Use numpy's lfilter to avoid type issues
#         audio: NDArray[np.float32] = np.asarray(
#             signal.lfilter(b, a, audio), dtype=np.float32
#         )

#         idx1 = 0
#         for sliced_audio in slicer.slice(audio):
#             i = 0
#             while True:
#                 start = int(sr * (per - overlap) * i)
#                 i += 1
#                 if len(sliced_audio[start:]) > (per + overlap) * sr:
#                     tmp_audio = sliced_audio[start : start + int(per * sr)]
#                     norm_audio = normalize_audio(tmp_audio, max_amp, alpha)
#                     if norm_audio is not None:
#                         basename = f"{idx}_{idx1}.wav"
#                         save_audio(norm_audio, sr, gt_wavs_dir / basename)
#                         save_audio(norm_audio, sr, wavs16k_dir / basename, 16000)
#                     else:
#                         logger.warning(
#                             f"Skipping segment {idx}_{idx1} from file {path} due to amplitude issues."
#                         )
#                     idx1 += 1
#                 else:
#                     tmp_audio = sliced_audio[start:]
#                     norm_audio = normalize_audio(tmp_audio, max_amp, alpha)
#                     if norm_audio is not None:
#                         basename = f"{idx}_{idx1}.wav"
#                         save_audio(norm_audio, sr, gt_wavs_dir / basename)
#                         save_audio(norm_audio, sr, wavs16k_dir / basename, 16000)
#                     break

#         logger.debug(f"Processed: {path}")

#     except Exception:
#         logger.error(f"Failed to process {path}\n{traceback.format_exc()}")

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
    """Load an audio file and pass to process_audio_array."""
    try:
        audio = load_audio(file=path, resample_rate=sr)

        if audio is None:
            logger.error(f"Failed to load audio: {path}")
            return

        process_audio_array(
            audio_in=audio,
            identifier=str(path),
            idx=idx,
            sr=sr,
            slicer=slicer,
            filter_coeffs=filter_coeffs,
            gt_wavs_dir=gt_wavs_dir,
            wavs16k_dir=wavs16k_dir,
            per=per,
            overlap=overlap,
            max_amp=max_amp,
            alpha=alpha,
        )

    except Exception:
        # Catch errors from load_audio itself
        logger.error(f"Failed to process {path}\n{traceback.format_exc()}")

def process_audio_array(
    audio_in: NDArray[np.float32],
    identifier: str,  # For logging (e.g., filename or dataset ID)
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
    """Process a loaded audio array."""
    try:
        b, a = filter_coeffs
        # Use numpy's lfilter to avoid type issues
        audio: NDArray[np.float32] = np.asarray(
            signal.lfilter(b, a, audio_in), dtype=np.float32
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
                            f"Skipping segment {idx}_{idx1} from {identifier} due to amplitude issues."
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
        
        logger.debug(f"Processed: {identifier}")

    except Exception:
        logger.error(f"Failed to process {identifier}\n{traceback.format_exc()}")


def preprocess_dataset(
    audio_collection: Union[Path, Literal["phoneme_asr"]],
    exp_dir: Optional[Path] = None,
    sample_rate: int = 48000,
    per: float = 3.7,
    overlap: float = 0.3,
    max_amp: float = 0.9,
    alpha: float = 0.75,
    recursive: bool = False,
    max_sample_size: Optional[int] = None,
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
        exp_dir = Path("experiments") / (
            audio_collection
            if isinstance(audio_collection, str)
            else audio_collection.name
        )
        logger.info(f"No exp_dir provided. Using default: {exp_dir}")

    slicer = Slicer(
        sr=sample_rate,
        threshold=-42,
        min_length=1500,
        min_interval=400,
        hop_size=15,
        max_sil_kept=500,
    )
    res: tuple[NDArray, NDArray] = signal.butter(
        N=5, Wn=48, btype="high", fs=sample_rate
    )  # type: ignore
    if isinstance(res, tuple) and len(res) == 2:
        b, a = res
    else:
        raise ValueError("Unexpected result from signal.butter")

    gt_wavs_dir = exp_dir / GT_DIR
    wavs16k_dir = exp_dir / RESAMPLED_16K_DIR
    gt_wavs_dir.mkdir(parents=True, exist_ok=True)
    wavs16k_dir.mkdir(parents=True, exist_ok=True)
    if audio_collection == "phoneme_asr":
        logger.info(
            f"Loading Phoneme ASR dataset. Sampling {max_sample_size or 'all'} items."
        )
        try:
            from datasets import load_dataset, Audio
        except ImportError:
            logger.error("Please 'pip install datasets' to use Common Voice.")
            return

        cv_17 = load_dataset("mirfan899/phoneme_asr", "default", split="train")
        # print(cv_17)
        print(next(iter(cv_17)))
        
        # 
        
        counter = 0
        for item in tqdm(
            cv_17,
            dynamic_ncols=True,
            desc="Processing Phoneme ASR",
        ):
            audio_data = item["audio"]["array"]
            identifier = item["audio"].get("path", f"i{counter}")
            # Resample the audio to 48000Hz if needed
            original_sr = item["audio"].get("sampling_rate", 16000)
            if original_sr != sample_rate:
                audio_data = librosa.resample(
                    np.array(audio_data, dtype=np.float32),
                    orig_sr=original_sr,
                    target_sr=sample_rate,
                )
            process_audio_array(
                audio_in=audio_data.astype(np.float32),  # Ensure correct dtype
                identifier=identifier,
                idx=counter,
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
            counter += 1
            if max_sample_size and counter >= max_sample_size:
                break
        logger.info("Finished processing Phoneme ASR dataset.")
        return

    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    if recursive:
        files = [p for p in audio_collection.rglob("*") if p.suffix.lower() in audio_exts]
    else:
        files = [p for p in audio_collection.iterdir() if p.suffix.lower() in audio_exts]

    files = sorted(files)
    if max_sample_size is not None:
        files = list(islice(files, max_sample_size))
        logger.info(f"Limiting to first {max_sample_size} files for preprocessing.")

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

def preprocess_cli(
    audio_collection: str,
    exp_dir: Optional[Path] = None,
    sample_rate: int = 48000,
    per: float = 3.7,
    overlap: float = 0.3,
    max_amp: float = 0.9,
    alpha: float = 0.75,
    recursive: bool = False,
    max_sample_size: Optional[int] = None,
) -> None:
    """CLI entry point for preprocessing."""
    if audio_collection == "phoneme_asr":
        audio_collection_arg: Union[Path, Literal["phoneme_asr"]] = "phoneme_asr"
    else:
        audio_collection_arg = Path(audio_collection)

    preprocess_dataset(
        audio_collection=audio_collection_arg,
        exp_dir=exp_dir,
        sample_rate=sample_rate,
        per=per,
        overlap=overlap,
        max_amp=max_amp,
        alpha=alpha,
        recursive=recursive,
        max_sample_size=max_sample_size,
    )


if __name__ == "__main__":
    from tap import tapify

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    tapify(preprocess_cli)
