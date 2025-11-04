import logging
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)


def process_audio(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    silence_threshold: float = -30.0,  # dB
    silence_duration: float = 1.0,  # seconds
    chunk_duration: float = 4.0,  # seconds
    overlap: float = 0.3,  # seconds
    target_sr: int = 16000,
    recursive: bool = False,
    file_extensions: Sequence[str] = (".wav", ".mp3", ".flac", ".ogg"),
):
    """
    Process audio files in a directory using ffmpeg.
    Steps:
    1. Remove silence longer than `silence_duration`.
    2. Split into chunks of `chunk_duration` with `overlap`.
    3. Normalize volume.
    4. Resample to `target_sr`.
    5. Save to `output_dir`.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files: list[Path] = []
    for ext in file_extensions:
        audio_files.extend(
            input_dir.rglob(f"*{ext}") if recursive else input_dir.glob(f"*{ext}")
        )

    def process_file(file_path: Path) -> None:
        try:
            base_name = file_path.stem
            temp_no_silence = output_dir / f"{base_name}_nosil.wav"

            # Remove silence
            silence_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(file_path),
                "-af",
                f"silenceremove=stop_periods=-1:stop_duration={silence_duration}:stop_threshold={silence_threshold}dB",
                str(temp_no_silence),
            ]
            subprocess.run(
                silence_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Get duration of processed file
            probe_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(temp_no_silence),
            ]
            result = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            duration = float(result.stdout.strip())

            # Split into chunks with overlap and normalize
            start = 0.0
            chunk_idx = 0
            while start < duration:
                output_path = output_dir / f"{base_name}_chunk{chunk_idx}.wav"

                # Apply padding so every chunk is exactly `chunk_duration`
                split_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(temp_no_silence),
                    "-ss",
                    str(start),
                    "-t",
                    str(chunk_duration),
                    "-af",
                    f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample={target_sr},apad=pad_dur={chunk_duration}",
                    "-ar",
                    str(target_sr),
                    "-ac",
                    "1",  # mono output (optional, remove if stereo is needed)
                    str(output_path),
                ]
                subprocess.run(
                    split_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                start += chunk_duration - overlap
                chunk_idx += 1

            temp_no_silence.unlink(missing_ok=True)
            logger.info("Processed %s -> %s chunks", file_path, chunk_idx)
        except Exception as e:
            logger.error("Failed processing %s: %s", file_path, e)
            raise e

    for f in tqdm(audio_files, desc="Processing audio files"):
        process_file(f)


def test():
    """
    Sanity check
    """
    process_audio(
        "./models/sample_audio_input", "./models/output_audio", silence_duration=0.01
    )


if __name__ == "__main__":
    test()
