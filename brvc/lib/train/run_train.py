from pathlib import Path
from typing import Union, Optional

from lib.train.preprocess import preprocess_dataset
from lib.train.extract_f0 import extract_f0
from lib.train.extract_features import extract_features
from lib.train.train_model import train_model


def run_training(
    audio_dir: Path,
    exp_dir: Optional[Path] = None,
):
    if exp_dir is None:
        exp_dir = Path("experiments") / audio_dir.name

    exp_dir.mkdir(parents=True, exist_ok=True)

    preprocess_dataset(
        audio_dir=audio_dir,
        exp_dir=exp_dir,
        sample_rate=48000,
        per=10.0,
        overlap=1.0,
        max_amp=0.95,
        alpha=0.97,
    )

    extract_f0(
        exp_dir=exp_dir,
        sample_rate=48000,
    )

    extract_features(
        exp_dir=exp_dir,
    )
    
    train_files = []

    # What we want to have is a list of tuples (audio_path, spec_path, f0_path)
    wav_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / "3_feature768"
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"
    # Generate the list of files
    # Each row should be (audio_path, feature_path, f0_path, f0nsf_path, speaker_id)
    for wav_path in wav_dir.glob("*.wav"):
        feature_path = feature_dir / wav_path.name.replace(".wav", ".npy")
        f0_path = f0_dir / wav_path.name.replace(".wav", ".npy")
        f0nsf_path = f0nsf_dir / wav_path.name.replace(".wav", ".npy")
        speaker_id = "0"
        train_files.append((str(wav_path), str(feature_path), str(f0_path), str(f0nsf_path), speaker_id))

    train_model(
        train_files=train_files,
        exp_dir=exp_dir,
    )

    pass


def main():
    from tap import tapify

    tapify(run_training)


if __name__ == "__main__":
    main()
