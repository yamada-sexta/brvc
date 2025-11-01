from pathlib import Path
from typing import Union, Optional

from lib.train.preprocess import preprocess_dataset
from lib.train.extract_f0 import extract_f0
from lib.train.extract_features import extract_features

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
        sample_rate=44100,
        per=10.0,
        overlap=1.0,
        max_amp=0.95,
        alpha=0.97,
    )
    
    extract_f0(
        exp_dir=exp_dir,
        sample_rate=44100,
    )
    
    pass


def main():
    from tap import tapify

    tapify(run_training)


if __name__ == "__main__":
    main()
