from pathlib import Path
from typing import Literal, Union, Optional, TYPE_CHECKING
from accelerate import Accelerator

import logging

from lib.train.config import F0_DIR, GT_DIR, HUBERT_DIR, ONLINE_DATASETS

logger = logging.getLogger(__name__)

def run_training_cli(
    dataset: str,
    exp_dir: Optional[Path] = None,
    save_every: Optional[int] = None,
    epochs: int = 200,
    load_pretrain: Union[Literal["last", "base"], None] = "base",
):
    """
    Run the training process with command-line arguments.
    Args:
        dataset (str | Path): Path to the dataset or name of the online dataset (e.g. "ljspeech").
        exp_dir (Optional[Path]): Directory to save experiment outputs.
        save_every (Optional[int]): Save model every N epochs. If None, only save at the end.
        epochs (int): Number of training epochs.
        load_pretrain (Union[Literal["last", "base"], None]): Pretrained model to load.
    """
    if dataset not in ONLINE_DATASETS:
        dataset = Path(dataset)
    run_training(
        dataset=dataset,
        exp_dir=exp_dir,
        save_every_epoch=save_every,
        epochs=epochs,
        load_pretrain=load_pretrain,
    )


def run_training(
    dataset: Union[str, Path],
    exp_dir: Optional[Path] = None,
    save_every_epoch: Optional[int] = None,
    epochs: int = 200,
    load_pretrain: Union[Literal["last", "base"], None] = "base",
    accelerator: Accelerator = Accelerator(),
):
    dataset_name = dataset if isinstance(dataset, str) else dataset.name
    if isinstance(dataset, Path) and not dataset.exists():
        raise FileNotFoundError(f"Audio directory {dataset} does not exist.")
    if isinstance(dataset, str) and dataset not in ONLINE_DATASETS:
        raise ValueError(f"Unknown online dataset: {dataset}. Supported: {ONLINE_DATASETS}")
    from lib.train.preprocess import preprocess_dataset
    from lib.train.extract_f0 import extract_f0
    from lib.train.extract_features import extract_features
    from lib.train.train_model import train_model

    if exp_dir is None:
        exp_dir = Path("experiments") / dataset_name
        logger.info(f"No exp_dir provided. Using default: {exp_dir}")

    exp_dir.mkdir(parents=True, exist_ok=True)

    preprocess_dataset(
        dataset=dataset,
        exp_dir=exp_dir,
        sample_rate=48000,
        per=10.0,
        overlap=1.0,
        max_amp=0.95,
        alpha=0.97,
    )

    extract_f0(
        exp_dir=exp_dir,
        # sample_rate=48000,
        accelerator=accelerator,
    )

    extract_features(
        exp_dir=exp_dir,
        accelerator=accelerator,
    )

    train_files: list[tuple[Path, Path, Path]] = []

    # What we want to have is a list of tuples (audio_path, spec_path, f0_path)
    wav_dir = exp_dir / GT_DIR
    feature_dir = exp_dir / HUBERT_DIR
    f0_dir = exp_dir / F0_DIR
    # f0nsf_dir = exp_dir / "2b-f0nsf"
    # Generate the list of files
    # Each row should be (audio_path, feature_path, f0_path, f0nsf_path, speaker_id)
    for wav_path in wav_dir.glob("*.wav"):
        feature_path = feature_dir / wav_path.name.replace(".wav", ".safetensors")
        f0_path = f0_dir / wav_path.name.replace(".wav", ".safetensors")
        train_files.append(
            (
                (wav_path),
                (feature_path),
                f0_path,
                # str(f0nsf_path),
                # speaker_id,
            )
        )

    train_model(
        train_files=train_files,
        exp_dir=exp_dir,
        epochs=epochs,
        save_every_epoch=save_every_epoch,
        pretrain_d=load_pretrain,
        pretrain_g=load_pretrain,
        opt_state=None if load_pretrain != "last" else "last",
        accelerator=accelerator,
    )

    pass


def main():
    from tap import tapify
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Starting training process...")
    from accelerate import Accelerator

    tapify(run_training_cli)


if __name__ == "__main__":
    main()
