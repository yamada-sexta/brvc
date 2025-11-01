import os
import shutil
import sys
import logging
import traceback
from pathlib import Path
from tqdm import tqdm
from tap import Tap

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from accelerate import Accelerator
from accelerate.logging import get_logger
from huggingface_hub import hf_hub_download

class Args(Tap):
    """
    Feature extraction arguments
    """

    input_dir: str  # Path to input data directory containing wavs
    output_dir: str  # Path to save extracted features
    version: str = "v1"  # 'v1' or 'v2'
    device: str = "auto"  # 'cpu', 'cuda', 'mps', 'privateuseone', or 'auto'
    n_part: int = 1  # Total parts (for parallel jobs)
    i_part: int = 0  # Current part index
    i_gpu: int = 0  # GPU index if CUDA
    is_half: bool = False  # Use half precision (fp16)
    model_path: str = "assets/hubert/hubert_base.pt"  # HuBERT model path


logger = get_logger(__name__)

def read_wave(path: Path, normalize: bool = False) -> torch.Tensor:
    """Load a mono 16kHz waveform and optionally normalize."""
    wav, sr = sf.read(path)
    assert sr == 16000, f"Expected 16kHz, got {sr}Hz"
    feats = torch.from_numpy(wav).float()
    if feats.ndim == 2:
        feats = feats.mean(-1)
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats.view(1, -1)


def load_model(model_path: str, accelerator: Accelerator, version: str):
    """Load and prepare the HuBERT model."""
    if not os.path.exists(model_path):
        logging.info(f"{model_path} not found. Downloading from Hugging Face...")

        # Download hubert_base.pt from the VoiceConversionWebUI repo
        downloaded_model_path = hf_hub_download(
            repo_id="lj1995/VoiceConversionWebUI",
            filename="hubert_base.pt",
            repo_type="model",  # optional but good practice
        )

        logging.info(f"Downloaded model to {downloaded_model_path}")
        # Copy to local path for future use
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        shutil.copy(downloaded_model_path, model_path)
    import fairseq
    from fairseq.data.dictionary import Dictionary
    from fairseq import checkpoint_utils
    from torch.serialization import safe_globals

    with safe_globals([Dictionary]):
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [model_path]
        )

    model = models[0]
    model.eval()
    
    # Move to accelerator device, handle fp16 automatically
    model = accelerator.prepare(model)
    if accelerator.mixed_precision == "fp16":
        model = model.half()

    logging.info(f"Model loaded and prepared on device(s): {accelerator.device}")
    return model, saved_cfg


def extract_feature(
    file: Path,
    out_file: Path,
    model,
    saved_cfg,
    accelerator: Accelerator,
    version: str,
):
    """Extract HuBERT features for a single file."""
    feats = read_wave(file, normalize=saved_cfg.task.normalize)
    padding_mask = torch.BoolTensor(feats.shape).fill_(False)

    inputs = {
        "source": feats.to(accelerator.device),
        "padding_mask": padding_mask.to(accelerator.device),
        "output_layer": 9 if version == "v1" else 12,
    }

    with torch.no_grad():
        logits = model.extract_features(**inputs)
        feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

    feats = feats.squeeze(0).float().cpu().numpy()
    if not np.isnan(feats).any():
        np.save(out_file, feats, allow_pickle=False)
    else:
        logging.warning(f"{file.name} contains NaNs, skipped.")

def main(args: Args):
    # Initialize accelerator for multi-GPU, mixed precision, etc.
    accelerator = Accelerator(mixed_precision="fp16" if args.is_half else "no")
    logger.info(f"Using device: {accelerator.device}")

    model, saved_cfg = load_model(args.model_path, accelerator, args.version)

    wav_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(wav_dir.glob("*.wav"))
    wav_files = wav_files[args.i_part :: args.n_part]

    if not wav_files:
        logger.warning("No .wav files found to process.")
        return

    logger.info(f"Processing {len(wav_files)} files...")

    for file in tqdm(wav_files, desc="Extracting features", ncols=80, disable=not accelerator.is_local_main_process):
        try:
            out_file = out_dir / file.with_suffix(".npy").name
            if out_file.exists():
                continue
            extract_feature(file, out_file, model, saved_cfg, accelerator, args.version)
        except Exception:
            logger.error(f"Error processing {file.name}:\n{traceback.format_exc()}")

    logger.info("✅ Feature extraction completed successfully.")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
