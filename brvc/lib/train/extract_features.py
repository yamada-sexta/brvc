import sys
import logging
import traceback
from pathlib import Path
from tqdm import tqdm
import torch
import soundfile as sf
from accelerate import Accelerator
from accelerate.logging import get_logger

from typing import TYPE_CHECKING

from lib.features.emb.hubert import get_hf_hubert_model
from safetensors.torch import save_file

from lib.train.config import HUBERT_DIR, RESAMPLED_16K_DIR


logger = get_logger(__name__)


if TYPE_CHECKING:
    from transformers import Wav2Vec2FeatureExtractor, HubertModel


@torch.no_grad()
def extract_feature(
    file: Path,
    out_file: Path,
    model: "HubertModel",
    extractor: "Wav2Vec2FeatureExtractor",
    accelerator: Accelerator,
):
    """Extract HuBERT features for a single file."""
    wav, sr = sf.read(file)
    assert sr == 16000, f"Expected 16kHz, got {sr}Hz"
    if wav.ndim == 2:
        wav = wav.mean(-1)  # Convert to mono
    # Make sure the waveform is 1D
    assert wav.ndim == 1, f"Expected 1D waveform, got {wav.ndim}D"

    inputs = extractor(wav, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    feats_tensor = outputs.last_hidden_state

    feats = feats_tensor.squeeze(0).float().cpu()
    if not torch.isnan(feats).any():
        save_file({"feats": feats}, out_file)
    else:
        logger.warning(f"{file.name} contains NaNs, skipped.")


@torch.no_grad()
def extract_features(
    exp_dir: Path,
    accelerator: Accelerator = Accelerator(),
):
    """Extract HuBERT features for all files in the dataset."""

    model, extractor = get_hf_hubert_model()
    model.eval()
    model.to(accelerator.device)

    wav_dir = exp_dir / RESAMPLED_16K_DIR
    out_dir = exp_dir / HUBERT_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(wav_dir.glob("*.wav"))

    if not wav_files:
        logger.warning("No .wav files found to process.")
        return

    logger.info(f"Processing {len(wav_files)} files...")
    for file in tqdm(
        wav_files,
        desc="Extracting features",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    ):
        try:
            out_file = out_dir / file.with_suffix(".safetensors").name
            if out_file.exists():
                # logger.info(f"Skipping {file.name}, already processed.")
                continue
            extract_feature(file, out_file, model, extractor, accelerator)
        except Exception:
            logger.error(f"Error processing {file.name}:\n{traceback.format_exc()}")
    
    accelerator.wait_for_everyone()
    
    logger.info("[DONE] Feature extraction completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    from tap import tapify

    tapify(extract_features)
