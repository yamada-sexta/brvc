from pathlib import Path
from typing import TYPE_CHECKING
from accelerate import Accelerator
from omegaconf import DictConfig

if TYPE_CHECKING:
    from fairseq.models.hubert.hubert import HubertModel
import logging
import os
import shutil
import transformers
from transformers import Wav2Vec2FeatureExtractor, HubertModel

logger = logging.getLogger(__name__)

repo_id: str = "lj1995/VoiceConversionWebUI"
def load_hubert_model(
    accelerator: Accelerator,
    # model_path: Path = Path("assets/hubert/hubert_base.pt"),
    model_dir: Path = Path("assets/hubert_hf"),
    fairseq_model_path: Path = Path("assets/hubert/hubert_base.pt"),
) -> tuple["HubertModel", DictConfig]:
    config_path = model_dir / "config.json"
    if config_path.exists():
        logger.info(f"Found existing converted HuBERT model at {model_dir}")
        model = HubertModel.from_pretrained(model_dir)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

    from huggingface_hub import hf_hub_download
    from fairseq.models.hubert.hubert import HubertModel

    """Load and prepare the HuBERT model."""
    if not model_path.exists():
        logger.info(f"{model_path} not found. Downloading from Hugging Face...")

        # Download hubert_base.pt from the VoiceConversionWebUI repo
        downloaded_model_path = hf_hub_download(
            repo_id="lj1995/VoiceConversionWebUI",
            filename="hubert_base.pt",
            repo_type="model",  # optional but good practice
        )

        logger.info(f"Downloaded model to {downloaded_model_path}")
        # Copy to local path for future use
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        shutil.copy(downloaded_model_path, model_path)
        logger.info(f"Copied model to {model_path}")
        # Remove the downloaded file
        os.remove(downloaded_model_path)
    import fairseq
    from fairseq.data.dictionary import Dictionary
    from fairseq import checkpoint_utils
    from torch.serialization import safe_globals

    models: list[HubertModel] = []
    with safe_globals([Dictionary]):
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [str(model_path)]
        )
        if saved_cfg is None:
            raise ValueError("Could not find model configuration.")

    model = models[0]
    model.eval()

    # Move to accelerator device, handle fp16 automatically
    model = accelerator.prepare(model)
    if accelerator.mixed_precision == "fp16":
        logger.info("Converting model to half precision (fp16).")
        model = model.half()

    logger.info(f"Model loaded and prepared on device(s): {accelerator.device}")
    return model, saved_cfg


def check_hubert_model_pt(model_path: Path = Path("assets/hubert/hubert_base.pt")) -> bool:
    import torch

    checkpoint = torch.load("model.pt", map_location="cpu")
    print(type(checkpoint))
    
if __name__ == "__main__":
    check_hubert_model_pt()