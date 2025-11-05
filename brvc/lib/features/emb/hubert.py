from pathlib import Path
from typing import TYPE_CHECKING
from accelerate import Accelerator
from omegaconf import DictConfig

if TYPE_CHECKING:
    from fairseq.models.hubert.hubert import HubertModel
import logging
import os
import shutil
from transformers import Wav2Vec2FeatureExtractor, HubertModel

logger = logging.getLogger(__name__)

repo_id: str = "lj1995/VoiceConversionWebUI"
hf_model_path = Path("assets/hf/hubert_base")


def download_rvc_hubert():
    from huggingface_hub import hf_hub_download

    model_path = Path("assets/hubert/hubert_base.pt")
    if not model_path.exists():
        logger.info(f"{model_path} not found. Downloading model from Hugging Face...")
        downloaded_model_path = hf_hub_download(
            repo_id=repo_id,
            filename="hubert_base.pt",
            local_dir=model_path.parent,
            local_dir_use_symlinks=False,
            repo_type="model",  # optional but good practice
        )
        # Copy to local path for future use
        logger.info(f"Downloaded model to {model_path}")
        # Copy to local path for future use
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        shutil.copy(downloaded_model_path, model_path)
        logger.info(f"Copied model to {model_path}")
        # Remove the downloaded file
        os.remove(downloaded_model_path)
        logger.info(f"Model downloaded and saved to {model_path}")
    else:
        logger.info(f"Model already exists at {model_path}, skipping download.")


def convert_fairseq_to_hf_hubert():
    from lib.features.emb.convert import convert_hubert_checkpoint

    fairseq_model_path = "assets/hubert/hubert_base.pt"
    if not Path(hf_model_path).exists():
        logger.info("Converting Fairseq HuBERT model to Hugging Face format...")
        convert_hubert_checkpoint(
            checkpoint_path=fairseq_model_path,
            pytorch_dump_folder_path=str(hf_model_path),
        )
        logger.info(f"Converted model saved to {hf_model_path}")
    else:
        logger.info(
            f"Hugging Face HuBERT model already exists at {hf_model_path}, skipping conversion."
        )


from transformers import HubertModel, Wav2Vec2FeatureExtractor


def get_hf_hubert_model(
    accelerator: Accelerator,
) -> tuple[HubertModel, Wav2Vec2FeatureExtractor]:
    # Check if model exists, if not download and convert
    if not (
        hf_model_path.exists()
        and (hf_model_path / "config.json").exists()
        and (hf_model_path / "model.safetensors").exists()
        and (hf_model_path / "preprocessor_config.json").exists()
    ):
        download_rvc_hubert()
        convert_fairseq_to_hf_hubert()

    model = HubertModel.from_pretrained("assets/hf/hubert_base")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "assets/hf/hubert_base"
    )
    model.to(accelerator.device)
    # feature_extractor.to(accelerator.device)
    model.eval()

    return model, feature_extractor
