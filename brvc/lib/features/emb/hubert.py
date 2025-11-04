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
    model_path: Path = Path("assets/hubert/hubert_base.pt"),
) -> tuple["HubertModel", DictConfig]:
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


# def check_hubert_model_pt(
#     model_path: Path = Path("assets/hubert/hubert_base.pt"),
# ) -> None:
#     import torch

#     checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
#     print(type(checkpoint))
#     print(checkpoint.keys())
#     print(checkpoint["args"])
#     print(checkpoint["model"].keys())


# def convert_fairseq_to_hf(
#     fairseq_model_path: Path = Path("assets/hubert/hubert_base.pt"),
#     hf_model_dir: Path = Path("assets/hubert_hf"),
# ) -> None:
#     from fairseq.models.hubert.hubert import HubertModel as FairseqHubertModel

#     # Load Fairseq HuBERT model
#     fairseq_model, _ = load_hubert_model(
#         accelerator=Accelerator(cpu=True),
#         model_path=fairseq_model_path,
#     )

#     # Create Hugging Face HuBERT model from Fairseq model
#     hf_model = HubertModel.from_pretrained(
#         pretrained_model_name_or_path=None,
#         state_dict=fairseq_model.state_dict(),
#         config={
#             "hidden_size": fairseq_model.encoder.embed_dim,
#             "num_hidden_layers": fairseq_model.encoder.layers,
#             "num_attention_heads": fairseq_model.encoder.attention_heads,
#             "intermediate_size": fairseq_model.encoder.ffn_embed_dim,
#             "hidden_act": "gelu",
#             "layer_norm_eps": 1e-5,
#             "vocab_size": fairseq_model.dictionary.__len__(),
#             "max_position_embeddings": 514,
#         },
#     )

#     # Save the Hugging Face model
#     hf_model.save_pretrained(hf_model_dir)
#     feature_extractor = Wav2Vec2FeatureExtractor(
#         feature_size=1,
#         sampling_rate=16000,
#         padding_value=0.0,
#         do_normalize=True,
#         return_attention_mask=False,
#     )
#     feature_extractor.save_pretrained(hf_model_dir)

#     logger.info(f"Converted Fairseq HuBERT model saved to {hf_model_dir}")


# if __name__ == "__main__":
#     check_hubert_model_pt()
#     convert_fairseq_to_hf()
from transformers import HubertModel, Wav2Vec2FeatureExtractor
def get_hf_hubert_model():
    model = HubertModel.from_pretrained("assets/m")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("assets/m")
    return model, feature_extractor


# # The directory where your conversion script saved the files
# model_dir = "assets/m"

# # Load the Hugging Face HuBERT model
# hf_model = HubertModel.from_pretrained(model_dir)

# # Load the feature extractor (for preparing audio data)
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)

# print("Hugging Face HuBERT model and feature extractor loaded successfully!")