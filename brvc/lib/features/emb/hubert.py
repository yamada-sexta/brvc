from pathlib import Path
from typing import TYPE_CHECKING
from accelerate import Accelerator
import logging
from transformers import AutoProcessor, Wav2Vec2FeatureExtractor, HubertModel
logger = logging.getLogger(__name__)
def get_hf_hubert_model(
    accelerator: Accelerator,
) -> tuple[HubertModel, Wav2Vec2FeatureExtractor]:
    model_name = "facebook/hubert-base-ls960"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    # feature_extractor.to(accelerator.device)
    model.eval()
    model.to(accelerator.device)

    return model, processor
