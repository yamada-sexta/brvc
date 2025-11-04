import os
from transformers import Wav2Vec2FeatureExtractor

OUTPUT_DIR = "assets/m"
os.makedirs(OUTPUT_DIR, exist_ok=True) 

# Initialize the Feature Extractor based on the standard HuBERT setup
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,            # Standard for raw audio
    sampling_rate=16000,       # Confirmed by 'sample_rate': 16000
    padding_value=0.0,
    do_normalize=True,         # Standard for pre-trained HuBERT features
    return_attention_mask=False,
)

# This creates the necessary preprocessor_config.json file
feature_extractor.save_pretrained(OUTPUT_DIR) 
print(f"Successfully saved preprocessor_config.json to {OUTPUT_DIR}")