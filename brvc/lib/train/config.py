from typing import Literal

GT_DIR = "ft_gt"
RESAMPLED_16K_DIR = "ft_16k_wavs"
F0_DIR = "ft_f0"
HUBERT_DIR = "ft_hubert"

ONLINE_DATASETS = (
    "ljspeech",
    "phoneme_asr",
)

ONLINE_DATASET_TYPE = Literal["ljspeech", "phoneme_asr"]