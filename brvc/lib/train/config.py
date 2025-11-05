from typing import Literal

GT_DIR = "ft_gt"
RESAMPLED_16K_DIR = "ft_16k_wavs"
F0_DIR = "ft_f0"
HUBERT_DIR = "ft_hubert"

ONLINE_DATASET_TYPE = Literal["ljspeech", "phoneme_asr"]
ONLINE_DATASETS: tuple[ONLINE_DATASET_TYPE] = (
    "ljspeech",
    "phoneme_asr",
)
