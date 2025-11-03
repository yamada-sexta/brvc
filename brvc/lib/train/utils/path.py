from pathlib import Path
from typing import Union
from scipy.io.wavfile import read
import torch
import numpy as np


def load_filepaths_and_text(
    filename: Union[str, Path], split="|"
) -> list[tuple[str, str, str, str, str]]:
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line_to_tuple(line, split) for line in f]
        return filepaths_and_text

def line_to_tuple(line: str, split="|") -> tuple[str, str, str, str, str]:
    parts = line.strip().split(split)
    if len(parts) != 5:
        raise ValueError(
            f"Line must contain exactly 5 fields separated by '{split}'."
        )
    return parts[0], parts[1], parts[2], parts[3], parts[4]


def load_wav_to_torch(full_path: Union[str, Path]) -> tuple[torch.FloatTensor, int]:
    full_path = str(full_path)
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), int(sampling_rate)
