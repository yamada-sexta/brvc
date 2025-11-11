from pathlib import Path
from typing import Union
from scipy.io.wavfile import read
import torch
import numpy as np

def load_wav_to_torch(full_path: Union[str, Path]) -> tuple[torch.FloatTensor, int]:
    full_path = str(full_path)
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), int(sampling_rate)
