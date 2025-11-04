from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
import torch
import torchcrepe

# from brvc.lib.features.pitch import pitch_predictor
from lib.features.pitch.pitch_predictor import PitchExtractor

class CRePE(PitchExtractor):
    def __init__(
        self, sample_rate: int, window_size: int, f0_min: int, f0_max: int, device: str
    ):
        super().__init__(sample_rate, window_size, f0_min, f0_max)
        self.device = device

    def extract_pitch(self, audio: NDArray[np.float32], p_len: int) -> NDArray[np.float32]:
        model = "full"
        batch_size = 512
        audio_tensor = torch.tensor(np.copy(audio))[None].float()
        f0, pd = torchcrepe.predict(
            audio_tensor,
            self.sr,
            self.window,
            self.f0_min,
            self.f0_max,
            model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        return f0[0].cpu().numpy()