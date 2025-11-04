from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
import torch
import torchcrepe

# from brvc.lib.features.pitch import pitch_predictor
from lib.features.pitch.pitch_predictor import PitchExtractor


class CRePE(PitchExtractor):
    def __init__(
        self,
        sample_rate: int,
        device: str,
        window_size: int = 160,
        f0_min: int = 50,
        f0_max: int = 1100,
    ):
        super().__init__()
        self.device = device

    def extract_pitch(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        model = "full"
        batch_size = 512
        audio_tensor = torch.tensor(np.copy(audio))[None].float()
        f0, pd = torchcrepe.predict(
            audio_tensor,
            sample_rate=self.sr,
            hop_length=self.window,
            fmin=self.f0_min,
            fmax=self.f0_max,
            model=model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        return f0[0].cpu().numpy()
