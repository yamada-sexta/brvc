from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
import torch
import torchcrepe

from .pitch_predictor import PitchPredictor

class CRePE(PitchPredictor):
    def __init__(
        self,
        hop_length=512,
        f0_min=50,
        f0_max=1100,
        sampling_rate=44100,
        device="cpu",
    ):
        if "privateuseone" in str(device):
            device = "cpu"
        super().__init__(
            hop_length,
            f0_min,
            f0_max,
            sampling_rate,
            device,
        )

    def compute_f0(
        self,
        wav: NDArray[np.float32],
        p_len: Optional[int] = None,
    ) -> NDArray[np.float32]:
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        # Keep original numpy array type for the signature, create a tensor
        # for torch/torchcrepe operations so types remain consistent.
        if torch.is_tensor(wav):
            wav_tensor = wav
        else:
            wav_tensor = torch.from_numpy(wav)
        # Pick a batch size that doesn't cause memory errors on your gpu
        batch_size = 512
        # Compute pitch using device 'device'
        f0, pd = torchcrepe.predict(
            wav_tensor.float().to(str(self.device)).unsqueeze(dim=0),
            self.sampling_rate,
            self.hop_length,
            self.f0_min,
            self.f0_max,
            batch_size=batch_size,
            device=str(self.device),
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        # Ensure p_len is an int when passed to _resize_f0
        if p_len is None:
            target_len = int(wav.shape[0] // self.hop_length)
        else:
            target_len = int(p_len)

        interp, _ = self._interpolate_f0(self._resize_f0(f0, target_len))
        return interp
