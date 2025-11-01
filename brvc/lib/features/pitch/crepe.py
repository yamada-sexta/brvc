from typing import Optional
import numpy as np
from numpy.typing import NDArray
import torch
import torchcrepe

from brvc.lib.features.pitch import pitch_predictor

class CRePE(pitch_predictor.PitchPredictor):
    def __init__(
        self,
        hop_length=512,
        f0_min=50,
        f0_max=1100,
        sample_rate=44100,
        device="cpu",
    ):
        if "privateuseone" in str(device):
            device = "cpu"
        super().__init__(
            hop_length,
            f0_min,
            f0_max,
            sample_rate,
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


def test():
    import soundfile as sf

    # Create a test sine wave file at 440 Hz for 1 second
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    # Use the generated array directly instead of writing/reading a file
    wav = sine_wave
    # Convert to float32 if necessary
    wav = wav.astype(np.float32)
    pitch_extractor = CRePE(device="cpu")
    f0 = pitch_extractor.compute_f0(wav)
    print(f0)


if __name__ == "__main__":
    test()
