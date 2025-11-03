from typing import Optional, Union

import torch
import numpy as np
import abc
from numpy.typing import NDArray

def mel_scale(f0: np.ndarray) -> np.ndarray:
    """Convert linear frequency (Hz) to Mel scale."""
    return 1127 * np.log1p(f0 / 700)

def coarse_f0(
    f0: NDArray[np.float32],
    f0_bin: int = 256,
    f0_min: float = 50.0,
    f0_max: float = 1100.0,
) -> NDArray[np.int32]:
    """Quantize continuous F0 into discrete coarse Mel bins."""
    f0_mel = mel_scale(f0)
    f0_mel_min = mel_scale(np.array([f0_min]))[0]
    f0_mel_max = mel_scale(np.array([f0_max]))[0]

    mask = f0_mel > 0
    f0_mel[mask] = (f0_mel[mask] - f0_mel_min) * (f0_bin - 2) / (
        f0_mel_max - f0_mel_min
    ) + 1

    f0_mel = np.clip(f0_mel, 1, f0_bin - 1)
    return np.rint(f0_mel).astype(int)



class PitchPredictor(abc.ABC):
    """
    Abstract base class for frame-wise pitch (f0) prediction.

    Subclasses must implement `compute_f0` which takes a waveform and returns a
    1-D numpy array of f0 values (Hz) for each frame. Unvoiced frames should be
    represented as 0.
    """

    device: torch.device

    def __init__(
        self,
        hop_length: int = 512,
        f0_min: int = 50,
        f0_max: int = 1100,
        sampling_rate: int = 48000,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    # Abstract method
    @abc.abstractmethod
    @torch.no_grad()
    def compute_f0(
        self,
        wav: NDArray[np.float32],
        p_len: Optional[int] = None,
    ) -> NDArray[np.float32]:
        """
        Compute the f0 contour from the input waveform.
        Should be implemented by subclasses."""

    def _interpolate_f0(self, f0: np.ndarray):
        """
        Interpolate the unvoiced regions of the f0 contour using linear interpolation.
        """

        data = np.reshape(f0, (f0.size, 1))

        vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
        vuv_vector[data > 0.0] = 1.0
        vuv_vector[data <= 0.0] = 0.0

        ip_data = data

        frame_number = data.size
        last_value = 0.0
        for i in range(frame_number):
            if data[i] <= 0.0:
                j = i + 1
                for j in range(i + 1, frame_number):
                    if data[j] > 0.0:
                        break
                if j < frame_number - 1:
                    if last_value > 0.0:
                        step = (data[j] - data[i - 1]) / float(j - i)
                        for k in range(i, j):
                            ip_data[k] = data[i - 1] + step * (k - i + 1)
                    else:
                        for k in range(i, j):
                            ip_data[k] = data[j]
                else:
                    for k in range(i, frame_number):
                        ip_data[k] = last_value
            else:
                ip_data[i] = data[i]  # This might be redundant
                last_value = data[i]

        return ip_data[:, 0], vuv_vector[:, 0]

    def _resize_f0(self, x: NDArray[np.float32], target_len: int) -> NDArray[np.float32]:
        # Ensure source is float32 to match the annotated input type.
        source = np.array(x, dtype=np.float32)
        source[source < 0.001] = np.nan
        # np.interp returns float64 by default, so cast the final result back to float32
        target = np.interp(
            np.arange(0, len(source) * target_len, len(source)) / target_len,
            np.arange(0, len(source)),
            source,
        )
        res = np.nan_to_num(target).astype(np.float32)
        return res

    def coarse_f0(
        self,
        f0: NDArray[np.float32],
        f0_bin: int = 256,
    ) -> NDArray[np.int32]:
        """Quantize continuous F0 into discrete coarse Mel bins."""
        return coarse_f0(
            f0,
            f0_bin=f0_bin,
            f0_min=self.f0_min,
            f0_max=self.f0_max,
        )