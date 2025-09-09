from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import numpy as np


class F0Predictor(ABC):
    """
    An abstract base class for F0 (fundamental frequency) predictors.

    Subclasses must implement the `compute_f0` method.
    """

    def __init__(
        self,
        hop_length: int = 512,
        f0_min: int = 50,
        f0_max: int = 1100,
        sampling_rate: int = 44100,
        device: Optional[str] = None,
    ):
        """
        Initializes the F0 predictor.

        Args:
            hop_length (int): The number of samples between successive analysis frames.
            f0_min (int): Minimum F0 to search for, in Hz.
            f0_max (int): Maximum F0 to search for, in Hz.
            sampling_rate (int): The audio sampling rate.
            device (Optional[str]): The device to use for computation (e.g., "cpu", "cuda:0").
                                   Defaults to GPU if available, otherwise CPU.
        """
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    @abstractmethod
    def compute_f0(
        self,
        wav: torch.Tensor,
        p_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Computes the fundamental frequency (F0) from a waveform.

        This method must be implemented by any subclass.
        """
        raise NotImplementedError

    def _interpolate_f0(self, f0: np.ndarray):
        """
        Interpolate f0
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
                ip_data[i] = data[i]  # 这里可能存在一个没有必要的拷贝
                last_value = data[i]

        return ip_data[:, 0], vuv_vector[:, 0]

    def _resize_f0(self, x: np.ndarray, target_len: int):
        source = np.array(x)
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * target_len, len(source)) / target_len,
            np.arange(0, len(source)),
            source,
        )
        res = np.nan_to_num(target)
        return res
