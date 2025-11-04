from numpy.typing import NDArray
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from numba import njit

# @njit
# def change_rms(
#     data1: NDArray[np.float32],
#     sr1: int,
#     data2: NDArray[np.float32],
#     sr2: int,
#     rate: float,
# ) -> NDArray[np.float32]:
#     rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
#     rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
#     rms1 = torch.from_numpy(rms1)
#     rms1 = F.interpolate(
#         rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
#     ).squeeze()
#     rms2 = torch.from_numpy(rms2)
#     rms2 = F.interpolate(
#         rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
#     ).squeeze()
#     rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
#     data2 *= (
#         torch.pow(rms1, torch.tensor(1 - rate))
#         * torch.pow(rms2, torch.tensor(rate - 1))
#     ).numpy()
#     return data2

from numpy.typing import NDArray
import numpy as np
import librosa


# Removed torch imports
def change_rms(
    data1: NDArray[np.float32],
    sr1: int,
    data2: NDArray[np.float32],
    sr2: int,
    rate: float,
) -> NDArray[np.float32]:
    """
    Changes the RMS of data2 to be a weighted blend between the RMS of data1 and data2.
    This version uses only NumPy for interpolation and calculations.
    """

    # Calculate RMS for both signals.
    # librosa.feature.rms returns shape (1, t), so we take [0] to get a 1D array.
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)[
        0
    ]
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)[
        0
    ]

    # Get the target length for interpolation (must match data2)
    target_len = data2.shape[0]

    # Create the x-coordinates for the original RMS arrays
    xp1 = np.linspace(0, 1, num=rms1.shape[0])
    xp2 = np.linspace(0, 1, num=rms2.shape[0])

    # Create the x-coordinates for the target interpolated array
    x_target = np.linspace(0, 1, num=target_len)

    # Perform linear interpolation using NumPy
    rms1_interp = np.interp(x_target, xp1, rms1)
    rms2_interp = np.interp(x_target, xp2, rms2)

    # Clamp rms2 to avoid division by zero (equivalent to torch.max(..., 1e-6))
    rms2_interp = np.maximum(rms2_interp, 1e-6)

    # Calculate the scaling factor using NumPy operations
    # (rms1 ** (1 - rate)) * (rms2 ** (rate - 1))
    scaling_factor = np.power(rms1_interp, 1 - rate) * np.power(rms2_interp, rate - 1)

    # Apply the scaling factor to data2 in-place
    # NumPy will handle the broadcast multiplication
    # The result will be stored back into data2, preserving its float32 type
    data2 *= scaling_factor

    return data2
