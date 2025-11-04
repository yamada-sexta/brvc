from lib.features.pitch.pitch_predictor import PitchExtractor
from numpy.typing import NDArray
import numpy as np
from swift_f0 import SwiftF0  # Assuming swift_f0 is the correct module name
class Swift(PitchExtractor):
    def __init__(
        self,
    ):
        super().__init__()
        self.swift_f0 = SwiftF0(
            # sample_rate=self.sr,
            # frame_length=self.window,
            fmin=self.f0_min,
            fmax=self.f0_max,
            confidence_threshold=0.9
        )

    def extract_pitch(
        self, audio: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        # Placeholder implementation for SwiftF0 pitch extraction
        # Replace with actual SwiftF0 extraction logic
        # duration = len(audio) // self.window
        # f0 = np.linspace(self.f0_min, self.f0_max, num=duration).astype(
        #     np.float32
        # )
        # return f0
#         @dataclass
# class PitchResult:
#     pitch_hz: np.ndarray      # F0 estimates (Hz) for each frame
#     confidence: np.ndarray    # Model confidence [0.0–1.0] for each frame
#     timestamps: np.ndarray    # Frame centers in seconds for each frame
#     voicing: np.ndarray       # Boolean voicing decisions for each frame

    
        result = self.swift_f0.detect_from_array(audio, self.sr)
        window = self.window / self.sr  # Convert window size to seconds
        frame_times = np.arange(len(result.pitch_hz)) * window + (window / 2)
        expected_length = int(len(audio) / self.window)  # Number of frames
        f0_resampled = np.interp(
            np.linspace(0, frame_times[-1], expected_length),
            frame_times,
            result.pitch_hz,
            left=0,
            right=0
        ).astype(np.float32)
        # Convert the result to the expected format
        # Padding or trimming to match the expected length
        if f0_resampled.size < expected_length:
            f0_resampled = np.pad(f0_resampled, (0, expected_length - f0_resampled.size), mode='edge')
        else:
            f0_resampled = f0_resampled[:expected_length]
        return f0_resampled