from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np

class PitchExtractor(ABC):
    """Abstract base class for all pitch extraction methods."""
    
    sr: int = 16000
    window: int = 160
    f0_min: int = 50
    f0_max: int = 1100

    def __init__(self):
        pass
    
    @abstractmethod
    def extract_pitch(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Extracts and returns the fundamental frequency (f0) from audio."""
        pass
