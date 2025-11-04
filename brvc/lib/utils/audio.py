from typing import Optional, Union, cast
from pathlib import Path
import av
from av import AudioResampler
from av import AudioFrame
import numpy as np
from numpy.typing import NDArray


def load_audio(file: Union[str, Path], resample_rate: int) -> NDArray[np.float32]:
    """Load audio file using PyAV and resample to target sample rate."""
    audio_data: list[NDArray] = []
    with av.open(file, "r") as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = AudioResampler(format="flt", layout="mono", rate=resample_rate)

        for frame in container.decode(stream):
            # Ensure frame is audio frame
            if not isinstance(frame, AudioFrame):
                continue
            # Resample returns either a frame or a list of frames
            resampled = resampler.resample((frame))
            if not resampled:
                continue
            if isinstance(resampled, list):
                frames = resampled
            else:
                frames = [resampled]

            for f in frames:
                arr = f.to_ndarray()
                audio_data.append(arr)

    res: NDArray[np.float32] = (
        np.concatenate(audio_data, axis=1).flatten().astype(np.float32)
    )
    return res
