import logging
from pathlib import Path
import traceback
from typing import Union
import torch
import os
import numpy as np
from numpy.typing import NDArray
import safetensors.torch

from lib.train.utils.mel_processing import spectrogram_torch
from lib.train.utils.path import load_filepaths_and_text, load_wav_to_torch
import csv

logger = logging.getLogger(__name__)


class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset):
    """
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(
        self,
        audio_and_text_path: Union[str, Path, list[tuple[Path, Path, Path]]],
        max_wav_value: float,  # e.g. 32768.0
        sampling_rate: int,  # e.g. 48000
        filter_length: int,  # e.g. 2048
        hop_length: int,  # e.g. 480
        win_length: int,  # e.g. 2048
        max_text_len: int,  # e.g. 5000
        min_text_len: int,  # e.g. 1
    ):
        if isinstance(audio_and_text_path, (str, Path)):
            # Load from file, assuming it's CSV with 3 columns
            self.audio_and_text_path = []
            with open(audio_and_text_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) != 3:
                        raise ValueError("Each row must have exactly 3 columns.")
                    self.audio_and_text_path.append(
                        (Path(row[0]), Path(row[1]), Path(row[2]))
                    )
        elif isinstance(audio_and_text_path, list) and all(
            isinstance(t, tuple) and len(t) == 3 for t in audio_and_text_path
        ):
            self.audio_and_text_path = audio_and_text_path
        else:
            raise ValueError("Invalid type for audio_and_text_path")

        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        
        logger.info(
            f"Loaded {len(self.audio_and_text_path)} audio-text pairs before filtering."
        )
        self._filter()

    def _filter(self) -> None:
        """
        Filter text & store spec lengths
        """
        audiopaths_and_text_new: list[tuple[Path, Path, Path]] = []
        lengths: list[int] = []
        for audiopath, phone_path, pitch_path in self.audio_and_text_path:
            # phone_path is expected to be a .safetensors file containing 'feats'
            try:
                phone_dict = safetensors.torch.load_file(phone_path)
                feats = phone_dict.get("feats")
                if feats is None:
                    logger.warning(f"No 'feats' key in {phone_path}, skipping")
                    continue
                n_frames = int(feats.shape[0])
            except Exception as e:
                logger.warning(f"Failed to read phone safetensors {phone_path}: {e}\n{traceback.format_exc()}")
                continue

            if self.min_text_len <= n_frames <= self.max_text_len:
                audiopaths_and_text_new.append((audiopath, phone_path, pitch_path))
                lengths.append(n_frames)
            else:
                logger.debug(
                    f"Skipping {phone_path}: frame count {n_frames} outside [{self.min_text_len}, {self.max_text_len}]"
                )

        self.audio_and_text_path = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self) -> torch.LongTensor:
        return torch.LongTensor([0])  # single speaker

    def get_audio_text_pair(
        self,
        audio_and_text_path: tuple[Path, Path, Path],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        file, phone, pitch = audio_and_text_path
        phone, pitch, pitchf = self.get_labels(phone, pitch)
        spec, wav = self.get_audio(file)
        dv = self.get_sid()
        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length
            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

        return (spec, wav, phone, pitch, pitchf, dv)

    def get_labels(
        self,
        phone_p: Path,
        pitch_p: Path,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert phone_p.suffix == ".safetensors", "Only .safetensors files are supported"
        assert pitch_p.suffix == ".safetensors", "Only .safetensors files are supported"

        phone_dict = safetensors.torch.load_file(phone_p)
        pitch_dict = safetensors.torch.load_file(pitch_p)
        phone = phone_dict["feats"]
        pitch = pitch_dict["coarse_f0"]
        pitchf = pitch_dict["f0"]
        n_num = min(phone.shape[0], 900)  # DistributedBucketSampler
        phone = phone[:n_num, :]
        pitch = pitch[:n_num]
        pitchf = pitchf[:n_num]

        return (
            phone,
            pitch,
            pitchf,
        )

    def get_audio(self, filename: Path) -> tuple[torch.Tensor, torch.Tensor]:
        assert filename.suffix == ".wav", "Only .wav files are supported"
        audio_tensor = filename.with_suffix(".safetensors")
        if os.path.exists(audio_tensor):
            try:
                data = safetensors.torch.load_file(audio_tensor)
                spec = data["spec"]
                audio_norm = data["audio"]
                return spec, audio_norm
            except Exception as e:
                logger.warning(f"Failed to load {audio_tensor}: {e}")
                logger.warning("Recomputing spectrogram and audio tensor.")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Sample rate mismatch: {sampling_rate}Hz doesn't match target {self.sampling_rate}Hz"
            )
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        safetensors.torch.save_file({"spec": spec.contiguous(), "audio": audio_norm.contiguous()}, audio_tensor)
        return spec, audio_norm

    def __getitem__(self, index: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return self.get_audio_text_pair(self.audio_and_text_path[index])

    def __len__(self):
        return len(self.audio_and_text_path)
