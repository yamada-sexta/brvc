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

logger = logging.getLogger(__name__)


class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset):
    """
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(
        self,
        audiopaths_and_text: Union[str, Path, list[tuple[str, str, str, str, str]]],
        max_wav_value: float,  # e.g. 32768.0
        sampling_rate: int,  # e.g. 48000
        filter_length: int,  # e.g. 2048
        hop_length: int,  # e.g. 480
        win_length: int,  # e.g. 2048
        max_text_len: int,  # e.g. 5000
        min_text_len: int,  # e.g. 1
    ):
        if isinstance(audiopaths_and_text, (str, Path)):
            self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        elif isinstance(audiopaths_and_text, list) and all(
            isinstance(t, tuple) and len(t) == 5 for t in audiopaths_and_text
        ):
            self.audiopaths_and_text = audiopaths_and_text
        else:
            raise ValueError("Invalid type for audiopaths_and_text")

        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self._filter()

    def _filter(self) -> None:
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        audiopaths_and_text_new: list[tuple[str, str, str, str, str]] = []
        lengths: list[int] = []
        for audiopath, text, pitch, pitchf, dv in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append((audiopath, text, pitch, pitchf, dv))
                lengths.append(os.path.getsize(audiopath) // (3 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid: Union[int, str, torch.Tensor]) -> torch.LongTensor:
        sid = torch.LongTensor([int(sid)])
        return sid

    def get_audio_text_pair(
        self,
        audiopath_and_text: tuple[str, str, str, str, str],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # separate filename and text
        file = audiopath_and_text[0]
        phone = audiopath_and_text[1]
        pitch = audiopath_and_text[2]
        pitchf = audiopath_and_text[3]
        dv = audiopath_and_text[4]

        phone, pitch, pitchf = self.get_labels(phone, pitch, pitchf)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]
        # print(123,phone.shape,pitch.shape,spec.shape)
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            # amor
            len_wav = len_min * self.hop_length

            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]

            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

        return (spec, wav, phone, pitch, pitchf, dv)

    def get_labels(
        self, phone: Union[str, NDArray], pitch: Union[str, NDArray], pitchf: Union[str, NDArray]
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        if isinstance(phone, str):
            phone = np.load(phone)
        phone = np.repeat(phone, 2, axis=0)
        if isinstance(pitch, str):
            pitch = np.load(pitch)
        if isinstance(pitchf, str):
            pitchf = np.load(pitchf)
        n_num = min(phone.shape[0], 900)  # DistributedBucketSampler
        phone = phone[:n_num, :]
        pitch = pitch[:n_num]
        pitchf = pitchf[:n_num]
        phone_t = torch.FloatTensor(phone)
        pitch_t = torch.LongTensor(pitch)
        pitchf_t = torch.FloatTensor(pitchf)
        return phone_t, pitch_t, pitchf_t

    def get_audio(self, filename: str) -> tuple[torch.Tensor, torch.Tensor]:
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Sample rate mismatch: {sampling_rate}Hz doesn't match target {self.sampling_rate}Hz"
            )
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.safetensors")
        if os.path.exists(spec_filename):
            try:
                spec = safetensors.torch.load_file(spec_filename)["spec"]
            except Exception as e:
                # logger.warning("%s %s", spec_filename, traceback.format_exc())
                # logger.warning(f"Failed to load {spec_filename}, recomputing spectrogram.")
                logger.warning(f"Failed to load {spec_filename}: {e}")
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
                spec = torch.squeeze(spec, 0)
                safetensors.torch.save_file({"spec": spec}, spec_filename)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            safetensors.torch.save_file({"spec": spec}, spec_filename)
        return spec, audio_norm

    def __getitem__(self, index: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)
