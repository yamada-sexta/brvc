import logging
import traceback
from typing import Union
import torch
import os
import numpy as np
from torch import FloatTensor
from torch import LongTensor

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
        audiopaths_and_text: str,
        max_wav_value: float,  # e.g. 32768.0
        sampling_rate: int,  # e.g. 48000
        filter_length: int,  # e.g. 2048
        hop_length: int,  # e.g. 480
        win_length: int,  # e.g. 2048
        max_text_len: int,  # e.g. 5000
        min_text_len: int,  # e.g. 1
    ):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
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
        audiopath_and_text: tuple[str, str, str, str, Union[int, str, torch.Tensor]],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.LongTensor,
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

    def get_labels(self, phone, pitch, pitchf):
        phone = np.load(phone)
        phone = np.repeat(phone, 2, axis=0)
        pitch = np.load(pitch)
        pitchf = np.load(pitchf)
        n_num = min(phone.shape[0], 900)  # DistributedBucketSampler
        # print(234,phone.shape,pitch.shape)
        phone = phone[:n_num, :]
        pitch = pitch[:n_num]
        pitchf = pitchf[:n_num]
        phone = torch.FloatTensor(phone)
        pitch = torch.LongTensor(pitch)
        pitchf = torch.FloatTensor(pitchf)
        return phone, pitch, pitchf

    def get_audio(self, filename: str) -> tuple[torch.Tensor, torch.Tensor]:
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio
        #        audio_norm = audio / self.max_wav_value
        #        audio_norm = audio / np.abs(audio).max()

        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename, weights_only=False)
            except:
                logger.warning("%s %s", spec_filename, traceback.format_exc())
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
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
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollateMultiNSFsid:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids: bool = False):
        self.return_ids = return_ids

    def __call__(
        self,
        batch: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.LongTensor,
            ]
        ],
    ) -> tuple[
        FloatTensor,
        LongTensor,
        LongTensor,
        FloatTensor,
        FloatTensor,
        LongTensor,
        FloatTensor,
        LongTensor,
        LongTensor,
    ]:
        """Collate's training batch from normalized text and audio lengths
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])
        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
        spec_padded.zero_()
        wave_padded.zero_()

        max_phone_len = max([x[2].size(0) for x in batch])
        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.FloatTensor(
            len(batch), max_phone_len, batch[0][2].shape[1]
        )  # (spec, wav, phone, pitch)
        pitch_padded = torch.LongTensor(len(batch), max_phone_len)
        pitchf_padded = torch.FloatTensor(len(batch), max_phone_len)
        phone_padded.zero_()
        pitch_padded.zero_()
        pitchf_padded.zero_()
        # dv = torch.FloatTensor(len(batch), 256)#gin=256
        sid = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)

            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)

            pitch = row[3]
            pitch_padded[i, : pitch.size(0)] = pitch
            pitchf = row[4]
            pitchf_padded[i, : pitchf.size(0)] = pitchf

            # dv[i] = row[5]
            sid[i] = row[5]

        return (
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            # dv
            sid,
        )
