from pathlib import Path
from typing import Literal, Optional, List
import librosa
from numpy.typing import NDArray
from lib.config.v2_config import ConfigV2
from lib.infer.utils.resample import resample_audio
from lib.infer.utils.rms import change_rms
import numpy as np
import logging
import torch
import torch.nn.functional as F
from scipy import signal
from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid
    from accelerate import Accelerator
    from transformers import HubertModel as HfHubertModel, Wav2Vec2FeatureExtractor

logger = logging.getLogger(__name__)

# High-pass filter for audio preprocessing
res = signal.butter(N=5, Wn=48, btype="high", fs=16000)
if res is None:
    raise ValueError("Failed to create high-pass filter coefficients.")
if len(res) != 2:
    raise ValueError("High-pass filter coefficients should be a tuple of (b, a).")
bh, ah = res
print(f"High-pass filter coefficients:\nb: {bh}\na: {ah}")


def interface_cli(
    g_path: Path,
    audio: Path,
    output: Optional[Path] = None,
    sample_rate: int = 48000,
    f0_offset: int = 0,
    protect: float = 0.33,
    rms_mix_rate: float = 0.25,
    resample_sr: int = 0,
    load_mode: Literal["rvc", "train"] = "train",
):
    if not g_path.exists():
        raise FileNotFoundError(f"Generator model file not found: {g_path}")
    if not g_path.is_file():
        raise ValueError(f"Generator model path is not a file: {g_path}")
    from accelerate import Accelerator
    import torch
    import soundfile as sf

    target_processing_sr = 16000
    logger.info("Starting inference...")
    accelerator = Accelerator()
    device = accelerator.device

    logger.info(f"Loading audio from {audio}...")

    audio_data, original_sr = sf.read(str(audio), dtype="float32")
    logger.info(
        f"Loaded audio dtype: {audio_data.dtype}, shape: {audio_data.shape}, original_sr: {original_sr}"
    )

    # ensure we have a 1-D mono signal: if multi-channel, convert to mono by averaging channels
    if audio_data.ndim == 2:
        # averaging channels is the simplest and usually acceptable approach
        logger.info(
            f"Input has {audio_data.shape[1]} channels — converting to mono by averaging."
        )
        audio_data = np.mean(audio_data, axis=1)

    gain = 1.933333333333333  # adjust as desired, e.g., 1.1–1.5
    audio_data *= gain
    # At this point audio_data should be float32 in approximately -1..1 range.
    # If it's outside that range (some files store int16 but requested as float),
    # we normalize by the max absolute value to avoid clipping.
    max_abs = float(np.abs(audio_data).max()) if audio_data.size else 0.0
    if max_abs == 0.0:
        logger.warning("Loaded audio is silence (all zeros).")
    elif max_abs > 1.0:
        logger.info(f"Audio samples outside [-1,1]. Normalizing by factor {max_abs}.")
        audio_data = (audio_data / max_abs).astype(np.float32)
    else:
        # ensure dtype is float32
        audio_data = audio_data.astype(np.float32)
    # Save the audio data to check if loading is correct
    # np.save("debug_loaded_audio.npy", audio_data, allow_pickle=False)

    # Resample to the model's processing SR (if needed).
    if original_sr != target_processing_sr:
        logger.info(
            f"Resampling audio from {original_sr} Hz -> {target_processing_sr} Hz for processing..."
        )
        # Preferably use your project's resample_audio helper if available
        # try:
        # if resample_audio is defined/importable in your codebase, use it
        audio_data = resample_audio(
            audio_data, orig_sr=original_sr, target_sr=target_processing_sr
        )
        # except NameError:
        #     # resample_audio not defined in this scope — use fallback
        #     audio_data = fallback_resample(audio_data, orig_sr=original_sr, target_sr=target_processing_sr)
        # except Exception as e:
        #     logger.exception("Resampling failed.")
        #     raise

    post_max = float(np.abs(audio_data).max()) if audio_data.size else 0.0
    if post_max > 1.0:
        logger.info(
            f"Post-resample peak {post_max} > 1. Normalizing to avoid clipping."
        )
        audio_data /= post_max
    times = [0.0, 0.0, 0.0]

    logger.info("Loading synthesis model...")

    from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid

    filter_length = ConfigV2.Data.filter_length
    hop_length = ConfigV2.Data.hop_length
    M = ConfigV2.Model

    net_g = SynthesizerTrnMsNSFsid(
        spec_channels=filter_length // 2 + 1,
        segment_size=32,
        inter_channels=M.inter_channels,
        hidden_channels=M.hidden_channels,
        filter_channels=M.filter_channels,
        n_heads=M.n_heads,
        n_layers=M.n_layers,
        kernel_size=M.kernel_size,
        p_dropout=M.p_dropout,
        resblock=M.resblock,
        resblock_kernel_sizes=M.resblock_kernel_sizes,
        resblock_dilation_sizes=M.resblock_dilation_sizes,
        upsample_rates=M.upsample_rates,
        upsample_initial_channel=M.upsample_initial_channel,
        upsample_kernel_sizes=M.upsample_kernel_sizes,
        spk_embed_dim=M.spk_embed_dim,
        gin_channels=M.gin_channels,
        sr=ConfigV2.Data.sampling_rate,
        # is_half=is_half,
        lrelu_slope=0.1,
        txt_channels=768,
    )

    print(f"Spectrum channels: {filter_length // 2 + 1}")
    print(f"Segment size: {ConfigV2.Train.segment_size // hop_length}")
    print(f"Hop length: {hop_length}")
    if g_path.suffix == ".safetensors":
        import safetensors.torch

        cpt = safetensors.torch.load_file(g_path)
        net_g.load_state_dict(cpt)
    elif g_path.suffix == ".pth" or g_path.suffix == ".pt":
        cpt = torch.load(g_path, map_location="cpu")
        if load_mode == "rvc":
            net_g.load_state_dict(cpt["weight"])
        elif load_mode == "train":
            net_g.load_state_dict(cpt["model"])
        else:
            raise ValueError(
                f"Invalid load_mode: {load_mode}. Choose 'rvc' or 'train'."
            )
    else:
        raise ValueError(f"Unsupported model file extension: {g_path.suffix}")

    del net_g.enc_q  # remove unused module to save memory
    net_g.eval()
    net_g.to(device)

    audio_out = inference(
        net_g=net_g,
        audio=audio_data.astype(np.float32),
        # sample_rate=sample_rate,
        accelerator=accelerator,
        f0_offset=f0_offset,
        protect=protect,
        rms_mix_rate=rms_mix_rate,
    )

    # Determine output path
    if output is None:
        output = audio.parent / f"{audio.stem}_out.wav"
        # Make sure we don't overwrite existing files
        count = 1
        while output.exists():
            output = audio.parent / f"{audio.stem}_out_{count}.wav"
            count += 1
    logger.info(f"Saving output audio to {output}...")

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save output audio at the target sample rate

    sf.write(output, audio_out, samplerate=sample_rate, subtype="PCM_16")

    # logger.info("Inference complete!")
    return output


from time import time


def process_chunk(
    hubert_model: "HfHubertModel",
    feature_extractor: "Wav2Vec2FeatureExtractor",
    net_g: "SynthesizerTrnMsNSFsid",
    # sid: int,
    sid: torch.Tensor,
    audio: NDArray[np.float32],
    pitch: torch.Tensor,
    pitchf: torch.Tensor,
    times: List[float],
    protect: float,
    device: torch.device,
    window: int = 160,
    # big_npy:
) -> NDArray[np.float32]:
    feats = torch.from_numpy(audio)
    feats = feats.float()
    if feats.dim() == 2:  # stereo audio
        feats = feats.mean(-1)

    assert feats.dim() == 1, "Input audio should be 1D."
    feats = feats.view(1, -1)
    # feats = feats.to(device)
    # padding_mask = torch.BoolTensor(feats.shape).to(device).fill_(False)

    # inputs = {
    #     "source": feats.to(device),
    #     "padding_mask": padding_mask,
    #     "output_layer": 12,
    # }

    t0 = time()
    with torch.no_grad():
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Forward pass
        outputs = hubert_model(**inputs)
        feats = outputs.last_hidden_state  # shape: (1, T, 768)

    # Save feats as npy for debugging
    feats_np: NDArray[np.float32] = feats.squeeze(0).float().cpu().numpy()
    np.save("debug_feats.npy", feats_np, allow_pickle=False)
    if protect < 0.5:
        feats0 = feats.clone()
    else:
        feats0 = None

    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    if protect < 0.5 and feats0 is not None:
        feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

    # Save interpolated feats as npy for debugging
    feats_interp_np: NDArray[np.float32] = feats.squeeze(0).float().cpu().numpy()
    np.save("debug_feats_interp.npy", feats_interp_np, allow_pickle=False)

    t1 = time()

    p_len = audio.shape[0] // window
    if feats.shape[1] < p_len:
        p_len = feats.shape[1]
        pitch = pitch[:, :p_len]
        pitchf = pitchf[:, :p_len]

    if protect < 0.5 and feats0 is not None:
        pitchff = pitchf.clone()
        pitchff[pitchf > 0] = 1
        pitchff[pitchf < 1] = protect
        pitchff = pitchff.unsqueeze(-1)
        feats = feats * pitchff + feats0 * (1 - pitchff)
        feats = feats.to(feats0.dtype)
    else:
        pitchff = None

    p_len = torch.tensor([p_len], device=device).long()
    with torch.no_grad():
        args = (feats, p_len, pitch, pitchf, sid)
        res = net_g.infer(
            phone=feats,
            phone_lengths=p_len,
            pitch=pitch,
            nsff0=pitchf,
            sid=torch.tensor([sid], device=device).long(),
        )
        audio1: NDArray[np.float32] = res[0][0, 0].data.cpu().float().numpy()

    del feats, p_len
    t2 = time()
    # Run Garbage Collection
    from gc import collect

    collect()
    # Empty CUDA Cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    times[0] += t1 - t0
    times[1] += t2 - t1

    # save converted audio for debugging
    np.save("debug_converted_audio.npy", audio1, allow_pickle=False)
    return audio1


import json


def get_f0(
    accelerator: "Accelerator",
    x: np.ndarray,
    p_len: int,
    f0_up_key: int,
    window: int = 160,
    f0_min: int = 50,
    f0_max: int = 1100,
    sr: int = 16000,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    logger.info("Loading F0 extractor...")
    # debug_info = {
    #     "p_len": p_len,
    #     "f0_up_key": f0_up_key,
    #     "window": window,
    #     "f0_min": f0_min,
    #     "f0_max": f0_max,
    # }
    # print(f"F0 Extraction Debug Info: {json.dumps(debug_info, indent=2)}")
    from lib.features.pitch.swift import Swift

    f0_extractor = Swift(
        # sample_rate=sr, window_size=window, f0_min=f0_min, f0_max=f0_max, device=device
    )

    f0 = f0_extractor.extract_pitch(x)

    f0 *= pow(2, f0_up_key / 12)

    tf0 = sr // window
    f0bak = f0.copy()
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int32)

    # Save for debugging
    np.save("debug_f0bak.npy", f0bak)
    np.save("debug_f0_coarse.npy", f0_coarse)
    # Should be the same as the RVC implementation
    return f0_coarse, f0bak


def inference(
    net_g: "SynthesizerTrnMsNSFsid",
    audio: NDArray[np.float32],
    accelerator: "Accelerator",
    f0_offset: int = 0,
    protect: float = 0.33,
    rms_mix_rate: float = 0.25,
    sr: int = 16000,
    tgt_sr: int = 48000,
    resample_sr: int = 0,
    window: int = 160,  # hop_length for 16kHz (should match sr // 100)
    x_pad: int = 3,
    x_query: int = 10,
    x_center: int = 60,
    x_max: int = 65,
    f0_min: int = 50,
    f0_max: int = 1100,
    times: List[float] = [0.0, 0.0, 0.0],
) -> NDArray[np.int16]:

    logger.info("Loading HuBERT model...")
    from lib.train.extract_features import get_hf_hubert_model

    hubert, feature_extractor = get_hf_hubert_model()

    logger.info("Starting inference pipeline...")
    device = accelerator.device
    t_pad = sr * x_pad
    t_pad_tgt = tgt_sr * x_pad
    t_pad2 = t_pad * 2
    t_query = sr * x_query
    t_center = sr * x_center
    t_max = sr * x_max
    import json

    hubert.to(device)
    hubert.eval()
    # feature_extractor.to(device)
    # feature_extractor.eval()

    debug_info = {
        "x_pad": x_pad,
        "x_query": x_query,
        "x_center": x_center,
        "x_max": x_max,
        "t_pad": t_pad,
        "t_pad_tgt": t_pad_tgt,
        "t_pad2": t_pad2,
        "t_query": t_query,
        "t_center": t_center,
        "t_max": t_max,
    }
    logger.info(f"Debug Info: {json.dumps(debug_info, indent=2)}")
    audio = signal.filtfilt(bh, ah, audio)
    # Save audio after filtering for debugging
    np.save("debug_filtered_audio.npy", audio, allow_pickle=False)

    # Also write the filtered audio to a WAV file for listening
    import soundfile as sf

    sf.write("debug_filtered_audio.wav", audio, samplerate=sr)
    audio_pad = np.pad(audio, (window // 2, window // 2), mode="reflect")
    opt_ts: list[int] = []
    if audio_pad.shape[0] < t_max:
        audio_sum = np.zeros_like(audio)
        for i in range(window):
            audio_sum += np.abs(audio_pad[i : i - window])
        for t in range(t_center, audio.shape[0], t_center):
            n = (
                t
                - t_query
                + np.where(
                    audio_sum[t - t_query : t + t_query]
                    == audio_sum[t - t_query : t + t_query].min()
                )[0][0]
            )
            opt_ts.append(n)
    s = 0
    audio_opt: List[NDArray[np.float32]] = []
    t = None
    t1 = time()
    audio_pad = np.pad(audio, (t_pad, t_pad), mode="reflect")
    p_len = audio_pad.shape[0] // window
    # sid = torch.tensor([0], device=device).long().item()

    pitch, pitchf = get_f0(
        accelerator=accelerator,
        p_len=p_len,
        x=audio_pad,
        f0_up_key=f0_offset,
        sr=sr,
    )
    pitch = pitch[:p_len]
    pitchf = pitchf[:p_len]

    pitch = torch.from_numpy(pitch).unsqueeze(0).to(device).long()
    pitchf = torch.from_numpy(pitchf).unsqueeze(0).to(device).float()

    t2 = time()
    times[1] += t2 - t1

    total_segments = len(opt_ts) + 1
    sid = torch.tensor([0], device=device).long()
    for i, t in enumerate(opt_ts):
        logger.info(f"Processing segment {i + 1}/{total_segments}...")
        t = t // window * window
        audio_opt.append(
            process_chunk(
                hubert_model=hubert,
                feature_extractor=feature_extractor,
                net_g=net_g,
                sid=sid,
                audio=audio_pad[s : t + t_pad2 + window],
                pitch=pitch[:, s // window : (t + t_pad2) // window],
                pitchf=pitchf[:, s // window : (t + t_pad2) // window],
                times=times,
                protect=protect,
                device=device,
                window=window,
            )[t_pad_tgt:-t_pad_tgt]
        )
        s = t

    logger.info(f"Processing final segment {total_segments}/{total_segments}...")
    audio_opt.append(
        process_chunk(
            hubert_model=hubert,
            feature_extractor=feature_extractor,
            net_g=net_g,
            sid=sid,
            audio=audio_pad[t:],
            pitch=pitch[:, t // window :] if t is not None else pitch,
            pitchf=pitchf[:, t // window :] if t is not None else pitchf,
            times=times,
            protect=protect,
            device=device,
            window=window,
        )[t_pad_tgt:-t_pad_tgt]
    )

    audio_opt_array: NDArray[np.float32] = np.concatenate(audio_opt)

    # save for debugging
    np.save("debug_audio_opt_array.npy", audio_opt_array, allow_pickle=False)

    if rms_mix_rate != 1:
        audio_out = change_rms(
            data1=audio,
            sr1=16000,  # Input is at 16kHz
            data2=audio_opt_array,
            sr2=tgt_sr,  # Output is at 48kHz (model's native rate)
            rate=rms_mix_rate,
        )
    else:
        audio_out = audio_opt_array

    # Resample to target sample rate if different from model's output
    # if sample_rate != tgt_sr:
    #     logger.info(f"Resampling output from {tgt_sr}Hz to {sample_rate}Hz...")
    #     audio_out = resample_audio(
    #         audio=audio_out, orig_sr=tgt_sr, target_sr=sample_rate
    #     )
    if tgt_sr != resample_sr >= 16000:
        audio_out = librosa.resample(audio_out, orig_sr=tgt_sr, target_sr=resample_sr)

    audio_max = np.abs(audio_out).max() / 0.99
    max_int16 = 32768
    if audio_max > 1:
        max_int16 /= audio_max
    audio_int: NDArray[np.int16] = (audio_out * max_int16).astype(np.int16)

    # Save final output audio for debugging
    np.save("debug_final_output_audio.npy", audio_int, allow_pickle=False)
    # Do GC
    from gc import collect

    collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return audio_int


def main():
    from tap import tapify

    logging.basicConfig(level=logging.INFO)

    tapify(interface_cli)


if __name__ == "__main__":
    main()
