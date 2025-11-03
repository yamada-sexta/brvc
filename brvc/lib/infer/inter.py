from pathlib import Path
from typing import Optional, Union, List
from numpy.typing import NDArray
import resampy
from lib.config.v2_config import default_config, ConfigV2
from lib.utils.audio import load_audio
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
    from fairseq.models.hubert.hubert import HubertModel

logger = logging.getLogger(__name__)

# High-pass filter for audio preprocessing
res = signal.butter(N=5, Wn=48, btype="high", fs=16000)
if res is None:
    raise ValueError("Failed to create high-pass filter coefficients.")
if len(res) != 2:
    raise ValueError("High-pass filter coefficients should be a tuple of (b, a).")
bh, ah = res


def change_rms(
    data1: NDArray[np.float32],
    sr1: int,
    data2: NDArray[np.float32],
    sr2: int,
    rate: float,
) -> NDArray[np.float32]:
    """
    Mix RMS levels from input audio (data1) with output audio (data2).

    Parameters
    ----------
    data1 : NDArray[np.float32]
        Input audio data
    sr1 : int
        Sample rate of input audio
    data2 : NDArray[np.float32]
        Output audio data
    sr2 : int
        Sample rate of output audio
    rate : float
        Mix rate (0.0 = use data1 RMS, 1.0 = use data2 RMS)

    Returns
    -------
    NDArray[np.float32]
        Audio with mixed RMS levels
    """
    # Calculate frame size (half second)
    frame_size1 = sr1 // 2
    frame_size2 = sr2 // 2

    # Pad data to make it divisible by frame size
    pad1 = (frame_size1 - (len(data1) % frame_size1)) % frame_size1
    pad2 = (frame_size2 - (len(data2) % frame_size2)) % frame_size2

    data1_padded = np.pad(data1, (0, pad1), mode="constant") if pad1 > 0 else data1
    data2_padded = np.pad(data2, (0, pad2), mode="constant") if pad2 > 0 else data2

    # Reshape and calculate RMS
    rms1 = np.sqrt(np.mean(np.square(data1_padded.reshape(-1, frame_size1)), axis=1))
    rms2 = np.sqrt(np.mean(np.square(data2_padded.reshape(-1, frame_size2)), axis=1))

    # Convert to tensors and interpolate to match output length
    rms1_tensor = torch.from_numpy(rms1).unsqueeze(0).unsqueeze(0)
    rms2_tensor = torch.from_numpy(rms2).unsqueeze(0).unsqueeze(0)

    rms1_interp = F.interpolate(
        rms1_tensor, size=len(data2), mode="linear", align_corners=False
    ).squeeze()
    rms2_interp = F.interpolate(
        rms2_tensor, size=len(data2), mode="linear", align_corners=False
    ).squeeze()

    # Avoid division by zero
    rms2_interp = torch.max(rms2_interp, torch.ones_like(rms2_interp) * 1e-6)

    # Apply RMS mixing
    data2_tensor = torch.from_numpy(data2)
    data2_tensor *= torch.pow(rms1_interp, 1 - rate) * torch.pow(rms2_interp, rate - 1)

    return data2_tensor.numpy().astype(np.float32)


def resample_audio(
    audio: NDArray[np.float32],
    orig_sr: int,
    target_sr: int,
) -> NDArray[np.float32]:
    # Check if the audio is stereo and downmix to mono
    if audio.ndim > 1 and audio.shape[1] > 1:
        # print("Detected stereo audio, downmixing to mono.")
        # Average the channels to create a mono signal
        audio_mono = audio.mean(axis=1)
    else:
        # Already mono or 1D array
        audio_mono = audio.flatten()  # Ensure it's 1D in case it's (N, 1)

    # print(f"Mono audio shape after downmixing: {audio_mono.shape}")

    if audio_mono.size < 10:  # A reasonable minimum length for resampling
        raise ValueError(
            f"Mono audio signal length ({audio_mono.size}) is too small to resample from {orig_sr} to {target_sr}. "
            "Ensure the audio file contains actual sound data."
        )

    # Perform resampling on the mono signal
    resampled_audio = resampy.resample(audio_mono, orig_sr, target_sr)
    # print(f"Resampled audio shape: {resampled_audio.shape}")
    return resampled_audio


def interface_cli(
    g_path: Path,
    audio: Path,
    output: Optional[Path] = None,
    sample_rate: int = 48000,
    f0_offset: int = 0,
    f0_method: str = "crepe",
    index_rate: float = 0.0,
    protect: float = 0.33,
    rms_mix_rate: float = 0.25,
    resample_sr: int = 0,
):
    from accelerate import Accelerator
    import torch
    import soundfile as sf

    logger.info("Starting inference...")
    accelerator = Accelerator()
    device = accelerator.device

    logger.info(f"Loading audio from {audio}...")
    audio_data = load_audio(audio, resample_rate=sample_rate)

    logger.info("Loading synthesis model...")

    from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid

    m = default_config["model"]
    filter_length = ConfigV2.Data.filter_length
    hop_length = ConfigV2.Data.hop_length
    M = ConfigV2.Model

    net_g = SynthesizerTrnMsNSFsid(
        spec_channels=filter_length // 2 + 1,
        segment_size=ConfigV2.Train.segment_size // hop_length,
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

    cpt = torch.load(g_path, map_location="cpu")
    net_g.load_state_dict(cpt["model"])
    net_g.eval()
    net_g.to(device)

    audio_out = inference(
        net_g=net_g,
        audio=audio_data,
        sample_rate=sample_rate,
        accelerator=accelerator,
        f0_offset=f0_offset,
        f0_method=f0_method,
        index_rate=index_rate,
        protect=protect,
        rms_mix_rate=rms_mix_rate,
        resample_sr=resample_sr if resample_sr > 0 else sample_rate,
    )

    # Determine output path
    if output is None:
        output = audio.parent / f"{audio.stem}_out.wav"

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save output audio
    logger.info(f"Saving output to {output}...")
    sf.write(output, audio_out, sample_rate if resample_sr == 0 else resample_sr)

    logger.info("Inference complete!")
    return output


from time import time


def process_chunk(
    hubert_model: "HubertModel",
    net_g: "SynthesizerTrnMsNSFsid",
    sid: int,
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
    # assert feats.dim() == 1, feats.dim(), "Input audio should be 1D."
    assert feats.dim() == 1, "Input audio should be 1D."
    feats = feats.view(1, -1)
    feats = feats.to(device)
    padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(device)

    inputs = {
        "source": feats,
        "padding_mask": padding_mask,
        "output_layer": 12,
    }

    t0 = time()
    with torch.no_grad():
        logits = hubert_model.extract_features(**inputs)
        feats: torch.Tensor = logits[0]

    if protect < 0.5:
        feats0 = feats.clone()
    else:
        feats0 = None

    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    if protect < 0.5 and feats0:
        feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

    t1 = time()

    p_len = audio.shape[0] // window
    if feats.shape[1] < p_len:
        p_len = feats.shape[1]
        pitch = pitch[:, :p_len]
        pitchf = pitchf[:, :p_len]

    if protect < 0.5 and feats0:
        # Create protection mask based on F0 (voiced/unvoiced)
        pitchff = pitch.clone()
        pitchff[pitch > 0] = 1
        pitchff[pitch < 1] = protect
        pitchff = pitchff.unsqueeze(-1)

        # Mix original and current features
        feats = feats * pitchff + feats0 * (1 - pitchff)
        feats = feats.to(feats0.dtype)
    else:
        pitchff = None

    p_len = torch.tensor([p_len], device=device).long()
    with torch.no_grad():
        res = net_g.infer(
            phone=feats,
            phone_lengths=p_len,
            pitch=pitch,
            nsff0=pitchf,
            sid=torch.tensor([sid], device=device).long(),
        )

        converted_audio: NDArray[np.float32] = res[0][0, 0].data.cpu().float().numpy()

    t2 = time()
    # Run Garbage Collection
    from gc import collect

    collect()
    # Empty CUDA Cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    times[0] += t1 - t0
    times[1] += t2 - t1
    return converted_audio


def get_f0(
    accelerator: "Accelerator",
    p_len: int,
    sample_rate: int = 48000,
    window: int = 160,
    f0_min=50,
    f0_max=1100,
):
    logger.info("Loading F0 extractor...")
    device = accelerator.device
    from lib.features.pitch.crepe import CRePE

    f0_mel_min = 1127 * np.log1p(f0_min / 700)
    f0_mel_max = 1127 * np.log1p(f0_max / 700)
    f0_extractor = CRePE(device=device, sample_rate=sample_rate, hop_length=window)

    f0 = f0_extractor.compute_f0(audio_16k)

    pass


def inference(
    net_g: "SynthesizerTrnMsNSFsid",
    audio: NDArray[np.float32],
    accelerator: "Accelerator",
    sample_rate: int = 48000,
    f0_offset: int = 0,
    f0_method: str = "crepe",
    index_rate: float = 0.0,
    protect: float = 0.33,
    rms_mix_rate: float = 0.25,
    sr: int = 16000,
    # Pipeline config constants
    tgt_sr: int = 48000,
    window: int = 160,
    x_pad: int = 1,
    x_query: int = 6,
    x_center: int = 38,
    x_max: int = 41,
    f0_min: int = 50,
    f0_max: int = 1100,
) -> NDArray[np.float32]:
    device = accelerator.device

    t_max = sr * x_max

    audio = signal.filtfilt(bh, ah, audio).copy()  # Make contiguous copy
    audio_pad = np.pad(audio, (window // 2, window // 2), mode="reflect")
    opt_ts = []
    if audio_pad.shape[0] < t_max:
        audio_sum = np.zeros_like(audio)
        for i in range(window):
            audio_sum += audio_pad[i : i - window]

    logger.info("Loading HuBERT model...")
    from lib.train.extract_features import load_hubert_model

    hubert_model, _ = load_hubert_model(accelerator=accelerator)

    if isinstance(audio, Path):
        logger.info(f"Loading audio from {audio}...")
        audio = load_audio(audio, resample_rate=sample_rate)

    # Apply high-pass filter
    logger.info("Applying high-pass filter...")
    audio_16k = resample_audio(audio, orig_sr=sample_rate, target_sr=16000)
    audio_16k = signal.filtfilt(bh, ah, audio_16k).copy()  # Make contiguous copy


def main():
    from tap import tapify

    logging.basicConfig(level=logging.INFO)

    tapify(interface_cli)


if __name__ == "__main__":
    main()
