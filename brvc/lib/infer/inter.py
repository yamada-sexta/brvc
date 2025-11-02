from pathlib import Path
from typing import Optional, Union, List
from numpy.typing import NDArray
import resampy


from lib.config.v2_config import default_config, ConfigV2
from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid
from lib.utils.audio import load_audio
import numpy as np
import logging
import torch
import torch.nn.functional as F
from scipy import signal

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid
    from accelerate import Accelerator
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# High-pass filter for audio preprocessing
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


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
    
    data1_padded = np.pad(data1, (0, pad1), mode='constant') if pad1 > 0 else data1
    data2_padded = np.pad(data2, (0, pad2), mode='constant') if pad2 > 0 else data2
    
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
    data2_tensor *= (
        torch.pow(rms1_interp, 1 - rate)
        * torch.pow(rms2_interp, rate - 1)
    )
    
    return data2_tensor.numpy().astype(np.float32)


def resample_audio(
    audio: NDArray,
    orig_sr: int,
    target_sr: int,
) -> NDArray:
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
    f0_up_key: int = 0,
    f0_method: str = "crepe",
    index_rate: float = 0.0,
    protect: float = 0.33,
    rms_mix_rate: float = 0.25,
    resample_sr: int = 0,
):
    """
    CLI interface for voice conversion inference.
    
    Parameters
    ----------
    g_path : Path
        Path to the generator model checkpoint
    audio : Path
        Path to input audio file
    output : Optional[Path]
        Path to save output audio (default: input_path + '_out.wav')
    sample_rate : int
        Target sample rate for processing (default: 48000)
    f0_up_key : int
        Pitch shift in semitones (default: 0)
    f0_method : str
        F0 extraction method: 'crepe', 'harvest', 'pm', 'rmvpe' (default: 'crepe')
    index_rate : float
        Feature index retrieval rate, 0.0-1.0 (default: 0.0, disabled)
    protect : float
        Protection for consonants, 0.0-0.5 (default: 0.33)
    rms_mix_rate : float
        RMS mixing rate with original audio, 0.0-1.0 (default: 0.25)
    resample_sr : int
        Final output sample rate, 0 to keep same as sample_rate (default: 0)
    """
    from accelerate import Accelerator
    import torch
    import soundfile as sf

    logger.info("Starting inference...")
    accelerator = Accelerator()
    device = accelerator.device

    logger.info(f"Loading audio from {audio}...")
    audio_data = load_audio(audio, sr=sample_rate)

    logger.info("Loading synthesis model...")

    from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid

    m = default_config["model"]
    filter_length = ConfigV2.Data.filter_length
    hop_length = ConfigV2.Data.hop_length

    net_g = SynthesizerTrnMsNSFsid(
        spec_channels=filter_length // 2 + 1,
        segment_size=ConfigV2.Train.segment_size // hop_length,
        inter_channels=m["inter_channels"],
        hidden_channels=m["hidden_channels"],
        filter_channels=m["filter_channels"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        kernel_size=m["kernel_size"],
        p_dropout=m["p_dropout"],
        resblock_version=m["resblock"],
        resblock_kernel_sizes=m["resblock_kernel_sizes"],
        resblock_dilation_sizes=m["resblock_dilation_sizes"],
        upsample_rates=m["upsample_rates"],
        upsample_initial_channel=m["upsample_initial_channel"],
        upsample_kernel_sizes=m["upsample_kernel_sizes"],
        spk_embed_dim=m["spk_embed_dim"],
        gin_channels=m["gin_channels"],
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
        f0_up_key=f0_up_key,
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


def inference(
    net_g: SynthesizerTrnMsNSFsid,
    audio: NDArray[np.float32],
    accelerator: "Accelerator",
    sample_rate: int = 48000,
    f0_up_key: int = 0,
    f0_method: str = "crepe",
    index_rate: float = 0.0,
    protect: float = 0.33,
    rms_mix_rate: float = 0.25,
    resample_sr: int = 48000,
) -> NDArray[np.float32]:
    """
    Perform voice conversion inference.
    
    Parameters
    ----------
    net_g : SynthesizerTrnMsNSFsid
        Generator model
    audio : NDArray[np.float32]
        Input audio data
    accelerator : Accelerator
        Accelerate device wrapper
    sample_rate : int
        Sample rate of input audio
    f0_up_key : int
        Pitch shift in semitones
    f0_method : str
        F0 extraction method
    index_rate : float
        Feature index retrieval rate (not currently used)
    protect : float
        Protection for consonants
    rms_mix_rate : float
        RMS mixing rate
    resample_sr : int
        Final output sample rate
    
    Returns
    -------
    NDArray[np.float32]
        Converted audio
    """
    import torch

    device = accelerator.device
    
    # Constants from Pipeline class
    sr = 16000  # HuBERT input sample rate
    window = 512  # Hop length for HuBERT
    x_pad = 1  # Padding in seconds
    
    # Store original audio for RMS mixing
    original_audio = audio.copy()

    logger.info("Loading F0 extractor...")
    from lib.features.pitch.crepe import CRePE

    f0_extractor = CRePE(device=device, sample_rate=sample_rate, hop_length=window)

    logger.info("Loading HuBERT model...")
    from lib.train.extract_features import load_hubert_model

    hubert_model, _ = load_hubert_model(
        model_path=Path("assets/hubert/hubert_base.pt"), accelerator=accelerator
    )

    if isinstance(audio, Path):
        logger.info(f"Loading audio from {audio}...")
        audio = load_audio(audio, sr=sample_rate)

    # Apply high-pass filter
    logger.info("Applying high-pass filter...")
    audio_16k = resample_audio(audio, orig_sr=sample_rate, target_sr=16000)
    audio_16k = signal.filtfilt(bh, ah, audio_16k).copy()  # Make contiguous copy
    
    # Pad audio
    t_pad = sr * x_pad
    audio_pad = np.pad(audio_16k, (t_pad, t_pad), mode="reflect")
    p_len = audio_pad.shape[0] // window

    logger.info("Extracting F0 and coarse F0...")
    f0 = f0_extractor.compute_f0(audio_16k, p_len=p_len)
    
    # Apply pitch shift
    if f0_up_key != 0:
        f0 *= pow(2, f0_up_key / 12)
    
    f0nsf = f0_extractor.coarse_f0(f0)

    logger.info("Extracting HuBERT features...")
    wav_tensor = torch.from_numpy(audio_pad).unsqueeze(0).float().to(device)
    
    padding_mask = torch.BoolTensor(wav_tensor.shape).fill_(False).to(device)
    
    inputs = {
        "source": wav_tensor,
        "padding_mask": padding_mask,
        "output_layer": 12,
    }
    
    with torch.no_grad():
        logits = hubert_model.extract_features(**inputs)
        feats_tensor: torch.Tensor = logits[0]

    # Check for NaNs
    if torch.isnan(feats_tensor).any():
        logger.warning("Extracted features contain NaNs, which may lead to issues.")
    
    # Store original features for protection
    if protect < 0.5:
        feats0 = feats_tensor.clone()
    
    # Interpolate features (upscale by 2x)
    feats_tensor = F.interpolate(
        feats_tensor.permute(0, 2, 1), scale_factor=2
    ).permute(0, 2, 1)
    
    if protect < 0.5 and 'feats0' in locals():
        feats0 = F.interpolate(
            feats0.permute(0, 2, 1), scale_factor=2
        ).permute(0, 2, 1)
    
    # Adjust lengths to match - features should match p_len
    feat_len = feats_tensor.shape[1]
    if feat_len < p_len:
        p_len = feat_len
    
    # Ensure F0 matches feature length
    f0 = f0[:p_len]
    f0nsf = f0nsf[:p_len]
    
    # Convert to tensors
    f0_tensor = torch.from_numpy(f0).unsqueeze(0).float().to(device)
    f0nsf_tensor = torch.from_numpy(f0nsf).unsqueeze(0).long().to(device)
    
    # Trim features to match F0 length if needed
    if feats_tensor.shape[1] > p_len:
        feats_tensor = feats_tensor[:, :p_len, :]
        if protect < 0.5 and 'feats0' in locals():
            feats0 = feats0[:, :p_len, :]
    
    # Apply consonant protection
    if protect < 0.5 and 'feats0' in locals():
        # Create protection mask based on F0 (voiced/unvoiced)
        pitchff = f0_tensor.clone()
        pitchff[f0_tensor > 0] = 1
        pitchff[f0_tensor < 1] = protect
        pitchff = pitchff.unsqueeze(-1)
        
        # Mix original and current features
        feats_tensor = feats_tensor * pitchff + feats0 * (1 - pitchff)
        feats_tensor = feats_tensor.to(feats0.dtype)
    
    logger.info("Running synthesis...")
    p_len_tensor = torch.tensor([p_len], device=device).long()
    sid_tensor = torch.tensor([0], device=device).long()
    
    with torch.no_grad():
        o, _, _ = net_g.infer(
            phone=feats_tensor,
            phone_lengths=p_len_tensor,
            pitch=f0nsf_tensor,  # Coarse F0 for embedding lookup
            nsff0=f0_tensor,     # Fine F0 for NSF
            sid=sid_tensor,
        )
        # o shape: [batch, 1, time]
        audio_out = o[0, 0].data.cpu().float().numpy()
    
    # Remove padding from output
    t_pad_tgt = sample_rate * x_pad
    if len(audio_out) > 2 * t_pad_tgt:
        audio_out = audio_out[t_pad_tgt:-t_pad_tgt]
    
    # Apply RMS mixing
    if rms_mix_rate != 1:
        logger.info("Applying RMS mixing...")
        audio_out = change_rms(
            original_audio, sample_rate, audio_out, sample_rate, rms_mix_rate
        )
    
    # Resample if needed
    if resample_sr != sample_rate:
        logger.info(f"Resampling from {sample_rate}Hz to {resample_sr}Hz...")
        audio_out = resample_audio(audio_out, orig_sr=sample_rate, target_sr=resample_sr)
    
    # Normalize
    audio_max = np.abs(audio_out).max() / 0.99
    if audio_max > 1:
        audio_out = audio_out / audio_max

    return audio_out

def main():
    from tap import tapify

    tapify(interface_cli)


if __name__ == "__main__":
    main()
