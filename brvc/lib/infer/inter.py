from pathlib import Path
from typing import Optional, Union
from numpy.typing import NDArray
import resampy


from lib.config.v2_config import default_config, ConfigV2
from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid
from lib.utils.audio import load_audio
import numpy as np
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.modules.synthesizer_trn_ms import SynthesizerTrnMsNSFsid
    from accelerate import Accelerator
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    sample_rate: int = 48000,
):
    from accelerate import Accelerator
    import torch

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

    return inference(
        net_g=net_g,
        audio=audio_data,
        sample_rate=sample_rate,
        accelerator=accelerator,
    )


def inference(
    net_g: SynthesizerTrnMsNSFsid,
    audio: NDArray[np.float32],
    accelerator: "Accelerator",
    sample_rate: int = 48000,
) -> NDArray[np.float32]:
    import torch

    device = accelerator.device

    logger.info("Loading F0 extractor...")
    from lib.features.pitch.crepe import CRePE

    f0_extractor = CRePE(device=device, sample_rate=sample_rate)

    logger.info("Loading HuBERT model...")
    from lib.train.extract_features import load_hubert_model

    hubert_model, _ = load_hubert_model(
        model_path=Path("assets/hubert/hubert_base.pt"), accelerator=accelerator
    )

    if isinstance(audio, Path):
        logger.info(f"Loading audio from {audio}...")
        audio = load_audio(audio, sr=sample_rate)

    logger.info("Extracting F0 and coarse F0...")
    f0 = f0_extractor.compute_f0(audio)
    f0nsf = f0_extractor.coarse_f0(audio)

    logger.info("Extracting HuBERT features...")

    resampled_audio = resample_audio(audio, orig_sr=sample_rate, target_sr=16000)
    wav_tensor = torch.from_numpy(resampled_audio).unsqueeze(0).float().to(device)
    
    padding_mask = torch.BoolTensor(wav_tensor.shape).fill_(False).to(device)
    
    inputs = {
        "source": wav_tensor,
        "padding_mask": padding_mask,
        "output_layer": 12,
    }
    
    with torch.no_grad():
        logits = hubert_model.extract_features(**inputs)
        feats_tensor: torch.Tensor = logits[0]

    # Checks
    feats = feats_tensor.squeeze(0).float().cpu().numpy()
    if not np.isnan(feats).any():
        pass
    else:
        logger.warning("Extracted features contain NaNs, which may lead to issues.")
    logger.info("Running synthesis...")
    with torch.no_grad():
        f0_tensor = torch.from_numpy(f0).unsqueeze(0).float().to(device)
        f0nsf_tensor = torch.from_numpy(f0nsf).unsqueeze(0).long().to(device)
        feats_tensor = feats_tensor.to(device)

        audio_out = net_g.infer(
            feats_tensor,
            f0_tensor,
            f0nsf_tensor,
            max_len=None,
            use_noise_scale=0.3,
            use_length_scale=1.0,
        )[0][0, 0].data.cpu().float().numpy()

    return audio_out

def main():
    from tap import tapify

    tapify(interface_cli)


if __name__ == "__main__":
    main()
