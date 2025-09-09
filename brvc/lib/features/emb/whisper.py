from transformers import WhisperFeatureExtractor, BatchFeature
import torch

model_name = "openai/whisper-large-v3-turbo"

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)


def extract(waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    input_features: torch.Tensor = feature_extractor(
        waveform, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features

    return input_features


def test():

    from datasets import load_dataset

    # Load a sample dataset or your own audio file
    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    sample = dataset[2]["audio"]

    print(sample)
    print(
        extract(sample["array"], sampling_rate=sample["sampling_rate"]).shape
    )  # (batch_size, feature_size, sequence_length)


if __name__ == "__main__":
    test()
