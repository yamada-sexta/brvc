# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Hubert checkpoint."""

import argparse
import json
import os
from typing import Optional

import fairseq
import torch
from fairseq.data import Dictionary

from transformers import (
    HubertConfig,
    HubertForCTC,
    HubertModel,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    HubertForSequenceClassification,
    Wav2Vec2Processor,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.batch_norm",
    "encoder.pos_conv.1": "encoder.pos_conv_embed.conv",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
}


def set_recursively(hf_pointer, key: str, value: torch.Tensor, full_name: str, weight_type: Optional[str]):
    # 1. Navigate to the target module/tensor
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    # 2. Check if the target weight attribute exists and is not None
    # Use getattr with a default value of None to check for existence
    if weight_type is not None:
        target_weight_attr = getattr(hf_pointer, weight_type, None)
    else:
        target_weight_attr = hf_pointer

    # CRITICAL FIX: If the target weight (e.g., bias) is None, skip the transfer.
    if target_weight_attr is None:
        logger.info(
            f"Skipping {key}.{weight_type} because the corresponding weight is None in the HF model (likely bias=False)."
        )
        return

    # 3. Shape check (now on the existing target tensor)
    hf_shape = target_weight_attr.shape

    assert hf_shape == value.shape, (
        f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
        f" {value.shape} for {full_name}"
    )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "running_mean":
        hf_pointer.running_mean.data = value
    elif weight_type == "running_var":
        hf_pointer.running_var.data = value
    elif weight_type == "num_batches_tracked":
        hf_pointer.num_batches_tracked.data = value
    else:
        hf_pointer.data = value

    logger.info(
        f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}."
    )

from fairseq.models.hubert.hubert import HubertModel as FairseqHubertModel
def recursively_load_weights(fairseq_model: FairseqHubertModel, hf_model: HubertModel):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()
    feature_extractor = hf_model.feature_extractor

    for name, value in fairseq_dict.items():
        logger.info(f"Processing {name}...")
        is_used = False
        
        # Skip training-specific weights that don't exist in the base HuBERT model
        if name in ["label_embs_concat", "final_proj.weight", "final_proj.bias"]:
            # logger.info(f"Skipping training-specific weight: {name}")
            unused_weights.append(name)
            continue
        
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            for key, mapped_key in MAPPING.items():
                # mapped_key = (
                #     "hubert." + mapped_key
                #     if (is_finetuned and mapped_key != "lm_head")
                #     else mapped_key
                # )
                

                if key in name or (
                    key.split("w2v_model.")[-1] == name.split(".")[0]
                    # and not is_finetuned
                ):
                    is_used = True
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "weight" in name:
                        weight_type = "weight"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "running_mean" in name:
                        weight_type = "running_mean"
                    elif "running_var" in name:
                        weight_type = "running_var"
                    elif "num_batches_tracked" in name:
                        weight_type = "num_batches_tracked"
                    else:
                        weight_type = None
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


def load_conv_layer(
    full_name, value, feature_extractor, unused_weights, use_group_norm
):
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    if type_id == 0:
        if "bias" in name:
            assert (
                value.shape
                == feature_extractor.conv_layers[layer_id].conv.bias.data.shape
            ), (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(
                f"Feat extract conv layer {layer_id} was initialized from {full_name}."
            )
        elif "weight" in name:
            assert (
                value.shape
                == feature_extractor.conv_layers[layer_id].conv.weight.data.shape
            ), (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(
                f"Feat extract conv layer {layer_id} was initialized from {full_name}."
            )
    elif (type_id == 2 and not use_group_norm) or (
        type_id == 2 and layer_id == 0 and use_group_norm
    ):
        if "bias" in name:
            assert (
                value.shape
                == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape
            ), (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(
                f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}."
            )
        elif "weight" in name:
            assert (
                value.shape
                == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape
            ), (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(
                f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}."
            )
    else:
        unused_weights.append(full_name)


@torch.no_grad()
def convert_hubert_checkpoint(
    checkpoint_path: str,
    pytorch_dump_folder_path: str,
    config_path: Optional[str] = None,
    # dict_path: Optional[str] = None,
    # is_finetuned=False,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    from fairseq import checkpoint_utils
    model, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    if saved_cfg is None:
        raise ValueError("Could not find model configuration.")
    if config_path is not None:
        config = HubertConfig.from_pretrained(config_path)
    else:
        config = HubertConfig(
            hidden_size=saved_cfg.model.encoder_embed_dim,
            num_hidden_layers=saved_cfg.model.encoder_layers,
            num_attention_heads=saved_cfg.model.encoder_attention_heads,
            intermediate_size=saved_cfg.model.encoder_ffn_embed_dim,
            hidden_act=saved_cfg.model.activation_fn,
        )
    
    hf_wav2vec = HubertModel(config)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=False,
        return_attention_mask=False,
    )
    
    print("Saved cfg:")
    print(saved_cfg)
    model = model[0].eval()

    recursively_load_weights(model, hf_wav2vec)

    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

from fairseq.data.dictionary import Dictionary
from torch.serialization import safe_globals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="assets/hf/hubert_base",
        type=str,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument(
        "--checkpoint_path", default="assets/hubert/hubert_base.pt", type=str, help="Path to fairseq checkpoint"
    )
    args = parser.parse_args()
    with safe_globals([Dictionary]):
        convert_hubert_checkpoint(
            args.checkpoint_path,
            args.pytorch_dump_folder_path,
            # args.config_path,
            # args.dict_path,
            # not args.not_finetuned,
        )
