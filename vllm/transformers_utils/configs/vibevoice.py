# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from vibevoice.modular.configuration_vibevoice import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceSemanticTokenizerConfig,
)


def _convert_dtype_to_string(config_dict: dict[str, Any]) -> dict[str, Any]:
    if "torch_dtype" in config_dict and config_dict["torch_dtype"] is not None:
        dtype = config_dict["torch_dtype"]
        if isinstance(dtype, torch.dtype):
            config_dict["torch_dtype"] = str(dtype).replace("torch.", "")
    return config_dict


class VibeVoiceASRConfig(PretrainedConfig):
    model_type = "vibevoice"
    is_composition = True
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig,
        "semantic_tokenizer_config": VibeVoiceSemanticTokenizerConfig,
        "decoder_config": Qwen2Config,
    }

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        acoustic_tokenizer_config: (
            dict[str, Any] | VibeVoiceAcousticTokenizerConfig | None
        ) = None,
        semantic_tokenizer_config: (
            dict[str, Any] | VibeVoiceSemanticTokenizerConfig | None
        ) = None,
        decoder_config: dict[str, Any] | Qwen2Config | None = None,
        diffusion_head_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs["_attn_implementation_autoset"] = False

        if acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = VibeVoiceAcousticTokenizerConfig()
        elif isinstance(acoustic_tokenizer_config, dict):
            acoustic_tokenizer_config["model_type"] = "vibevoice_acoustic_tokenizer"
            self.acoustic_tokenizer_config = VibeVoiceAcousticTokenizerConfig(
                **acoustic_tokenizer_config
            )
        else:
            self.acoustic_tokenizer_config = acoustic_tokenizer_config

        if semantic_tokenizer_config is None:
            self.semantic_tokenizer_config = VibeVoiceSemanticTokenizerConfig()
        elif isinstance(semantic_tokenizer_config, dict):
            semantic_tokenizer_config["model_type"] = "vibevoice_semantic_tokenizer"
            self.semantic_tokenizer_config = VibeVoiceSemanticTokenizerConfig(
                **semantic_tokenizer_config
            )
        else:
            self.semantic_tokenizer_config = semantic_tokenizer_config

        if decoder_config is None:
            self.decoder_config = Qwen2Config()
        elif isinstance(decoder_config, dict):
            if decoder_config.get("model_type", "") == "qwen2":
                self.decoder_config = Qwen2Config(**decoder_config)
            else:
                raise ValueError(
                    "Unsupported decoder model type: "
                    f"{decoder_config.get('model_type', '')}"
                )
        else:
            self.decoder_config = decoder_config

        self.diffusion_head_config = diffusion_head_config

        self.acoustic_vae_dim = getattr(self.acoustic_tokenizer_config, "vae_dim", 64)
        self.semantic_vae_dim = getattr(self.semantic_tokenizer_config, "vae_dim", 128)

        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False):
        return self.decoder_config

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        return _convert_dtype_to_string(output)

    @property
    def vocab_size(self) -> int:
        return self.decoder_config.vocab_size

    @property
    def num_attention_heads(self) -> int:
        return self.decoder_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.decoder_config.num_key_value_heads

    @property
    def hidden_size(self) -> int:
        return self.decoder_config.hidden_size

    @property
    def num_hidden_layers(self) -> int:
        return self.decoder_config.num_hidden_layers

    @property
    def head_dim(self) -> int:
        return getattr(
            self.decoder_config,
            "head_dim",
            self.hidden_size // self.num_attention_heads,
        )


__all__ = [
    "VibeVoiceAcousticTokenizerConfig",
    "VibeVoiceSemanticTokenizerConfig",
    "VibeVoiceASRConfig",
]
