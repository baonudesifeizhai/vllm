# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only GLM-ASR model compatible with HuggingFace weights."""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, cast

import numpy as np
import torch
from torch import nn
from transformers import BatchFeature, WhisperFeatureExtractor

from vllm.attention.backends.abstract import AttentionType
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul, get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.glm4 import Glm4Attention
from vllm.model_executor.models.whisper_utils import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.jsontree import json_map_leaves
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsTranscription
from .utils import (
    WeightsMapper,
    _merge_multimodal_embeddings,
    cast_overflow_tensors,
    init_vllm_registered_model,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _compute_post_conv_lengths(audio_lengths: torch.Tensor) -> torch.Tensor:
    """Compute audio lengths after conv layers and reshape."""
    for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
        audio_lengths = (audio_lengths + 2 * padding - kernel_size) // stride + 1
    return (audio_lengths - 4) // 4 + 1


def _extract_audio_array(audio_data: object) -> object:
    """Extract audio array from data (tuple or direct)."""
    return audio_data[0] if isinstance(audio_data, tuple) else audio_data


def _get_audio_dummy_length(feature_extractor: WhisperFeatureExtractor) -> int:
    """Get dummy audio length from feature extractor."""
    return feature_extractor.chunk_length * feature_extractor.sampling_rate


class GlmAsrAudioInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - nmb: Number of mel bins
        - t: Time frames (M)
    """

    input_features: Annotated[
        list[torch.Tensor] | None,
        TensorShape("b", "nmb", "t"),
    ]
    input_features_mask: Annotated[torch.Tensor | None, TensorShape("b", "t")] = None


class GlmAsrMLP(nn.Module):
    """MLP layer for GLM-ASR encoder with gelu activation.

    Reuses the same structure as Glm4vVisionMLP but with GELU activation.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "gelu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only gelu is supported."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = GeluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class GlmAsrEncoderLayer(nn.Module):
    """Encoder layer for GLM-ASR audio encoder.

    Reuses the same structure as Glm4vVisionBlock but with Glm4Attention.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        audio_config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = audio_config.hidden_size

        audio_config.rope_parameters.setdefault("partial_rotary_factor", 0.5)
        self.self_attn = Glm4Attention(
            config=audio_config,
            hidden_size=self.hidden_size,
            num_heads=audio_config.num_attention_heads,
            num_kv_heads=audio_config.num_key_value_heads,
            max_position=audio_config.max_position_embeddings,
            head_dim=getattr(audio_config, "head_dim", None),
            qkv_bias=getattr(audio_config, "attention_bias", False),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            attn_type=AttentionType.ENCODER,
        )

        self.mlp = GlmAsrMLP(
            hidden_size=self.hidden_size,
            intermediate_size=audio_config.intermediate_size,
            hidden_act=audio_config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        eps = getattr(audio_config, "rms_norm_eps", 1e-5)
        self.norm1 = RMSNorm(self.hidden_size, eps=eps)
        self.norm2 = RMSNorm(self.hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = cast_overflow_tensors(hidden_states)

        return hidden_states


class GlmAsrEncoder(nn.Module):
    """Audio encoder for GLM-ASR, reusing WhisperEncoder conv logic."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        audio_config = config.audio_config
        embed_dim = audio_config.hidden_size

        self.num_mel_bins = audio_config.num_mel_bins
        self.max_source_positions = audio_config.max_position_embeddings

        # Conv1d layers (same as WhisperEncoder)
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.total_stride = self.conv1.stride[0] * self.conv2.stride[0]

        # Encoder layers
        self.start_layer, self.end_layer, self.layers = make_layers(
            audio_config.num_hidden_layers,
            lambda prefix: GlmAsrEncoderLayer(
                vllm_config=vllm_config,
                audio_config=audio_config,
                prefix=f"{prefix}.layers",
            ),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward_conv(self, input_features: list[torch.Tensor]) -> torch.Tensor:
        """Forward conv layers, reusing WhisperEncoder logic without pos embeddings."""
        hidden_states = []
        input_is_batched = False
        for features in input_features:
            embeds = nn.functional.gelu(self.conv1(features))
            embeds = nn.functional.gelu(self.conv2(embeds))
            embeds = embeds.transpose(-1, -2)
            hidden_states.append(embeds)
            input_is_batched = embeds.ndim > 2
        if input_is_batched:
            hidden_states = torch.cat(hidden_states)
        else:
            hidden_states = torch.stack(hidden_states)
        return hidden_states

    def forward(self, input_features: list[torch.Tensor]) -> torch.Tensor:
        hidden_states = self.forward_conv(input_features)
        batch_size, seq_len, _ = hidden_states.shape
        positions = (
            torch.arange(seq_len, device=hidden_states.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        return self.layer_norm(hidden_states)


class GlmAsrProjector(nn.Module):
    """Projector to connect audio encoder and text decoder."""

    def __init__(
        self,
        audio_intermediate_size: int,
        text_hidden_size: int,
        projector_hidden_act: str = "gelu",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        # Reuse ColumnParallelLinear for TP support
        self.linear = ColumnParallelLinear(
            input_size=audio_intermediate_size,
            output_size=text_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )
        # Reuse get_act_fn
        self.act = get_act_fn(projector_hidden_act)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear(audio_features)
        hidden_states = self.act(hidden_states)
        return hidden_states


class GlmAsrProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        return self.get_hf_processor(**kwargs).feature_extractor

    def get_num_audio_tokens(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        """Calculate number of audio tokens from audio lengths."""
        return _compute_post_conv_lengths(audio_lengths)


class GlmAsrDummyInputsBuilder(BaseDummyInputsBuilder[GlmAsrProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        # Use <|pad|> as audio token placeholder
        return "<|pad|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()
        audio_len = _get_audio_dummy_length(feature_extractor)
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


class GlmAsrMultiModalProcessor(BaseMultiModalProcessor[GlmAsrProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
            mm_data = {"audio": mm_data.pop("audios")}
            mm_kwargs = {**mm_kwargs, "sampling_rate": feature_extractor.sampling_rate}
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(input_features=MultiModalFieldConfig.batched("audio"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        feature_extractor = self.info.get_feature_extractor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer(**hf_processor_mm_kwargs)
        audio_token = getattr(processor, "audio_token", "<|pad|>")
        audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)

        audios = mm_items.get_items("audio", AudioProcessorItems)
        audio_lengths = []
        for i in range(len(audios)):
            audio_array = audios.get(i)
            audio_lengths.append(
                math.ceil(len(audio_array) / feature_extractor.hop_length)
            )

        num_tokens = (
            self.info.get_num_audio_tokens(torch.tensor(audio_lengths))
            if audio_lengths
            else torch.tensor([0])
        )

        updates = []
        for i in range(len(audios)):
            updates.append(
                PromptReplacement(
                    modality="audio",
                    target=[audio_token_id],
                    replacement=[audio_token_id] * num_tokens[i].item(),
                )
            )
        return updates


@MULTIMODAL_REGISTRY.register_processor(
    GlmAsrMultiModalProcessor,
    info=GlmAsrProcessingInfo,
    dummy_inputs=GlmAsrDummyInputsBuilder,
)
class GlmAsrForConditionalGeneration(
    nn.Module, SupportsTranscription, SupportsMultiModal
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "audio_tower.": "audio_encoder.",
            "multi_modal_projector.": "projector.",
        },
        orig_to_new_substr={
            ".mlp.fc2": ".mlp.down_proj",
        },
    )

    supports_transcription_only = True
    supports_segment_timestamp = False
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        return language or "en"

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        decoder_prompt = (
            f"<|prev|>{request_prompt}" if request_prompt else ""
        ) + f"<|startoftranscript|><|{language}|><|{task_type}|><|notimestamps|>"
        return cast(
            PromptType,
            {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {"audio": (audio, stt_config.sample_rate)},
                },
                "decoder_prompt": decoder_prompt,
            },
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return None

        raise ValueError("Only audio modality is supported")

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)

        return SpeechToTextConfig(
            max_audio_clip_s=processor.feature_extractor.chunk_length,
            sample_rate=processor.feature_extractor.sampling_rate,
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        hop_length = cached_processor_from_config(
            model_config
        ).feature_extractor.hop_length
        return math.ceil(audio_duration_s * stt_config.sample_rate / hop_length)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        audio_config = config.audio_config
        text_config = config.text_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.dtype = vllm_config.model_config.dtype

        # Audio encoder
        self.audio_encoder = GlmAsrEncoder(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "audio_encoder")
        )

        # Projector
        projector_hidden_act = getattr(config, "projector_hidden_act", "gelu")
        self.projector = GlmAsrProjector(
            audio_intermediate_size=audio_config.intermediate_size,
            text_hidden_size=text_config.hidden_size,
            projector_hidden_act=projector_hidden_act,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "projector"),
        )

        # Language model (reuse Glm4ForCausalLM via init_vllm_registered_model)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Glm4ForCausalLM"],
        )

    def get_audio_features(
        self, input_features: torch.Tensor, input_features_mask: torch.Tensor
    ) -> torch.Tensor:
        audio_outputs = self.audio_encoder(input_features)
        batch_size = audio_outputs.shape[0]
        audio_hidden_states = audio_outputs.reshape(
            batch_size, -1, self.config.audio_config.intermediate_size
        )
        audio_embeds = self.projector(audio_hidden_states)

        audio_lengths = input_features_mask.sum(-1)
        post_lengths = _compute_post_conv_lengths(audio_lengths)

        valid_mask = (
            torch.arange(audio_embeds.shape[1], device=audio_embeds.device)[None, :]
            < post_lengths[:, None]
        )
        return audio_embeds[valid_mask]

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=kwargs.get("inputs_embeds"),
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if not audio_input or not audio_input["input_features"]:
            return []

        input_features = audio_input["input_features"]
        input_features_mask = audio_input["input_features_mask"]

        if isinstance(input_features, list):
            max_len = max(f.shape[-1] for f in input_features)
            padded_features = [
                nn.functional.pad(f, (0, max_len - f.shape[-1])) for f in input_features
            ]
            masks = [
                nn.functional.pad(
                    torch.ones(f.shape[-1], dtype=torch.bool, device=f.device),
                    (0, max_len - f.shape[-1]),
                    value=False,
                )
                for f in input_features
            ]
            input_features = torch.stack(padded_features)
            input_features_mask = torch.stack(masks)

        audio_embeds = self.get_audio_features(input_features, input_features_mask)

        audio_lengths = input_features_mask.sum(-1)
        post_lengths = _compute_post_conv_lengths(audio_lengths)
        return torch.split(audio_embeds, post_lengths.tolist())

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.embed_input_ids(input_ids)
        if not multimodal_embeddings:
            return inputs_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds, multimodal_embeddings, is_multimodal
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> GlmAsrAudioInputs | None:
        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        if input_features:
            input_features = json_map_leaves(lambda x: x.to(self.dtype), input_features)
        return (
            GlmAsrAudioInputs(
                input_features=input_features, input_features_mask=input_features_mask
            )
            if input_features
            else None
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # Encoder attention: q/k/v_proj -> qkv_proj
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            # Encoder MLP: fc1 -> gate_up_proj (load twice)
            (".mlp.gate_up_proj", ".mlp.fc1", 0),  # gate
            (".mlp.gate_up_proj", ".mlp.fc1", 1),  # up
        ]

        weights = list(self.hf_to_vllm_mapper.apply(weights))
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Handle fc1: load twice into gate_up_proj
            if ".mlp.fc1" in name and not name.endswith(".bias"):
                gate_up_name = name.replace(".mlp.fc1", ".mlp.gate_up_proj")
                if gate_up_name in params_dict:
                    param = params_dict[gate_up_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, loaded_shard_id=0)
                    weight_loader(param, loaded_weight, loaded_shard_id=1)
                    loaded_params.add(gate_up_name)
                    continue

            # Handle QKV and other stacked params
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params
