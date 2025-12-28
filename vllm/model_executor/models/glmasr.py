# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.glmasr_utils import (
    DEFAULT_CONV_PARAMS,
    DEFAULT_MAX_AUDIO_LEN_S,
    DEFAULT_MERGE_FACTOR,
    DEFAULT_SAMPLE_RATE,
    _flatten_audio_features_by_length,
    _get_audio_output_lengths_for_tower,
    _get_num_features_for_item,
    _group_audio_embeddings,
    _normalize_chunk_counts,
    extract_glmasr_features,
)
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityData,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix

DEFAULT_AUDIO_TOKEN = "<|pad|>"
DEFAULT_CHUNK_LENGTH_S = 30
DEFAULT_N_FFT = 400
DEFAULT_HOP_LENGTH = 160


class GlmAsrFeatureInputs(TensorSchema):
    """
    Dimensions:
        - num_chunks: Number of audio chunks (flattened)
        - nmb: Number of mel bins
        - num_audios: Number of original audio files
    """

    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("num_chunks", "nmb", "chunk_length", dynamic_dims={"chunk_length"}),
    ]
    feature_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("num_chunks", "chunk_length", dynamic_dims={"chunk_length"}),
    ]
    chunk_counts: Annotated[
        torch.Tensor,
        TensorShape("num_audios"),
    ]


class GlmAsrEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size
        - naf: Number of audio features
        - hs: Hidden size (must match the hidden size of language model
          backbone)
    """

    type: Literal["audio_embeds"] = "audio_embeds"
    audio_embeds: Annotated[
        list[torch.Tensor],
        TensorShape("bn", "naf", "hs", dynamic_dims={"naf"}),
    ]


GlmAsrInputs: TypeAlias = GlmAsrFeatureInputs | GlmAsrEmbeddingInputs


def _sinusoids(
    length: int, channels: int, max_timescale: float = 10000.0
) -> torch.Tensor:
    if channels % 2 != 0:
        raise ValueError("channels must be divisible by 2")
    log_timescale_increment = torch.log(torch.tensor(max_timescale)) / (
        channels // 2 - 1
    )
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1)
    return q, k


class GlmAsrRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        partial_rotary_factor: float,
    ) -> None:
        super().__init__()
        rotary_dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin


class GlmAsrSelfAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, rope: GlmAsrRotaryEmbedding
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.rope = rope

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(seq_len, hidden_states.device, hidden_states.dtype)
        cos = cos[None, None, :, : self.rope.rotary_dim]
        sin = sin[None, None, :, : self.rope.rotary_dim]
        q, k = _apply_rotary(q, k, cos, sin)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        return self.o_proj(attn)


class GlmAsrMLP(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, hidden_act: str
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.act = get_act_fn(hidden_act)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class GlmAsrEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        hidden_act: str,
        rope: GlmAsrRotaryEmbedding,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self.self_attn = GlmAsrSelfAttention(hidden_size, num_heads, rope)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = GlmAsrMLP(hidden_size, intermediate_size, hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class GlmAsrAudioTower(nn.Module):
    def __init__(self, audio_config: Any) -> None:
        super().__init__()
        self.audio_config = audio_config
        hidden_size = audio_config.hidden_size
        self.num_mel_bins = audio_config.num_mel_bins
        self.max_position_embeddings = audio_config.max_position_embeddings

        self.conv1 = nn.Conv1d(self.num_mel_bins, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=3, stride=2, padding=1
        )
        self.total_stride = self.conv1.stride[0] * self.conv2.stride[0]

        self.embed_positions = nn.Embedding(self.max_position_embeddings, hidden_size)
        with torch.no_grad():
            self.embed_positions.weight.copy_(
                _sinusoids(self.max_position_embeddings, hidden_size)
            )

        rope_params = getattr(audio_config, "rope_parameters", {})
        rope_theta = float(rope_params.get("rope_theta", 10000.0))
        partial_rotary_factor = float(rope_params.get("partial_rotary_factor", 1.0))
        rope = GlmAsrRotaryEmbedding(
            head_dim=getattr(
                audio_config,
                "head_dim",
                hidden_size // audio_config.num_attention_heads,
            ),
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )

        intermediate_size = getattr(audio_config, "intermediate_size", None)
        if intermediate_size is None or intermediate_size < hidden_size * 4:
            intermediate_size = hidden_size * 4

        self.layers = nn.ModuleList(
            [
                GlmAsrEncoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=audio_config.num_attention_heads,
                    hidden_act=audio_config.hidden_act,
                    rope=rope,
                    eps=getattr(audio_config, "rms_norm_eps", 1e-5),
                )
                for _ in range(audio_config.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(
            hidden_size, eps=getattr(audio_config, "rms_norm_eps", 1e-5)
        )

    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = input_lengths
        for padding, kernel_size, stride in DEFAULT_CONV_PARAMS:
            lengths = (lengths + 2 * padding - kernel_size) // stride + 1
        return input_lengths, lengths

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.nn.functional.gelu(self.conv1(input_features))
        hidden_states = torch.nn.functional.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.transpose(1, 2)

        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        hidden_states = hidden_states + self.embed_positions(positions)

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class GlmAsrMultiModalProjector(nn.Module):
    def __init__(self, config: Any, audio_feature_dim: int | None = None):
        super().__init__()
        if audio_feature_dim is None:
            audio_feature_dim = config.audio_config.hidden_size
        self.linear_1 = nn.Linear(
            audio_feature_dim,
            config.text_config.hidden_size * 2,
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size * 2,
            config.text_config.hidden_size,
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class GlmAsrProcessorAdapter:
    def __init__(self, config: Any):
        self.config = config
        self.audio_token = DEFAULT_AUDIO_TOKEN
        self.audio_token_id = getattr(config, "audio_token_id", None)
        self.sampling_rate = getattr(config, "sampling_rate", DEFAULT_SAMPLE_RATE)
        self.n_fft = getattr(config, "n_fft", DEFAULT_N_FFT)
        self.hop_length = getattr(config, "hop_length", DEFAULT_HOP_LENGTH)
        self.n_mels = getattr(config.audio_config, "num_mel_bins", 128)
        self.chunk_length_s = int(
            getattr(config, "chunk_length", DEFAULT_CHUNK_LENGTH_S)
        )
        self.max_audio_len_s = int(
            getattr(config, "max_audio_len", DEFAULT_MAX_AUDIO_LEN_S)
        )


class GlmAsrProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Any:
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **_: object) -> GlmAsrProcessorAdapter:
        return GlmAsrProcessorAdapter(self.get_hf_config())

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class GlmAsrDummyInputsBuilder(BaseDummyInputsBuilder[GlmAsrProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        processor = self.info.get_hf_processor()
        return processor.audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        processor = self.info.get_hf_processor()
        sampling_rate = processor.sampling_rate
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None

        audio_len = int(processor.max_audio_len_s * sampling_rate)

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


def _glmasr_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    chunk_counts = hf_inputs.get("chunk_counts")
    if chunk_counts is not None:
        return dict(
            audio_embeds=MultiModalFieldConfig.batched("audio"),
            input_features=MultiModalFieldConfig.flat_from_sizes(
                "audio", chunk_counts, dim=0
            ),
            feature_attention_mask=MultiModalFieldConfig.flat_from_sizes(
                "audio", chunk_counts, dim=0
            ),
            chunk_counts=MultiModalFieldConfig.batched("audio"),
        )
    return dict(
        audio_embeds=MultiModalFieldConfig.batched("audio"),
        input_features=MultiModalFieldConfig.batched("audio"),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        chunk_counts=MultiModalFieldConfig.batched("audio"),
    )


class GlmAsrMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[Any],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_glmasr_field_config,
            )
        return super()._parse_audio_data(data)


class GlmAsrMultiModalProcessor(BaseMultiModalProcessor[GlmAsrProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        processor = self.info.get_hf_processor()
        return GlmAsrMultiModalDataParser(target_sr=processor.sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, torch.Tensor]:
        if "audios" in mm_data:
            mm_data = dict(mm_data)
            mm_data["audio"] = mm_data.pop("audios")

        audio_list = mm_data.get("audio", [])
        if not isinstance(audio_list, list):
            audio_list = [audio_list]

        prompt_ids = self.info.get_tokenizer().encode(prompt)
        prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)

        if not audio_list:
            return {"input_ids": torch.tensor([prompt_ids], dtype=torch.long)}

        processor = self.info.get_hf_processor()
        input_features, feature_attention_mask, chunk_counts = extract_glmasr_features(
            audio_list,
            sampling_rate=processor.sampling_rate,
            n_fft=processor.n_fft,
            hop_length=processor.hop_length,
            n_mels=processor.n_mels,
            chunk_length_s=processor.chunk_length_s,
            max_audio_len_s=processor.max_audio_len_s,
        )

        return {
            "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "chunk_counts": chunk_counts,
        }

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, torch.Tensor],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _glmasr_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor()
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        config = self.info.get_hf_config()

        audio_token = processor.audio_token
        audio_token_id = vocab.get(audio_token)
        if audio_token_id is None:
            audio_token_id = int(config.audio_token_id)

        merge_factor = getattr(config, "merge_factor", DEFAULT_MERGE_FACTOR)
        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        chunk_counts = out_mm_data.get("chunk_counts")
        conv_params = getattr(config, "conv_params", DEFAULT_CONV_PARAMS)
        audio_embeds = out_mm_data.get("audio_embeds")

        def get_replacement_glmasr(item_idx: int):
            num_features = _get_num_features_for_item(
                feature_attention_mask,
                chunk_counts,
                item_idx,
                audio_embeds,
                merge_factor,
                conv_params,
            )

            if num_features == 0:
                raise ValueError("Audio is too short")

            audio_tokens = [audio_token_id] * int(num_features)
            return PromptUpdateDetails.select_token_id(
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=get_replacement_glmasr,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    GlmAsrMultiModalProcessor,
    info=GlmAsrProcessingInfo,
    dummy_inputs=GlmAsrDummyInputsBuilder,
)
class GlmAsrForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        audio_config = config.audio_config
        self.audio_tower = GlmAsrAudioTower(audio_config)
        self.audio_feature_dim = audio_config.hidden_size
        self.multi_modal_projector = GlmAsrMultiModalProjector(
            config, audio_feature_dim=self.audio_feature_dim
        )
        self.quant_config = quant_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["LlamaForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|begin_of_audio|><|pad|><|end_of_audio|>"

        raise ValueError("Only audio modality is supported")

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector="multi_modal_projector.",
            tower_model="audio_tower.",
        )

    def _parse_and_validate_audio_input(self, **kwargs: object) -> GlmAsrInputs | None:
        audio_embeds = kwargs.pop("audio_embeds", None)
        if audio_embeds is not None:
            return GlmAsrEmbeddingInputs(type="audio_embeds", audio_embeds=audio_embeds)

        input_features = kwargs.pop("input_features", None)
        if input_features is None:
            return None

        return GlmAsrFeatureInputs(
            type="audio_features",
            input_features=input_features,
            feature_attention_mask=kwargs.pop("feature_attention_mask", None),
            chunk_counts=kwargs.pop("chunk_counts", None),
        )

    def _process_audio_input(
        self, audio_input: GlmAsrInputs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return tuple(audio_input["audio_embeds"])

        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]

        if isinstance(input_features, list):
            input_features = torch.cat(input_features, dim=0)
            feature_attention_mask = torch.cat(feature_attention_mask, dim=0)

        num_chunks = input_features.shape[0]
        chunk_counts = _normalize_chunk_counts(
            audio_input.get("chunk_counts"), num_chunks=num_chunks
        )

        audio_hidden_states = self.audio_tower(input_features)
        audio_hidden_states = audio_hidden_states.reshape(
            num_chunks,
            -1,
            self.audio_feature_dim,
        )
        audio_features = self.multi_modal_projector(audio_hidden_states)

        merge_factor = getattr(self.config, "merge_factor", DEFAULT_MERGE_FACTOR)
        conv_params = getattr(self.config, "conv_params", DEFAULT_CONV_PARAMS)

        audio_output_lengths = _get_audio_output_lengths_for_tower(
            self.audio_tower,
            feature_attention_mask.sum(-1),
            merge_factor,
            conv_params,
        )

        masked_audio_features = _flatten_audio_features_by_length(
            audio_features, audio_output_lengths
        )

        chunk_embeddings = torch.split(
            masked_audio_features, audio_output_lengths.flatten().tolist()
        )
        return _group_audio_embeddings(chunk_embeddings, chunk_counts)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        masked_audio_features = self._process_audio_input(audio_input)
        return masked_audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["audio_tower.embed_positions"]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights)
