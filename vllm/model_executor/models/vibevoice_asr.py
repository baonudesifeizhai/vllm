# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only VibeVoice-ASR model compatible with HF weights."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.whisper.tokenization_whisper import LANGUAGES

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.interfaces_base import VllmModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.configs.vibevoice import VibeVoiceASRConfig

logger = init_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text "
    "output in JSON format."
)
SPEECH_START_TOKEN = "<|speech_start|>"
SPEECH_PAD_TOKEN = "<|speech_pad|>"
SPEECH_END_TOKEN = "<|speech_end|>"
DEFAULT_SAMPLE_RATE = 24_000


def _get_encoder_ratios(hf_config: Any) -> list[int] | None:
    acoustic_cfg = getattr(hf_config, "acoustic_tokenizer_config", None)
    if acoustic_cfg is None:
        return None
    if isinstance(acoustic_cfg, dict):
        ratios = acoustic_cfg.get("encoder_ratios")
    else:
        ratios = getattr(acoustic_cfg, "encoder_ratios", None)
    if not ratios:
        return None
    return [int(r) for r in ratios]


def _get_speech_tok_compress_ratio(hf_config: Any) -> int:
    ratios = _get_encoder_ratios(hf_config)
    if ratios:
        ratio = 1
        for r in ratios:
            ratio *= int(r)
        return ratio
    return 3200


def _get_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Token {token!r} is not a single token in tokenizer.")
        token_id = token_ids[0]
    return token_id


class AudioNormalizer:
    def __init__(self, target_dB_FS: float = -25, eps: float = 1e-6):
        self.target_dB_FS = target_dB_FS
        self.eps = eps

    def tailor_dB_FS(self, audio: np.ndarray) -> tuple[np.ndarray, float, float]:
        rms = np.sqrt(np.mean(audio**2))
        scalar = 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
        return audio * scalar, rms, scalar

    def avoid_clipping(self, audio: np.ndarray, scalar: float | None = None):
        if scalar is None:
            max_val = np.max(np.abs(audio))
            scalar = max_val + self.eps if max_val > 1.0 else 1.0
        return audio / scalar, scalar

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio, _, _ = self.tailor_dB_FS(audio)
        audio, _ = self.avoid_clipping(audio)
        return audio


class VibeVoiceASRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> VibeVoiceASRConfig:
        return self.ctx.get_hf_config(VibeVoiceASRConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_sample_rate(self) -> int:
        return DEFAULT_SAMPLE_RATE

    def get_target_channels(self) -> int:
        return 1

    def get_speech_tok_compress_ratio(self) -> int:
        return _get_speech_tok_compress_ratio(self.get_hf_config())


class VibeVoiceASRDummyInputsBuilder(
    BaseDummyInputsBuilder[VibeVoiceASRProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        if num_audios <= 0:
            return ""
        return " ".join([SPEECH_PAD_TOKEN] * num_audios)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None
        audio_len = int(self.info.get_sample_rate())
        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


class VibeVoiceASRMultiModalDataParser(MultiModalDataParser):
    def __init__(self, *, target_sr: int, target_channels: int) -> None:
        super().__init__(target_sr=target_sr, target_channels=target_channels)


class VibeVoiceASRMultiModalProcessor(
    BaseMultiModalProcessor[VibeVoiceASRProcessingInfo]
):
    def __init__(
        self,
        info: VibeVoiceASRProcessingInfo,
        dummy_inputs: BaseDummyInputsBuilder[VibeVoiceASRProcessingInfo],
        *,
        cache: Any | None = None,
    ) -> None:
        super().__init__(info, dummy_inputs, cache=cache)
        self._normalizer = AudioNormalizer()

    def _get_data_parser(self) -> MultiModalDataParser:
        return VibeVoiceASRMultiModalDataParser(
            target_sr=self.info.get_sample_rate(),
            target_channels=self.info.get_target_channels(),
        )

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if audio.size == 0:
            return audio
        return self._normalizer(audio)

    def _build_speech_inputs(
        self, audios: Sequence[np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not audios:
            return (
                torch.empty((0, 0), dtype=torch.float32),
                torch.empty((0, 0), dtype=torch.bool),
                torch.empty((0,), dtype=torch.long),
            )
        normalized = [self._normalize_audio(audio) for audio in audios]
        compress_ratio = self.info.get_speech_tok_compress_ratio()
        vae_tok_lens = [
            max(1, int(math.ceil(len(audio) / compress_ratio))) for audio in normalized
        ]
        max_audio_len = max(len(audio) for audio in normalized)
        max_vae_len = max(vae_tok_lens)

        speech_tensors = torch.zeros(
            (len(normalized), max_audio_len), dtype=torch.float32
        )
        speech_masks = torch.zeros((len(normalized), max_vae_len), dtype=torch.bool)

        for i, (audio, vae_len) in enumerate(zip(normalized, vae_tok_lens)):
            if len(audio) > 0:
                speech_tensors[i, : len(audio)] = torch.from_numpy(audio)
            speech_masks[i, :vae_len] = True

        return (
            speech_tensors,
            speech_masks,
            torch.tensor(vae_tok_lens, dtype=torch.long),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        if "audios" in mm_data:
            mm_data = dict(mm_data)
            mm_data["audio"] = mm_data.pop("audios")

        audios = mm_data.get("audio", [])
        audio_list = [audios] if audios and not isinstance(audios, list) else audios

        if not audio_list:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        speech_tensors, speech_masks, vae_tok_lens = self._build_speech_inputs(
            audio_list
        )

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        return BatchFeature(
            {
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
                "speech_tensors": speech_tensors,
                "speech_masks": speech_masks,
                "vae_tok_lens": vae_tok_lens,
            },
            tensor_type="pt",
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "speech_tensors": MultiModalFieldConfig.batched("audio"),
            "speech_masks": MultiModalFieldConfig.batched("audio"),
            "vae_tok_lens": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        speech_start_id = _get_token_id(tokenizer, SPEECH_START_TOKEN)
        speech_pad_id = _get_token_id(tokenizer, SPEECH_PAD_TOKEN)
        speech_end_id = _get_token_id(tokenizer, SPEECH_END_TOKEN)

        out_mm_data = out_mm_kwargs.get_data()
        vae_tok_lens = out_mm_data.get("vae_tok_lens")

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            if vae_tok_lens is None:
                raise ValueError("Missing vae_tok_lens for audio placeholder sizing.")

            if isinstance(vae_tok_lens, torch.Tensor):
                num_tokens = int(vae_tok_lens[item_idx].item())
            else:
                num_tokens = int(vae_tok_lens[item_idx])

            if num_tokens <= 0:
                raise ValueError("Audio token length must be positive.")

            speech_tokens = [speech_pad_id] * num_tokens
            return PromptUpdateDetails.select_token_id(
                [speech_start_id] + speech_tokens + [speech_end_id],
                embed_token_id=speech_pad_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=SPEECH_PAD_TOKEN,
                replacement=get_replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceASRMultiModalProcessor,
    info=VibeVoiceASRProcessingInfo,
    dummy_inputs=VibeVoiceASRDummyInputsBuilder,
)
class VibeVoiceASRForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsTranscription, SupportsPP
):
    supported_languages = LANGUAGES
    supports_transcription_only = True
    supports_multimodal_raw_input_only = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return SPEECH_PAD_TOKEN
        raise ValueError("Only audio modality is supported.")

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: Any, task_type: Literal["transcribe", "translate"]
    ) -> SpeechToTextConfig:
        if task_type == "translate":
            raise ValueError("VibeVoice-ASR does not support translation.")
        return SpeechToTextConfig(
            sample_rate=DEFAULT_SAMPLE_RATE,
            max_audio_clip_s=None,
            overlap_chunk_second=0,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: Any,
    ) -> int | None:
        ratio = _get_speech_tok_compress_ratio(model_config.hf_config)
        if ratio <= 0:
            return None
        return int(math.ceil(audio_duration_s * stt_config.sample_rate / ratio))

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: Any,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        if task_type == "translate":
            raise ValueError("VibeVoice-ASR does not support translation.")

        tokenizer = cached_tokenizer_from_config(model_config)
        audio_duration = len(audio) / stt_config.sample_rate
        show_keys = ["Start time", "End time", "Speaker ID", "Content"]

        context_info = request_prompt.strip() if request_prompt else ""
        if context_info:
            user_suffix = (
                f"This is a {audio_duration:.2f} seconds audio, with extra info: "
                f"{context_info}\n\nPlease transcribe it with these keys: "
                + ", ".join(show_keys)
            )
        else:
            user_suffix = (
                f"This is a {audio_duration:.2f} seconds audio, "
                "please transcribe it with these keys: " + ", ".join(show_keys)
            )

        user_content = f"{SPEECH_PAD_TOKEN}\n{user_suffix}"

        if hasattr(tokenizer, "apply_chat_template"):
            system_prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}], tokenize=False
            )
            system_tokens = tokenizer.encode(system_prompt_text)
            user_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}], tokenize=True
            )
            prompt_token_ids = system_tokens + user_tokens
        else:
            prompt_text = f"{SYSTEM_PROMPT}\n{user_content}"
            prompt_token_ids = tokenizer.encode(prompt_text)

        return {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"audio": audio},
        }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config

        self._ensure_vibevoice_imports()

        with self._mark_composite_model(
            vllm_config,
            language_targets=(VllmModel,),
            tower_targets={"audio": (PreTrainedModel, SpeechConnector)},
        ):
            self.model = VibeVoiceASRModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

        self.make_empty_intermediate_tensors = (
            self.model.language_model.make_empty_intermediate_tensors
        )

    def _ensure_vibevoice_imports(self) -> None:
        try:
            import vibevoice.modular.modular_vibevoice_tokenizer  # noqa: F401

            logger.info("Loaded VibeVoice tokenizer modules.")
        except Exception as exc:
            raise RuntimeError(
                "VibeVoice acoustic/semantic tokenizers require the "
                "`vibevoice` package. Install it with "
                "`pip install git+https://github.com/microsoft/VibeVoice.git` "
                "or add it to your environment before loading the model."
            ) from exc

    def _get_audio_dtype(self) -> torch.dtype:
        dtype = getattr(self.config, "torch_dtype", None)
        if dtype is None:
            return self.vllm_config.model_config.dtype
        if isinstance(dtype, str):
            return getattr(torch, dtype)
        return dtype

    def _pad_or_trim(self, features: torch.Tensor, target_len: int) -> torch.Tensor:
        if features.shape[0] == target_len:
            return features
        if features.shape[0] > target_len:
            return features[:target_len]
        pad_len = target_len - features.shape[0]
        pad = torch.zeros(
            (pad_len, features.shape[1]),
            device=features.device,
            dtype=features.dtype,
        )
        return torch.cat([features, pad], dim=0)

    def _split_by_masks(
        self, features: torch.Tensor, speech_masks: torch.Tensor | None
    ) -> list[torch.Tensor]:
        if speech_masks is None:
            return [features[i] for i in range(features.shape[0])]

        if speech_masks.ndim == 1:
            speech_masks = speech_masks.unsqueeze(0)
        speech_masks = speech_masks.to(device=features.device)

        outputs: list[torch.Tensor] = []
        for i in range(features.shape[0]):
            mask = speech_masks[i]
            seq = features[i]
            seq = self._pad_or_trim(seq, mask.shape[0])
            if mask.sum().item() == 0:
                raise ValueError("Audio token length is zero after masking.")
            outputs.append(seq[mask])
        return outputs

    def _encode_speech(
        self,
        speech_tensors: torch.Tensor,
        speech_masks: torch.Tensor | None = None,
        speech_semantic_tensors: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        audio_dtype = self._get_audio_dtype()
        speech_tensors = speech_tensors.to(dtype=audio_dtype)
        if speech_tensors.ndim == 1:
            speech_tensors = speech_tensors.unsqueeze(0)

        try:
            device = next(self.model.acoustic_tokenizer.parameters()).device
        except StopIteration:
            device = speech_tensors.device
        speech_tensors = speech_tensors.to(device)
        speech_tensors = speech_tensors.unsqueeze(1)

        with torch.no_grad():
            acoustic_output = self.model.acoustic_tokenizer.encode(speech_tensors)
            audio_tokens = acoustic_output.sample(
                dist_type=self.model.acoustic_tokenizer.std_dist_type
            )[0]
            acoustic_features = self.model.acoustic_connector(audio_tokens)

            if speech_semantic_tensors is None:
                semantic_output = self.model.semantic_tokenizer.encode(speech_tensors)
                semantic_tokens = semantic_output.mean
            else:
                semantic_tokens = speech_semantic_tensors.to(
                    device=acoustic_features.device, dtype=acoustic_features.dtype
                )
            semantic_features = self.model.semantic_connector(semantic_tokens)

            combined = acoustic_features + semantic_features

        return self._split_by_masks(combined, speech_masks)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        speech_tensors = kwargs.get("speech_tensors")
        if speech_tensors is None:
            return []
        speech_masks = kwargs.get("speech_masks")
        speech_semantic_tensors = kwargs.get("speech_semantic_tensors")

        if isinstance(speech_tensors, list):
            max_len = max(
                int(t.shape[0]) if isinstance(t, torch.Tensor) else len(t)
                for t in speech_tensors
            )
            padded = torch.zeros((len(speech_tensors), max_len), dtype=torch.float32)
            for i, audio in enumerate(speech_tensors):
                audio_tensor = torch.as_tensor(audio, dtype=torch.float32)
                padded[i, : audio_tensor.shape[0]] = audio_tensor
            speech_tensors = padded

        if isinstance(speech_masks, list):
            speech_masks = torch.stack(
                [torch.as_tensor(m, dtype=torch.bool) for m in speech_masks]
            )

        return self._encode_speech(
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            speech_semantic_tensors=speech_semantic_tensors,
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        return self.model.language_model.embed_input_ids(
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.model.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.model.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.language_model.": "model.language_model.model.",
                "lm_head.": "model.language_model.lm_head.",
            }
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)


class SpeechConnector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.fc1(features)
        x = self.norm(x)
        return self.fc2(x)


class VibeVoiceASRModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        audio_dtype = getattr(config, "torch_dtype", None)
        if audio_dtype is None:
            audio_dtype = vllm_config.model_config.dtype
        if isinstance(audio_dtype, str):
            audio_dtype = getattr(torch, audio_dtype)

        self.acoustic_tokenizer = AutoModel.from_config(
            config.acoustic_tokenizer_config
        ).to(audio_dtype)
        self.semantic_tokenizer = AutoModel.from_config(
            config.semantic_tokenizer_config
        ).to(audio_dtype)

        self.acoustic_connector = SpeechConnector(
            config.acoustic_vae_dim, config.decoder_config.hidden_size
        ).to(audio_dtype)
        self.semantic_connector = SpeechConnector(
            config.semantic_vae_dim, config.decoder_config.hidden_size
        ).to(audio_dtype)

        self.acoustic_tokenizer.eval()
        self.semantic_tokenizer.eval()

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.decoder_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        for module in (
            self.acoustic_tokenizer,
            self.semantic_tokenizer,
            self.acoustic_connector,
            self.semantic_connector,
        ):
            for param in module.parameters():
                param.requires_grad_(False)
