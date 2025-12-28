# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as AF

DEFAULT_MAX_AUDIO_LEN_S = 655
DEFAULT_MERGE_FACTOR = 4
DEFAULT_SAMPLE_RATE = 16000
# Default convolution parameters: (padding, kernel_size, stride)
# These correspond to the two conv layers in the Whisper-style frontend.
DEFAULT_CONV_PARAMS = [(1, 3, 1), (1, 3, 2)]


def _calculate_conv_output_length(
    input_length: torch.Tensor, padding: int, kernel_size: int, stride: int
) -> torch.Tensor:
    return (input_length + 2 * padding - kernel_size) // stride + 1


def _as_list_chunk_counts(
    chunk_counts: torch.Tensor | list[int] | list[torch.Tensor],
) -> list[int]:
    if isinstance(chunk_counts, torch.Tensor):
        return cast(list[int], chunk_counts.tolist())
    if chunk_counts and isinstance(chunk_counts[0], torch.Tensor):
        tensor_counts = cast(list[torch.Tensor], chunk_counts)
        return [int(c.item()) for c in tensor_counts]
    return [int(c) for c in chunk_counts]


def _normalize_chunk_counts(
    chunk_counts: torch.Tensor | list[int] | list[torch.Tensor] | None,
    num_chunks: int,
) -> list[int]:
    if chunk_counts is None:
        return [1] * num_chunks
    return _as_list_chunk_counts(chunk_counts)


def _get_audio_output_lengths_from_lengths(
    audio_lengths: torch.Tensor,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> torch.Tensor:
    for padding, kernel_size, stride in conv_params:
        audio_lengths = _calculate_conv_output_length(
            audio_lengths, padding, kernel_size, stride
        )
    return (audio_lengths - merge_factor) // merge_factor + 1


def _get_audio_output_lengths_from_mask(
    mask: torch.Tensor,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> torch.Tensor:
    audio_lengths = mask.sum(-1)
    return _get_audio_output_lengths_from_lengths(
        audio_lengths, merge_factor, conv_params
    )


def _get_audio_output_lengths_for_tower(
    audio_tower: nn.Module,
    audio_lengths: torch.Tensor,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> torch.Tensor:
    if hasattr(audio_tower, "_get_feat_extract_output_lengths"):
        _, audio_output_lengths = audio_tower._get_feat_extract_output_lengths(
            audio_lengths
        )
        return audio_output_lengths
    return _get_audio_output_lengths_from_lengths(
        audio_lengths, merge_factor, conv_params
    )


def _flatten_audio_features_by_length(
    audio_features: torch.Tensor,
    audio_output_lengths: torch.Tensor,
) -> torch.Tensor:
    num_chunks, max_audio_tokens, embed_dim = audio_features.shape
    audio_output_lengths = audio_output_lengths.unsqueeze(1)
    audio_features_mask = (
        torch.arange(max_audio_tokens)
        .expand(num_chunks, max_audio_tokens)
        .to(audio_output_lengths.device)
        < audio_output_lengths
    )
    return audio_features[audio_features_mask].view(-1, embed_dim)


def _group_audio_embeddings(
    chunk_embeddings: Sequence[torch.Tensor],
    chunk_counts: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    grouped_embeddings = []
    current_idx = 0
    for count in chunk_counts:
        audio_chunks = chunk_embeddings[current_idx : current_idx + count]
        grouped_embeddings.append(torch.cat(audio_chunks, dim=0))
        current_idx += count
    return tuple(grouped_embeddings)


def _normalize_to_tensor(mask: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(mask, list):
        return (
            torch.stack(mask)
            if mask and isinstance(mask[0], torch.Tensor)
            else torch.tensor(mask)
        )
    return mask


def _extract_mask_for_item(
    feature_attention_mask: torch.Tensor | list[torch.Tensor],
    chunk_counts: torch.Tensor | list[int] | None,
    item_idx: int,
) -> torch.Tensor:
    if chunk_counts is None:
        mask = feature_attention_mask[item_idx]
        if isinstance(feature_attention_mask, torch.Tensor):
            return mask.unsqueeze(0)
        return _normalize_to_tensor(mask)

    counts = _as_list_chunk_counts(chunk_counts)
    start_idx = sum(counts[:item_idx])
    end_idx = start_idx + counts[item_idx]

    if isinstance(feature_attention_mask, torch.Tensor):
        return feature_attention_mask[start_idx:end_idx]
    mask_slice = feature_attention_mask[start_idx:end_idx]
    return _normalize_to_tensor(mask_slice)


def _get_num_features_for_item(
    feature_attention_mask: torch.Tensor | None,
    chunk_counts: torch.Tensor | list[int] | None,
    item_idx: int,
    audio_embeds: list[torch.Tensor] | None,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> int:
    if feature_attention_mask is not None:
        mask = _extract_mask_for_item(feature_attention_mask, chunk_counts, item_idx)
        audio_output_lengths = _get_audio_output_lengths_from_mask(
            mask, merge_factor, conv_params
        )
        return int(audio_output_lengths.sum().item())
    if audio_embeds is not None:
        return int(audio_embeds[item_idx].shape[0])
    raise ValueError("Either feature_attention_mask or audio_embeds must be provided")


def _to_audio_tensor(
    audio: torch.Tensor | np.ndarray | list[float],
) -> torch.Tensor:
    if isinstance(audio, torch.Tensor):
        return audio
    if isinstance(audio, np.ndarray):
        return torch.from_numpy(audio)
    return torch.tensor(audio, dtype=torch.float32)


def compute_log_mel_spectrogram(
    audio: torch.Tensor,
    *,
    sampling_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    audio = audio.to(torch.float32)
    window = torch.hann_window(n_fft, device=audio.device)
    stft = torch.stft(
        audio,
        n_fft,
        hop_length,
        window=window,
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs().pow(2)
    mel_filters = AF.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        sample_rate=sampling_rate,
    ).to(magnitudes.device)
    mel_spec = mel_filters.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def extract_glmasr_features(
    audios: Sequence[torch.Tensor | list[float]],
    *,
    sampling_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    chunk_length_s: int,
    max_audio_len_s: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_length_frames = int(chunk_length_s * sampling_rate // hop_length)
    max_chunks = max(1, int(max_audio_len_s // max(1, chunk_length_s)))

    input_features = []
    feature_attention_mask = []
    chunk_counts = []

    for audio in audios:
        audio_tensor = _to_audio_tensor(audio)
        log_mel = compute_log_mel_spectrogram(
            audio_tensor,
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        num_frames = log_mel.shape[-1]
        chunks = []
        masks = []
        for start in range(0, num_frames, chunk_length_frames):
            end = start + chunk_length_frames
            chunk = log_mel[:, start:end]
            length = chunk.shape[-1]
            if length < chunk_length_frames:
                pad = chunk_length_frames - length
                chunk = torch.nn.functional.pad(chunk, (0, pad))
            mask = torch.zeros(chunk_length_frames, dtype=torch.long)
            mask[:length] = 1
            chunks.append(chunk)
            masks.append(mask)
            if len(chunks) >= max_chunks:
                break
        if not chunks:
            chunks = [log_mel[:, :chunk_length_frames]]
            masks = [torch.ones(chunk_length_frames, dtype=torch.long)]
        input_features.extend(chunks)
        feature_attention_mask.extend(masks)
        chunk_counts.append(len(chunks))

    input_features_tensor = torch.stack(input_features)
    feature_attention_mask_tensor = torch.stack(feature_attention_mask)
    chunk_counts_tensor = torch.tensor(chunk_counts, dtype=torch.long)
    return (
        input_features_tensor,
        feature_attention_mask_tensor,
        chunk_counts_tensor,
    )
