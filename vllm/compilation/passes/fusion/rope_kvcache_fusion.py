# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator
import os
from dataclasses import dataclass

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops import auto_functionalized
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.utils import get_kv_cache_layout

from ..fx_utils import find_getitem_maybe, is_auto_func, is_func
from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import (
    FLASHINFER_ROTARY_OP,
    QUANT_OPS,
    ROTARY_OP,
    MatcherRotaryEmbedding,
)
from .rms_quant_fusion import (
    empty_bf16,
    empty_i64,
)

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()


def fused_rope_and_unified_kv_cache_update_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> torch.Tensor:
    """
    This impl fetches the KV cache and slot mapping from the forward context,
    then calls the layer impl's `AttentionImpl.do_rope_and_kv_cache_update` method.
    It also returns a dummy tensor, similar to `Attention.unified_kv_cache_update`,
    that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        attn_layer.impl.do_rope_and_kv_cache_update(
            attn_layer,
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            kv_cache,
            layer_slot_mapping,
        )

    return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)


def fused_rope_and_unified_kv_cache_update_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> torch.Tensor:
    return torch.empty(0, device=query.device, dtype=query.dtype)


direct_register_custom_op(
    op_name="fused_rope_and_unified_kv_cache_update",
    op_func=fused_rope_and_unified_kv_cache_update_impl,
    mutates_args=["query", "key"],
    fake_impl=fused_rope_and_unified_kv_cache_update_fake,
)


def _fallback_rope_quant_kvcache_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    output: torch.Tensor,
    is_neox: bool,
    layer_name: str,
) -> None:
    attn_metadata, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(
        layer_name
    )
    if attn_metadata is None:
        output.zero_()
        return

    query = query.clone()
    key = key.clone()
    rotary_op = torch.ops._C.rotary_embedding.default
    rotary_op(
        positions=positions,
        query=query,
        key=key,
        head_size=attn_layer.head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )

    if (
        attn_layer.query_quant is not None
        and attn_layer.impl.supports_quant_query_input
    ):
        query, _ = attn_layer.query_quant(query, attn_layer._q_scale)

    num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
    query = query[:num_actual_tokens].view(
        -1, attn_layer.num_heads, attn_layer.head_size
    )
    key = key[:num_actual_tokens].view(
        -1, attn_layer.num_kv_heads, attn_layer.head_size
    )
    value = value[:num_actual_tokens]
    if value.ndim == 2:
        value = value.view(-1, attn_layer.num_kv_heads, attn_layer.head_size_v)

    if layer_slot_mapping is not None:
        attn_layer.impl.do_kv_cache_update(
            attn_layer,
            key,
            value,
            kv_cache,
            layer_slot_mapping,
        )

    attn_layer.impl.forward(
        attn_layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
    )


def _baseline_rope_quant_attention_only(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    output: torch.Tensor,
    is_neox: bool,
    attn_layer: Attention,
    kv_cache: torch.Tensor,
    attn_metadata: object,
) -> None:
    query = query.clone()
    key = key.clone()
    rotary_op = torch.ops._C.rotary_embedding.default
    rotary_op(
        positions=positions,
        query=query,
        key=key,
        head_size=attn_layer.head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )

    if (
        attn_layer.query_quant is not None
        and attn_layer.impl.supports_quant_query_input
    ):
        query, _ = attn_layer.query_quant(query, attn_layer._q_scale)

    num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
    query = query[:num_actual_tokens].view(
        -1, attn_layer.num_heads, attn_layer.head_size
    )
    key = key[:num_actual_tokens].view(
        -1, attn_layer.num_kv_heads, attn_layer.head_size
    )
    value = value[:num_actual_tokens]
    if value.ndim == 2:
        value = value.view(-1, attn_layer.num_kv_heads, attn_layer.head_size_v)

    attn_layer.impl.forward(
        attn_layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
    )


def _rope_append_metadata_matches_slot_mapping(
    attn_metadata: object,
    layer_slot_mapping: torch.Tensor | None,
    num_actual_tokens: int,
) -> bool:
    if layer_slot_mapping is None:
        return True
    batch_indices = getattr(attn_metadata, "rope_append_batch_indices", None)
    positions = getattr(attn_metadata, "rope_append_positions", None)
    paged_kv_indptr = getattr(attn_metadata, "paged_kv_indptr", None)
    paged_kv_indices = getattr(attn_metadata, "paged_kv_indices", None)
    page_size = getattr(attn_metadata, "page_size", None)
    if (
        batch_indices is None
        or positions is None
        or paged_kv_indptr is None
        or paged_kv_indices is None
        or page_size is None
    ):
        return False
    token_count = min(num_actual_tokens, int(layer_slot_mapping.shape[0]))
    if token_count <= 0:
        return True

    batch_indices = batch_indices[:token_count]
    positions = positions[:token_count]
    baseline_slots = layer_slot_mapping[:token_count].to(torch.int64)

    page_offsets = torch.div(positions, page_size, rounding_mode="floor")
    page_iters = paged_kv_indptr[batch_indices] + page_offsets
    page_ids = paged_kv_indices[page_iters]
    entry_idx = positions % page_size
    append_slots = page_ids.to(torch.int64) * page_size + entry_idx.to(torch.int64)
    return torch.equal(append_slots, baseline_slots)


def _rope_quant_runtime_debug_enabled() -> bool:
    return os.getenv("VLLM_DEBUG_ROPE_KVCACHE_RUNTIME", "0") == "1"


def _log_rope_quant_runtime_decision(
    layer_name: str,
    decision: str,
    reason: str | None = None,
) -> None:
    if not _rope_quant_runtime_debug_enabled():
        return
    if reason is None:
        logger.info_once(
            "rope_quant_kvcache runtime: %s layer=%s",
            decision,
            layer_name,
        )
    else:
        logger.info_once(
            "rope_quant_kvcache runtime: %s layer=%s reason=%s",
            decision,
            layer_name,
            reason,
        )


def fused_rope_quant_kvcache_attention_with_output_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    output: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> None:
    if os.getenv("VLLM_DEBUG_FORCE_ROPE_KVCACHE_FALLBACK", "0") == "1":
        _log_rope_quant_runtime_decision(
            layer_name,
            "fallback",
            "forced_by_env",
        )
        _fallback_rope_quant_kvcache_attention_with_output(
            query=query,
            key=key,
            value=value,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            output=output,
            is_neox=is_neox,
            layer_name=layer_name,
        )
        return

    attn_metadata, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(
        layer_name
    )
    if attn_metadata is None:
        output.zero_()
        return

    fallback_reason = None
    if not getattr(
        attn_layer.impl, "fused_rope_quant_kvcache_supported", lambda: False
    )():
        fallback_reason = "backend_unsupported"
    elif get_kv_cache_layout() != "HND":
        fallback_reason = "kv_cache_layout_not_hnd"
    elif getattr(attn_layer.impl, "kv_sharing_target_layer_name", None) is not None:
        fallback_reason = "kv_sharing_enabled"
    elif attn_layer.query_quant is None:
        fallback_reason = "query_quant_missing"
    elif attn_layer._q_scale.numel() != 1:
        fallback_reason = "q_scale_not_per_tensor"
    elif attn_layer._k_scale.numel() != 1:
        fallback_reason = "k_scale_not_per_tensor"
    elif attn_layer._v_scale.numel() != 1:
        fallback_reason = "v_scale_not_per_tensor"
    elif attn_layer._k_scale_float != attn_layer._v_scale_float:
        fallback_reason = "kv_scales_mismatch"
    elif getattr(attn_metadata, "num_prefills", 0) != 0:
        fallback_reason = "prefill_present"
    elif getattr(attn_metadata, "num_decodes", 0) == 0:
        fallback_reason = "no_decode_tokens"
    elif getattr(attn_metadata, "paged_kv_indptr", None) is None:
        fallback_reason = "paged_kv_indptr_missing"
    elif getattr(attn_metadata, "paged_kv_indices", None) is None:
        fallback_reason = "paged_kv_indices_missing"
    elif getattr(attn_metadata, "rope_append_batch_indices", None) is None:
        fallback_reason = "rope_append_batch_indices_missing"
    elif getattr(attn_metadata, "rope_append_positions", None) is None:
        fallback_reason = "rope_append_positions_missing"

    if fallback_reason is not None:
        _log_rope_quant_runtime_decision(layer_name, "fallback", fallback_reason)
        _fallback_rope_quant_kvcache_attention_with_output(
            query=query,
            key=key,
            value=value,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            output=output,
            is_neox=is_neox,
            layer_name=layer_name,
        )
        return

    from flashinfer.rope import _rope_quantize_fp8_append_paged_kv_cache
    from flashinfer.utils import TensorLayout

    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
    if not _rope_append_metadata_matches_slot_mapping(
        attn_metadata,
        layer_slot_mapping,
        num_actual_tokens,
    ):
        _log_rope_quant_runtime_decision(
            layer_name,
            "fallback",
            "slot_mapping_mismatch",
        )
        _fallback_rope_quant_kvcache_attention_with_output(
            query=query,
            key=key,
            value=value,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            output=output,
            is_neox=is_neox,
            layer_name=layer_name,
        )
        return

    query = query[:num_actual_tokens]
    key = key[:num_actual_tokens]
    value = value[:num_actual_tokens]
    positions = positions[:num_actual_tokens]

    if query.ndim == 2:
        query = query.view(-1, attn_layer.num_heads, attn_layer.head_size)
    if key.ndim == 2:
        key = key.view(-1, attn_layer.num_kv_heads, attn_layer.head_size)
    if value.ndim == 2:
        value = value.view(-1, attn_layer.num_kv_heads, attn_layer.head_size_v)

    q_nope_in = torch.empty(
        query.shape[0],
        attn_layer.num_heads,
        0,
        dtype=query.dtype,
        device=query.device,
    )
    k_nope_in = torch.empty(
        key.shape[0],
        attn_layer.num_kv_heads,
        0,
        dtype=key.dtype,
        device=key.device,
    )
    q_rope_out = torch.empty_like(query, dtype=FP8_DTYPE)
    q_nope_out = torch.empty(
        query.shape[0],
        attn_layer.num_heads,
        0,
        dtype=FP8_DTYPE,
        device=query.device,
    )

    cos_sin_cache_flashinfer = (
        cos_sin_cache.float() if cos_sin_cache.dtype != torch.float32 else cos_sin_cache
    )

    # vLLM stores dequant scales (x ~= fp8 * scale), while FlashInfer's
    # rope+quant kernels expect the quantization multiplier applied before cast.
    quant_scale_q = 1.0 / attn_layer._q_scale_float
    quant_scale_kv = 1.0 / attn_layer._k_scale_float

    kv_cache_flashinfer = kv_cache
    if attn_layer.impl.kv_cache_dtype.startswith("fp8"):
        torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
            attn_layer.impl.kv_cache_dtype
        )
        kv_cache_flashinfer = kv_cache_flashinfer.view(torch_dtype)
    stride_order = FlashInferBackend.get_kv_cache_stride_order()
    kv_layout = get_kv_cache_layout()
    kv_cache_flashinfer = kv_cache_flashinfer.permute(*stride_order)
    _log_rope_quant_runtime_decision(layer_name, "fast_path")

    _rope_quantize_fp8_append_paged_kv_cache(
        q_rope_in=query,
        k_rope_in=key,
        q_nope_in=q_nope_in,
        k_nope_in=k_nope_in,
        v_in=value,
        q_rope_out=q_rope_out,
        q_nope_out=q_nope_out,
        cos_sin_cache=cos_sin_cache_flashinfer,
        pos_ids=positions,
        k_cache=kv_cache_flashinfer[:, 0],
        v_cache=kv_cache_flashinfer[:, 1],
        ckv_cache=torch.empty(0, dtype=FP8_DTYPE, device=query.device),
        kpe_cache=torch.empty(0, dtype=FP8_DTYPE, device=query.device),
        kv_indices=attn_metadata.paged_kv_indices,
        kv_indptr=attn_metadata.paged_kv_indptr,
        batch_indices=attn_metadata.rope_append_batch_indices,
        positions=attn_metadata.rope_append_positions,
        kv_layout_code=TensorLayout[kv_layout].value,
        page_size=attn_metadata.page_size,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=quant_scale_kv,
        interleave=(not is_neox),
        enable_pdl=False,
    )

    _baseline_rope_quant_attention_only(
        query=query,
        key=key,
        value=value,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        output=output,
        is_neox=is_neox,
        attn_layer=attn_layer,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
    )


def fused_rope_quant_kvcache_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    output: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> None:
    return


direct_register_custom_op(
    op_name="fused_rope_quant_kvcache_attention_with_output",
    op_func=fused_rope_quant_kvcache_attention_with_output_impl,
    mutates_args=["output"],
    fake_impl=fused_rope_quant_kvcache_attention_with_output_fake,
)


class RopeReshapeKVCachePattern:
    """
    This pattern matches the following unfused inplace ops:
      q, k = rotary_embedding(positions, q, k, head_size, cos_sin_cache, is_neox)
      kv_cache_dummy = unified_kv_cache_update(k, v, layer_name)

    and replaces it with the fused inplace op:
      kv_cache_dummy = fused_rope_and_unified_kv_cache_update(
        q, k, v, positions, cos_sin_cache, is_neox, layer_name
      )
    """

    FUSED_OP = torch.ops.vllm.fused_rope_and_unified_kv_cache_update.default

    def __init__(
        self,
        layer: Attention,
        is_neox: bool,
    ) -> None:
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size
        self.head_size_v = layer.head_size_v
        self.is_neox = is_neox

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        self.v_size = self.num_kv_heads * self.head_size_v

        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=self.is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        # Sample inputs to help pattern tracing
        T = 5
        L = 4096
        qkv = empty_bf16(T, self.q_size + self.k_size + self.v_size)
        positions = empty_i64(T)
        cos_sin_cache = empty_bf16(L, self.head_size)
        return [
            qkv,
            positions,
            cos_sin_cache,
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
            q, k = self.rope_matcher(positions, q, k, cos_sin_cache)
            q = q.view(-1, self.num_heads, self.head_size)
            k = k.view(-1, self.num_kv_heads, self.head_size)
            v = v.view(-1, self.num_kv_heads, self.head_size_v)
            dummy = torch.ops.vllm.unified_kv_cache_update(k, v, self.layer_name)
            return dummy, q, k, v

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
            q = q.view(-1, self.num_heads, self.head_size)
            k = k.view(-1, self.num_kv_heads, self.head_size)
            v = v.view(-1, self.num_kv_heads, self.head_size_v)
            results = auto_functionalized(
                self.FUSED_OP,
                query=q,
                key=k,
                value=v,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                layer_name=self.layer_name,
            )
            return results[0], results[1], results[2], v

        # NOTE: use view_to_reshape to unify view/reshape to simplify
        # pattern and increase matching opportunities
        def fwd_and_view_to_reshape(*args, **kwargs) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), fwd_and_view_to_reshape, pm_pass
        )


@dataclass
class FlashInferRopeQuantKVCacheMatch:
    attention_node: fx.Node
    kv_update_node: fx.Node
    output_getitem: fx.Node | None
    query: fx.Node
    key: fx.Node
    value: fx.Node
    positions: fx.Node
    cos_sin_cache: fx.Node
    output: fx.Node
    layer_name: str
    is_neox: bool


class FlashInferRopeQuantKVCacheMatcher:
    FUSED_OP = torch.ops.vllm.fused_rope_quant_kvcache_attention_with_output.default
    ATTN_OP = torch.ops.vllm.unified_attention_with_output.default
    KV_UPDATE_OP = torch.ops.vllm.unified_kv_cache_update.default
    ROTARY_TARGETS = (ROTARY_OP, FLASHINFER_ROTARY_OP)

    def __init__(self, layers: dict[str, Attention]):
        self.layers = layers
        self.quant_ops = tuple(QUANT_OPS.values())

    @staticmethod
    def _is_view_like(node: fx.Node) -> bool:
        return (
            is_func(node, torch.ops.aten.view.default)
            or is_func(node, torch.ops.aten.reshape.default)
            or is_func(node, torch.ops.aten._unsafe_view.default)
            or is_func(node, torch.ops.aten.alias.default)
            or is_func(node, torch.ops.aten.contiguous.default)
            or is_func(node, torch.ops.aten.clone.default)
            or is_func(node, torch.ops.aten.detach.default)
        )

    def _iter_nodes(self, obj: object) -> list[fx.Node]:
        result: list[fx.Node] = []
        if isinstance(obj, fx.Node):
            return [obj]
        if isinstance(obj, list | tuple):
            for item in obj:
                result.extend(self._iter_nodes(item))
            return result
        if isinstance(obj, dict):
            for item in obj.values():
                result.extend(self._iter_nodes(item))
            return result
        return []

    def _depends_on(
        self,
        node: fx.Node,
        target: fx.Node,
        seen: set[fx.Node] | None = None,
    ) -> bool:
        if node is target:
            return True
        if seen is None:
            seen = set()
        if node in seen:
            return False
        seen.add(node)
        for inp in self._iter_nodes(node.args):
            if self._depends_on(inp, target, seen):
                return True
        for inp in self._iter_nodes(node.kwargs):
            if self._depends_on(inp, target, seen):
                return True
        return False

    def _find_rotary_ancestor(
        self,
        node: fx.Node,
        seen: set[fx.Node] | None = None,
    ) -> fx.Node | None:
        if seen is None:
            seen = set()
        if node in seen:
            return None
        seen.add(node)
        for target in self.ROTARY_TARGETS:
            if is_auto_func(node, target):
                return node
        for inp in self._iter_nodes(node.args):
            found = self._find_rotary_ancestor(inp, seen)
            if found is not None:
                return found
        for inp in self._iter_nodes(node.kwargs):
            found = self._find_rotary_ancestor(inp, seen)
            if found is not None:
                return found
        return None

    def _unwrap_view_like(self, node: fx.Node) -> fx.Node:
        cur = node
        seen: set[fx.Node] = set()
        while cur not in seen:
            seen.add(cur)
            if not self._is_view_like(cur):
                break
            if len(cur.args) == 0 or not isinstance(cur.args[0], fx.Node):
                break
            cur = cur.args[0]
        return cur

    def _get_quant_input(self, node: fx.Node) -> fx.Node | None:
        for quant_op in self.quant_ops:
            if is_auto_func(node, quant_op) or is_func(node, quant_op):
                quant_input = node.kwargs.get("input")
                if isinstance(quant_input, fx.Node):
                    return quant_input
                if len(node.args) >= 2 and isinstance(node.args[1], fx.Node):
                    return node.args[1]
                return None
        return None

    def _resolve_key_root(self, key_attn: fx.Node) -> fx.Node:
        return self._unwrap_view_like(key_attn)

    @staticmethod
    def _is_supported_query_transform(node: fx.Node) -> bool:
        if node.op != "call_function":
            return False
        target = str(node.target)
        return (
            target.startswith("prims.convert_element_type.default")
            or target.startswith("aten.mul.")
            or target.startswith("aten.div.")
            or target.startswith("aten.reciprocal.")
            or target.startswith("aten.clamp.")
            or target.startswith("aten.clamp_min.")
            or target.startswith("aten.clamp_max.")
            or target.startswith("aten.minimum.")
            or target.startswith("aten.maximum.")
            or target.startswith("aten._to_copy.")
        )

    def _has_supported_query_chain(
        self, query_attn: fx.Node, q_rot: fx.Node, k_rot: fx.Node
    ) -> bool:
        # Query must come from rotary query branch, and must not consume key branch.
        if not self._depends_on(query_attn, q_rot):
            return False
        if self._depends_on(query_attn, k_rot):
            return False

        stack = [query_attn]
        seen: set[fx.Node] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)

            if cur is q_rot:
                continue
            if cur is k_rot:
                return False
            if cur.op in ("placeholder", "get_attr"):
                continue

            quant_input = self._get_quant_input(cur)
            if quant_input is not None:
                stack.append(quant_input)
                continue

            if self._is_view_like(cur) or self._is_supported_query_transform(cur):
                stack.extend(self._iter_nodes(cur.args))
                stack.extend(self._iter_nodes(cur.kwargs))
                continue

            if is_func(cur, operator.getitem) and isinstance(cur.args[0], fx.Node):
                stack.append(cur.args[0])
                continue

            return False

        return True

    def _extract_attention(
        self, node: fx.Node
    ) -> tuple[fx.Node, fx.Node, fx.Node, fx.Node, str, fx.Node | None] | None:
        if is_auto_func(node, self.ATTN_OP):
            if node.kwargs.get("output_scale") is not None:
                return None
            if node.kwargs.get("output_block_scale") is not None:
                return None
            query = node.kwargs["query"]
            key = node.kwargs["key"]
            value = node.kwargs["value"]
            output = node.kwargs["output"]
            layer_name = node.kwargs["layer_name"]
            kv_cache_dummy_dep = node.kwargs.get("kv_cache_dummy_dep")
            return query, key, value, output, layer_name, kv_cache_dummy_dep

        if is_func(node, self.ATTN_OP):
            if node.kwargs.get("output_scale") is not None:
                return None
            if node.kwargs.get("output_block_scale") is not None:
                return None
            if len(node.args) < 5:
                return None
            query, key, value, output, layer_name = node.args[:5]
            kv_cache_dummy_dep = node.kwargs.get("kv_cache_dummy_dep")
            return query, key, value, output, layer_name, kv_cache_dummy_dep

        return None

    def _match_attention(self, node: fx.Node) -> FlashInferRopeQuantKVCacheMatch | None:
        attn = self._extract_attention(node)
        if attn is None:
            return None
        query_attn, key_attn, value_attn, output, layer_name, kv_update = attn
        if (
            not isinstance(query_attn, fx.Node)
            or not isinstance(key_attn, fx.Node)
            or not isinstance(value_attn, fx.Node)
            or not isinstance(output, fx.Node)
            or not isinstance(kv_update, fx.Node)
        ):
            return None
        if not isinstance(layer_name, str):
            return None
        if layer_name not in self.layers:
            return None
        kv_update_node: fx.Node = kv_update
        if not is_func(kv_update_node, self.KV_UPDATE_OP):
            return None
        if (
            kv_update_node.args[0] is not key_attn
            or kv_update_node.args[1] is not value_attn
        ):
            return None
        if kv_update_node.args[2] != layer_name:
            return None

        rotary_node = self._find_rotary_ancestor(key_attn)
        if rotary_node is None:
            return None
        q_rot = find_getitem_maybe(rotary_node, 1)
        k_rot = find_getitem_maybe(rotary_node, 2)
        if q_rot is None or k_rot is None:
            return None
        if not self._has_supported_query_chain(query_attn, q_rot, k_rot):
            return None
        key_root = self._resolve_key_root(key_attn)
        if key_root is not k_rot:
            return None

        output_getitem = None
        if is_auto_func(node, self.ATTN_OP):
            output_getitem = find_getitem_maybe(node, 1)
            if output_getitem is None:
                return None
            for user in list(node.users):
                if not is_func(user, operator.getitem):
                    return None
                if user.args[1] != 1 and len(user.users) != 0:
                    return None

        return FlashInferRopeQuantKVCacheMatch(
            attention_node=node,
            kv_update_node=kv_update_node,
            output_getitem=output_getitem,
            query=rotary_node.kwargs["query"],
            key=rotary_node.kwargs["key"],
            value=value_attn,
            positions=rotary_node.kwargs["positions"],
            cos_sin_cache=rotary_node.kwargs["cos_sin_cache"],
            output=output,
            layer_name=layer_name,
            is_neox=rotary_node.kwargs["is_neox"],
        )

    def _replace_match(
        self, graph: fx.Graph, match: FlashInferRopeQuantKVCacheMatch
    ) -> None:
        with graph.inserting_before(match.attention_node):
            graph.call_function(
                self.FUSED_OP,
                kwargs={
                    "query": match.query,
                    "key": match.key,
                    "value": match.value,
                    "positions": match.positions,
                    "cos_sin_cache": match.cos_sin_cache,
                    "output": match.output,
                    "is_neox": match.is_neox,
                    "layer_name": match.layer_name,
                },
            )

        if match.output_getitem is not None:
            match.output_getitem.replace_all_uses_with(match.output)
            graph.erase_node(match.output_getitem)
            for user in list(match.attention_node.users):
                if is_func(user, operator.getitem) and len(user.users) == 0:
                    graph.erase_node(user)

        graph.erase_node(match.attention_node)
        if len(match.kv_update_node.users) == 0:
            graph.erase_node(match.kv_update_node)

    def apply(self, graph: fx.Graph) -> int:
        matched = 0
        for node in list(graph.nodes):
            match = self._match_attention(node)
            if match is None:
                continue
            self._replace_match(graph, match)
            matched += 1
        return matched


class RopeKVCacheFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses the rotary embedding and KV cache update operations
    into a single fused kernel if available.

    It uses the pattern matcher and matches each layer manually, as strings
    cannot be wildcarded. This also lets us check support on attention layers
    upon registration instead of during pattern matching.

    This fusion eliminates the need for separate kernel launches and
    intermediate memory operations between the RoPE and cache update steps.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rope_kv_cache_fusion_pass"
        )

        cc = config.compilation_config
        self.max_token_num = cc.pass_config.rope_kvcache_fusion_max_token_num

        attn_layers = get_layers_from_vllm_config(config, Attention)
        flashinfer_layers: dict[str, Attention] = {}
        for _, layer in attn_layers.items():
            if layer.impl.fused_rope_kvcache_supported():
                for is_neox in [True, False]:
                    RopeReshapeKVCachePattern(
                        layer=layer,
                        is_neox=is_neox,
                    ).register(self.patterns)
            if (
                layer.query_quant is not None
                and layer.impl.fused_rope_quant_kvcache_supported()
                and layer._q_scale.numel() == 1
                and layer._k_scale.numel() == 1
                and layer._v_scale.numel() == 1
                and layer._k_scale_float == layer._v_scale_float
            ):
                flashinfer_layers[layer.layer_name] = layer

        self.flashinfer_matcher = FlashInferRopeQuantKVCacheMatcher(flashinfer_layers)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        matched = self.patterns.apply(graph)
        matched += self.flashinfer_matcher.apply(graph)
        self.matched_count = matched
        logger.debug("Replaced %s patterns", self.matched_count)
        if self.matched_count > 0:
            graph.lint()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # This pass works best for the small-batch decode setting.
        # For large-batch e.g. prefill, it is better to use two separate kernels
        # since they are compute bound and the fused kernels require further tuning.
        return compile_range.end <= self.max_token_num

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self,
            RopeReshapeKVCachePattern,
            FlashInferRopeQuantKVCacheMatcher,
            fused_rope_quant_kvcache_attention_with_output_impl,
        )
