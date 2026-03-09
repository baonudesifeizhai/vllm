# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import (
    MatcherQuantFP8,
    MatcherRotaryEmbedding,
)
from .rms_quant_fusion import (
    empty_bf16,
    empty_fp32,
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


def _rope_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops

    query_roped = query.clone()
    key_roped = key.clone()
    ops.rotary_embedding(
        positions,
        query_roped,
        key_roped,
        query.shape[-1],
        cos_sin_cache,
        is_neox,
    )
    return query_roped, key_roped


def _fallback_rope_quant_kvcache(
    attn_layer: Attention,
    kv_cache: torch.Tensor,
    layer_slot_mapping: torch.Tensor | None,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_roped, key_roped = _rope_qk(query, key, positions, cos_sin_cache, is_neox)

    if layer_slot_mapping is not None:
        attn_layer.impl.do_kv_cache_update(
            attn_layer,
            key_roped,
            value,
            kv_cache,
            layer_slot_mapping,
        )

    query_quant, _ = QuantFP8(
        static=True,
        group_shape=GroupShape.PER_TENSOR,
        compile_native=False,
    )(query_roped.reshape(query_roped.shape[0], -1), attn_layer._q_scale)
    dummy = torch.empty(0, device=query.device, dtype=query.dtype)
    return dummy, query_quant.view_as(query_roped), key_roped


def fused_rope_quant_and_unified_kv_cache_update_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attn_metadata, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(
        layer_name
    )

    try:
        from flashinfer import get_batch_indices_positions
        from flashinfer.rope import rope_quantize_fp8_append_paged_kv_cache

        from vllm.v1.attention.backends.flashinfer import FlashInferBackend
        from vllm.v1.attention.backends.utils import get_kv_cache_layout
    except ImportError:
        return _fallback_rope_quant_kvcache(
            attn_layer,
            kv_cache,
            layer_slot_mapping,
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
        )

    q_scale = getattr(attn_layer, "_q_scale_float", None)
    k_scale = getattr(attn_layer, "_k_scale_float", None)
    v_scale = getattr(attn_layer, "_v_scale_float", None)
    can_use_flashinfer_kernel = (
        attn_metadata is not None
        and getattr(
            attn_layer.impl, "fused_rope_quant_kvcache_supported", lambda: False
        )()
        and getattr(attn_metadata, "q_data_type", None) == FP8_DTYPE
        and getattr(attn_metadata, "num_prefills", 1) == 0
        and getattr(attn_metadata, "num_decodes", 0) > 0
        and not getattr(attn_metadata, "use_cascade", True)
        and getattr(attn_layer.impl, "dcp_world_size", 1) == 1
        and getattr(attn_metadata, "paged_kv_indptr", None) is not None
        and getattr(attn_metadata, "paged_kv_indices", None) is not None
        and q_scale is not None
        and k_scale is not None
        and v_scale is not None
        and attn_layer._q_scale.numel() == 1
        and attn_layer._k_scale.numel() == 1
        and attn_layer._v_scale.numel() == 1
        and k_scale == v_scale
    )

    if not can_use_flashinfer_kernel:
        return _fallback_rope_quant_kvcache(
            attn_layer,
            kv_cache,
            layer_slot_mapping,
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
        )

    query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
    seq_lens_before_append = attn_metadata.seq_lens - query_lens
    batch_indices, cache_positions = get_batch_indices_positions(
        attn_metadata.query_start_loc,
        seq_lens_before_append,
        query.shape[0],
    )

    if attn_layer.impl.kv_cache_dtype.startswith("fp8"):
        torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
            attn_layer.impl.kv_cache_dtype
        )
        kv_cache = kv_cache.view(torch_dtype)
    stride_order = FlashInferBackend.get_kv_cache_stride_order()
    kv_cache_view = kv_cache.permute(*stride_order)
    query_quant, _ = rope_quantize_fp8_append_paged_kv_cache(
        q_rope=query,
        k_rope=key,
        q_nope=None,
        k_nope=None,
        v=value,
        cos_sin_cache=(
            cos_sin_cache
            if cos_sin_cache.dtype == torch.float32
            else cos_sin_cache.float()
        ),
        pos_ids=positions,
        paged_kv_cache=(kv_cache_view[:, 0], kv_cache_view[:, 1]),
        kv_indices=attn_metadata.paged_kv_indices,
        kv_indptr=attn_metadata.paged_kv_indptr,
        batch_indices=batch_indices,
        positions=cache_positions,
        is_neox=is_neox,
        quantize_dtype=attn_metadata.q_data_type,
        quant_scale_q=q_scale,
        quant_scale_kv=k_scale,
        page_size=attn_metadata.page_size,
        kv_layout=get_kv_cache_layout(),
    )

    # The fused FlashInfer API writes quantized KV to cache but does not return
    # the RoPE-transformed key tensor, so reconstruct it for graph consumers.
    _, key_roped = _rope_qk(query, key, positions, cos_sin_cache, is_neox)
    dummy = torch.empty(0, device=query.device, dtype=query.dtype)
    return dummy, query_quant, key_roped


def fused_rope_quant_and_unified_kv_cache_update_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dummy = torch.empty(0, device=query.device, dtype=query.dtype)
    query_quant = torch.empty_like(query, dtype=FP8_DTYPE)
    return dummy, query_quant, key


direct_register_custom_op(
    op_name="fused_rope_quant_and_unified_kv_cache_update",
    op_func=fused_rope_quant_and_unified_kv_cache_update_impl,
    fake_impl=fused_rope_quant_and_unified_kv_cache_update_fake,
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


class FlashInferRopeQuantKVCachePattern:
    """
    Matches:
      q, k = rotary_embedding(...)
      q_quant = static_fp8_quant(q, scale)
      dummy = unified_kv_cache_update(k, v, layer_name)

    and replaces it with a FlashInfer-specific fused op that returns the
    quantized query tensor and updates the paged KV cache directly.
    """

    FUSED_OP = torch.ops.vllm.fused_rope_quant_and_unified_kv_cache_update.default

    def __init__(self, layer: Attention, is_neox: bool) -> None:
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
        self.quant_matcher = MatcherQuantFP8(kFp8StaticTensorSym)

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        L = 4096
        qkv = empty_bf16(T, self.q_size + self.k_size + self.v_size)
        positions = empty_i64(T)
        cos_sin_cache = empty_bf16(L, self.head_size)
        scale = empty_fp32(1, 1)
        return [qkv, positions, cos_sin_cache, scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
            q, k = self.rope_matcher(positions, q, k, cos_sin_cache)
            q = self.quant_matcher(q, scale)[0]
            q = q.view(-1, self.num_heads, self.head_size)
            k = k.view(-1, self.num_kv_heads, self.head_size)
            v = v.view(-1, self.num_kv_heads, self.head_size_v)
            dummy = torch.ops.vllm.unified_kv_cache_update(k, v, self.layer_name)
            return dummy, q, k, v

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            del scale
            q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
            q = q.view(-1, self.num_heads, self.head_size)
            k = k.view(-1, self.num_kv_heads, self.head_size)
            v = v.view(-1, self.num_kv_heads, self.head_size_v)
            dummy, query_quant, key_roped = self.FUSED_OP(
                query=q,
                key=k,
                value=v,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                layer_name=self.layer_name,
            )
            return dummy, query_quant, key_roped, v

        def fwd_and_view_to_reshape(*args, **kwargs) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), fwd_and_view_to_reshape, pm_pass
        )


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
        for _, layer in attn_layers.items():
            flashinfer_support = getattr(
                layer.impl, "fused_rope_quant_kvcache_supported", None
            )
            if callable(flashinfer_support) and flashinfer_support():
                for is_neox in [True, False]:
                    FlashInferRopeQuantKVCachePattern(
                        layer=layer,
                        is_neox=is_neox,
                    ).register(self.patterns)
            elif layer.impl.fused_rope_kvcache_supported():
                for is_neox in [True, False]:
                    RopeReshapeKVCachePattern(
                        layer=layer,
                        is_neox=is_neox,
                    ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # This pass works best for the small-batch decode setting.
        # For large-batch e.g. prefill, it is better to use two separate kernels
        # since they are compute bound and the fused kernels require further tuning.
        return compile_range.end <= self.max_token_num

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self, RopeReshapeKVCachePattern, FlashInferRopeQuantKVCachePattern
        )
