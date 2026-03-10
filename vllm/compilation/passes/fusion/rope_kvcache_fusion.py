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
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import (
    MatcherRotaryEmbedding,
)
from .rms_quant_fusion import (
    empty_bf16,
    empty_i64,
)

logger = init_logger(__name__)
ATTN_OP = torch.ops.vllm.unified_attention_with_output.default
RESHAPE_OP = torch.ops.aten.reshape.default
FP8_DTYPE = current_platform.fp8_dtype()


def empty_fp8(*args, **kwargs) -> torch.Tensor:
    return torch.empty(*args, **kwargs, dtype=FP8_DTYPE, device="cuda")


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


def _layer_supports_graph_safe_flashinfer_rope_quant(layer: Attention) -> bool:
    support = getattr(layer.impl, "fused_rope_quant_kvcache_supported", None)
    return (
        callable(support)
        and support()
        and layer.query_quant is not None
        and layer._q_scale.numel() == 1
        and layer._k_scale.numel() == 1
        and layer._v_scale.numel() == 1
        and layer._k_scale_float == layer._v_scale_float
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
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    attn_metadata, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(
        layer_name
    )

    query_quant: torch.Tensor
    key_for_attention: torch.Tensor
    value_for_attention: torch.Tensor

    try:
        from flashinfer.rope import _rope_quantize_fp8_append_paged_kv_cache
        from flashinfer.utils import TensorLayout

        from vllm.v1.attention.backends.flashinfer import (
            FlashInferBackend,
            TRTLLMDecode,
        )
        from vllm.v1.attention.backends.utils import get_kv_cache_layout
    except ImportError:
        attn_metadata = None

    q_scale = getattr(attn_layer, "_q_scale_float", None)
    k_scale = getattr(attn_layer, "_k_scale_float", None)
    can_use_flashinfer_kernel = (
        attn_metadata is not None
        and getattr(
            attn_layer.impl, "fused_rope_quant_kvcache_supported", lambda: False
        )()
        and isinstance(getattr(attn_metadata, "decode", None), TRTLLMDecode)
        and attn_metadata.num_prefills == 0
        and attn_metadata.num_decodes > 0
        and getattr(attn_metadata, "paged_kv_indptr", None) is not None
        and getattr(attn_metadata, "paged_kv_indices", None) is not None
        and getattr(attn_metadata, "rope_append_batch_indices", None) is not None
        and getattr(attn_metadata, "rope_append_positions", None) is not None
        and q_scale is not None
        and k_scale is not None
        and attn_layer._q_scale.numel() == 1
        and attn_layer._k_scale.numel() == 1
        and attn_layer._v_scale.numel() == 1
        and attn_layer._k_scale_float == attn_layer._v_scale_float
    )

    if can_use_flashinfer_kernel:
        num_actual_tokens = attn_metadata.num_actual_tokens
        query_actual = query[:num_actual_tokens]
        key_actual = key[:num_actual_tokens]
        value_actual = value[:num_actual_tokens]
        pos_ids = positions[:num_actual_tokens]
        batch_indices = attn_metadata.rope_append_batch_indices
        cache_positions = attn_metadata.rope_append_positions
        assert batch_indices is not None
        assert cache_positions is not None

        if attn_layer.impl.kv_cache_dtype.startswith("fp8"):
            torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                attn_layer.impl.kv_cache_dtype
            )
            kv_cache = kv_cache.view(torch_dtype)
        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        kv_cache_view = kv_cache.permute(*stride_order)
        query_quant = torch.empty_like(query, dtype=attn_metadata.q_data_type)
        query_quant_actual = query_quant[:num_actual_tokens]
        q_nope_in = query_actual[..., :0]
        k_nope_in = key_actual[..., :0]
        q_nope_out = query_quant_actual[..., :0]
        empty_cache = torch.empty(0, dtype=query_quant.dtype, device=query.device)
        _rope_quantize_fp8_append_paged_kv_cache(
            q_rope_in=query_actual,
            k_rope_in=key_actual,
            q_nope_in=q_nope_in,
            k_nope_in=k_nope_in,
            v_in=value_actual,
            q_rope_out=query_quant_actual,
            q_nope_out=q_nope_out,
            cos_sin_cache=(
                cos_sin_cache
                if cos_sin_cache.dtype == torch.float32
                else cos_sin_cache.float()
            ),
            pos_ids=pos_ids,
            k_cache=kv_cache_view[:, 0],
            v_cache=kv_cache_view[:, 1],
            ckv_cache=empty_cache,
            kpe_cache=empty_cache,
            kv_indices=attn_metadata.paged_kv_indices,
            kv_indptr=attn_metadata.paged_kv_indptr,
            batch_indices=batch_indices,
            positions=cache_positions,
            kv_layout_code=TensorLayout[get_kv_cache_layout()].value,
            page_size=attn_metadata.page_size,
            quant_scale_q=q_scale,
            quant_scale_kv=k_scale,
            interleave=not is_neox,
            enable_pdl=False,
        )
        # Pure TRTLLM decode reads from KV cache only, so external K/V tensors are
        # semantically dead once the fused append is complete.
        key_for_attention = key[:0]
        value_for_attention = value[:0]
    else:
        query_roped, key_roped = _rope_qk(query, key, positions, cos_sin_cache, is_neox)

        if layer_slot_mapping is not None:
            attn_layer.impl.do_kv_cache_update(
                attn_layer,
                key_roped,
                value,
                kv_cache,
                layer_slot_mapping,
            )

        if attn_layer.query_quant is None:
            raise AssertionError(
                "FlashInfer rope+cache+quant fallback requires query_quant "
                "to be initialized on the attention layer."
            )

        query_quant, _ = attn_layer.query_quant(query_roped, attn_layer._q_scale)
        key_for_attention = key_roped
        value_for_attention = value

    attn_layer.impl.forward(
        attn_layer,
        query_quant,
        key_for_attention,
        value_for_attention,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
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
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="fused_rope_quant_kvcache_attention_with_output",
    op_func=fused_rope_quant_kvcache_attention_with_output_impl,
    mutates_args=["output", "output_block_scale"],
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


class FlashInferRopeQuantKVCachePattern:
    """
    Matches:
      q, k = rotary_embedding(...)
      dummy = unified_kv_cache_update(k, v, layer_name)
      unified_attention_with_output(q_attn, k, v, ...)

    and replaces the whole chain with a FlashInfer-specific fused op that
    performs rope+quant+cache-append and then runs attention, so the graph no
    longer needs an externally produced `key_roped`.
    """

    FUSED_OP = torch.ops.vllm.fused_rope_quant_kvcache_attention_with_output.default

    def __init__(
        self,
        layer: Attention,
        is_neox: bool,
        use_flashinfer_rotary: bool = False,
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
            use_flashinfer=use_flashinfer_rotary,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        L = 4096
        qkv = empty_bf16(T, self.q_size + self.k_size + self.v_size)
        positions = empty_i64(T)
        cos_sin_cache = empty_bf16(L, self.head_size)
        q_attn = empty_fp8(T, self.num_heads, self.head_size)
        output = empty_bf16(T, self.num_heads, self.head_size_v)
        return [qkv, positions, cos_sin_cache, q_attn, output]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            q_attn: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
            _, k = self.rope_matcher(positions, q, k, cos_sin_cache)
            k = k.view(-1, self.num_kv_heads, self.head_size)
            v = v.view(-1, self.num_kv_heads, self.head_size_v)
            dummy = torch.ops.vllm.unified_kv_cache_update(k, v, self.layer_name)
            at1 = auto_functionalized(
                ATTN_OP,
                query=q_attn,
                key=k,
                value=v,
                output=output,
                layer_name=self.layer_name,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=dummy,
            )
            return RESHAPE_OP(at1[1], [-1, self.num_heads * self.head_size_v])

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            q_attn: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            del q_attn
            q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
            q = q.view(-1, self.num_heads, self.head_size)
            k = k.view(-1, self.num_kv_heads, self.head_size)
            v = v.view(-1, self.num_kv_heads, self.head_size_v)
            at1 = auto_functionalized(
                self.FUSED_OP,
                query=q,
                key=k,
                value=v,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
                output=output,
                is_neox=self.is_neox,
                layer_name=self.layer_name,
                output_scale=None,
                output_block_scale=None,
            )
            return RESHAPE_OP(at1[1], [-1, self.num_heads * self.head_size_v])

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
            if _layer_supports_graph_safe_flashinfer_rope_quant(layer):
                for is_neox in [True, False]:
                    for use_flashinfer_rotary in [False, True]:
                        FlashInferRopeQuantKVCachePattern(
                            layer=layer,
                            is_neox=is_neox,
                            use_flashinfer_rotary=use_flashinfer_rotary,
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
