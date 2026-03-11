# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops import auto_functionalized
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm import envs
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
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


def fused_rope_quant_and_unified_kv_cache_update_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_scale: torch.Tensor,
    head_size: int,
    head_size_v: int,
    is_neox: bool,
    layer_name: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)

    query = query.clone()
    key = key.clone()
    value = value.clone()
    torch.ops._C.rotary_embedding.default(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )
    assert attn_layer.query_quant is not None
    query, _ = attn_layer.query_quant(query, q_scale)

    num_heads = query.shape[-1] // head_size
    num_kv_heads = key.shape[-1] // head_size
    query = query.view(-1, num_heads, head_size)
    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, value.shape[-1] // head_size_v, head_size_v)

    if layer_slot_mapping is not None:
        attn_layer.impl.do_kv_cache_update(
            attn_layer,
            key,
            value,
            kv_cache,
            layer_slot_mapping,
        )

    dummy = torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)
    return dummy, query, key, value


def fused_rope_quant_and_unified_kv_cache_update_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_scale: torch.Tensor,
    head_size: int,
    head_size_v: int,
    is_neox: bool,
    layer_name: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fp8_dtype = current_platform.fp8_dtype()
    query = torch.empty(
        (query.shape[0], query.shape[-1] // head_size, head_size),
        dtype=fp8_dtype,
        device=query.device,
    )
    key = torch.empty(
        (key.shape[0], key.shape[-1] // head_size, head_size),
        dtype=key.dtype,
        device=key.device,
    )
    value = torch.empty(
        (value.shape[0], value.shape[-1] // head_size_v, head_size_v),
        dtype=value.dtype,
        device=value.device,
    )
    dummy = torch.empty(0, device=key.device, dtype=key.dtype)
    return dummy, query, key, value


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
    Match the real CUDA decode graph shape:
      split_with_sizes(qkv) -> auto_functionalized(rotary_embedding)
      -> inline fp8 query quant -> reshape(q, k, v)
      -> unified_kv_cache_update(k, v)
    and replace it with a smaller fused op, leaving attention in the graph.
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
        self.fp8_dtype = current_platform.fp8_dtype()
        self.fp8_min, self.fp8_max = get_fp8_min_max()
        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=self.is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        t = 5
        qkv = empty_bf16(t, self.q_size + self.k_size + self.v_size)
        positions = empty_i64(t)
        cos_sin_cache = empty_bf16(4096, self.head_size)
        q_scale = torch.empty((), dtype=torch.float32, device=qkv.device)
        return [qkv, positions, cos_sin_cache, q_scale]

    def _quantize_query(
        self, query: torch.Tensor, q_scale: torch.Tensor
    ) -> torch.Tensor:
        query = torch.ops.prims.convert_element_type.default(query, torch.float32)
        reciprocal = torch.ops.aten.reciprocal.default(q_scale)
        query = torch.ops.aten.mul.Tensor(query, reciprocal)
        query = query.clamp(min=self.fp8_min)
        query = query.clamp(max=self.fp8_max)
        return torch.ops.prims.convert_element_type.default(query, self.fp8_dtype)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            q_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            query, key, value = qkv.split(
                [self.q_size, self.k_size, self.v_size], dim=-1
            )
            query, key = self.rope_matcher(positions, query, key, cos_sin_cache)
            assert key is not None
            query = self._quantize_query(query, q_scale)
            query = query.view(-1, self.num_heads, self.head_size)
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size_v)
            kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                key, value, self.layer_name
            )
            return kv_cache_dummy_dep, query, key, value

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            q_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            query, key, value = qkv.split(
                [self.q_size, self.k_size, self.v_size], dim=-1
            )
            return self.FUSED_OP(
                query=query,
                key=key,
                value=value,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
                q_scale=q_scale,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                is_neox=self.is_neox,
                layer_name=self.layer_name,
            )

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
            if getattr(layer.impl, "fused_rope_kvcache_supported", lambda: False)():
                for is_neox in [True, False]:
                    RopeReshapeKVCachePattern(
                        layer=layer,
                        is_neox=is_neox,
                    ).register(self.patterns)
            if (
                layer.query_quant is not None
                and getattr(
                    layer.impl,
                    "fused_rope_quant_kvcache_supported",
                    lambda: False,
                )()
            ):
                for is_neox in [True, False]:
                    FlashInferRopeQuantKVCachePattern(
                        layer=layer,
                        is_neox=is_neox,
                    ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if envs.VLLM_DUMP_ONLY_ROPE_KVCACHE_PASS:
            self.matched_count = 0
            logger.debug(
                "Dump-only mode enabled for RopeKVCacheFusionPass; "
                "skipping replacements."
            )
            return
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # This pass works best for the small-batch decode setting.
        # For large-batch e.g. prefill, it is better to use two separate kernels
        # since they are compute bound and the fused kernels require further tuning.
        return compile_range.end <= self.max_token_num

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self,
            RopeReshapeKVCachePattern,
            FlashInferRopeQuantKVCachePattern,
            fused_rope_quant_and_unified_kv_cache_update_impl,
        )
