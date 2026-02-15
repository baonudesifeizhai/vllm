# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable

import pplx_kernels as pplx
import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _validate_scale_shape,
    moe_kernel_quantize_input,
)
from vllm.utils.math_utils import cdiv, round_up

logger = init_logger(__name__)
_PPLX_COMBINE_FIRST_CONTEXT_LOGGED: set[str] = set()


def _pplx_debug_enabled() -> bool:
    return envs.VLLM_PPLX_DEBUG


def _has_non_finite(tensor: torch.Tensor | None) -> bool:
    if tensor is None:
        return False
    return not torch.isfinite(tensor.float()).all().item()


def _tensor_stats(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "none"
    shape = tuple(tensor.shape)
    if tensor.numel() == 0:
        return f"shape={shape} dtype={tensor.dtype} empty"
    stats = tensor.float()
    return (
        f"shape={shape} dtype={tensor.dtype} min={stats.min().item():.4g} "
        f"max={stats.max().item():.4g} mean={stats.mean().item():.4g} "
        f"sum={stats.sum().item():.4g}"
    )


def _tensor_abs_max(tensor: torch.Tensor | None) -> float:
    if tensor is None or tensor.numel() == 0:
        return 0.0
    return float(tensor.float().abs().max().item())


def _combine_abs_max_threshold() -> float:
    value = os.getenv("VLLM_CUTLASS_MOE_MM_OUTPUT_ABS_MAX_THRESHOLD")
    if value is None:
        return 1e4
    try:
        threshold = float(value)
    except ValueError:
        return 1e4
    if threshold > 0.0:
        return threshold
    return 1e4


def _maybe_log_combine_first_context(
    *,
    stage: str,
    layer_id: int | None,
    layer_name: str | None,
    apply_router_weight_on_input: bool,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fused_expert_output: torch.Tensor,
    output: torch.Tensor,
    output_before_recv: torch.Tensor | None = None,
) -> None:
    global _PPLX_COMBINE_FIRST_CONTEXT_LOGGED
    if stage in _PPLX_COMBINE_FIRST_CONTEXT_LOGGED:
        return

    threshold = _combine_abs_max_threshold()
    fused_abs_max = _tensor_abs_max(fused_expert_output)
    output_abs_max = _tensor_abs_max(output)
    output_before_abs_max = _tensor_abs_max(output_before_recv)
    output_delta_stats = "none"
    output_delta_abs_max = 0.0
    if (
        output_before_recv is not None
        and output_before_recv.shape == output.shape
        and output_before_recv.dtype == output.dtype
    ):
        output_delta = output - output_before_recv
        output_delta_abs_max = _tensor_abs_max(output_delta)
        output_delta_stats = _tensor_stats(output_delta)

    # Skip all-zero warmup contexts; keep first meaningful or suspicious context.
    likely_warmup = (
        fused_abs_max == 0.0 and output_abs_max == 0.0 and output_delta_abs_max == 0.0
    )
    suspicious = (
        _has_non_finite(fused_expert_output)
        or _has_non_finite(topk_weights)
        or _has_non_finite(output_before_recv)
        or _has_non_finite(output)
        or fused_abs_max >= threshold
        or output_abs_max >= threshold
        or output_delta_abs_max >= threshold
    )
    if likely_warmup and not suspicious:
        return

    _PPLX_COMBINE_FIRST_CONTEXT_LOGGED.add(stage)
    logger.warning(
        "PPLX_COMBINE_FIRST_CONTEXT layer_id=%s layer_name=%s stage=%s "
        "abs_max_threshold=%s suspicious=%s "
        "apply_router_weight_on_input=%s topk_ids=%s topk_weights=%s "
        "fused_expert_output=%s output_before_recv=%s output=%s "
        "fused_abs_max=%s output_before_abs_max=%s output_abs_max=%s "
        "output_delta_abs_max=%s output_delta=%s",
        layer_id,
        layer_name,
        stage,
        threshold,
        suspicious,
        apply_router_weight_on_input,
        _tensor_stats(topk_ids),
        _tensor_stats(topk_weights),
        _tensor_stats(fused_expert_output),
        _tensor_stats(output_before_recv),
        _tensor_stats(output),
        fused_abs_max,
        output_before_abs_max,
        output_abs_max,
        output_delta_abs_max,
        output_delta_stats,
    )


def pplx_hidden_dim_scale_bytes(
    max_num_tokens: int,
    hidden_dim: int,
    in_dtype: torch.dtype,
    quant_dtype: torch.dtype | str | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None,
):
    # All pplx byte sizes must be 16-byte aligned.
    align = 16

    # For blocked per token: set to
    #   cdiv(hidden_dim, block_size) * sizeof(float32)
    # For per-token: set to 4 * sizeof(float32) (x4 for alignment)
    if quant_dtype is not None:
        assert isinstance(quant_dtype, torch.dtype)
        assert quant_dtype.itemsize == 1
        hidden_dim_bytes = hidden_dim * quant_dtype.itemsize
        elem_size = torch.float32.itemsize

        if per_act_token_quant:
            # per-token (M x 1)
            assert block_shape is None
            hidden_scale_bytes = elem_size
        elif block_shape is not None:
            # per-group (M x K_tiles)
            block_size = block_shape[1]
            num_blocks = cdiv(hidden_dim, block_size)
            hidden_scale_bytes = num_blocks * elem_size
        else:
            # per-tensor (1 x 1)
            hidden_scale_bytes = elem_size
    else:
        hidden_dim_bytes = hidden_dim * in_dtype.itemsize
        hidden_scale_bytes = 0

    return (
        round_up(hidden_dim_bytes, align),
        round_up(hidden_scale_bytes, align),
    )


class PplxPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """PPLX-based prepare and finalize for expert parallelism."""

    def __init__(
        self,
        a2a: pplx.AllToAll,
        max_num_tokens: int,
        num_local_experts: int,
        num_dispatchers: int,
    ):
        super().__init__()
        assert max_num_tokens > 0
        assert num_local_experts > 0
        self.a2a = a2a
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts
        self.num_dispatchers_ = num_dispatchers
        self.layer_id: int | None = None
        self.layer_name: str | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_num_tokens

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.uint32

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[Callable, mk.ReceiverType]:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )

        num_tokens = a1.size(0)  # M
        hidden_dim = a1.size(-1)  # K

        assert topk_ids.size(0) == num_tokens
        assert topk_ids.dtype == torch.uint32, (
            "PPLX expects topk_ids dtype torch.uint32. "
            "Check router indices dtype conversion for PPLX all2all."
        )
        assert topk_ids.is_contiguous(), "PPLX expects topk_ids to be contiguous."
        assert topk_ids.stride(-1) == 1, (
            f"PPLX expects topk_ids stride(-1) == 1, got {topk_ids.stride()}."
        )
        assert topk_ids.size() == topk_weights.size(), (
            f"{topk_ids.size()} == {topk_weights.size()}"
        )
        assert topk_weights.is_contiguous(), (
            "PPLX expects topk_weights to be contiguous."
        )
        assert topk_weights.stride(-1) == 1, (
            f"PPLX expects topk_weights stride(-1) == 1, got {topk_weights.stride()}."
        )
        if topk_ids.numel() != 0:
            topk_ids_i64 = topk_ids.to(torch.int64)
            topk_ids_min = int(topk_ids_i64.min().item())
            topk_ids_max = int(topk_ids_i64.max().item())
            assert topk_ids_min >= 0, (
                "PPLX saw negative topk_ids. "
                "Check router indices dtype conversion or padding."
            )
            assert topk_ids_max < num_experts, (
                "PPLX saw topk_ids out of range. "
                "Check router logits or expert count configuration."
            )
        if topk_weights.numel() != 0:
            assert topk_weights.is_floating_point(), (
                "PPLX expects topk_weights to be a floating-point tensor."
            )
            assert torch.isfinite(topk_weights).all(), (
                "PPLX saw non-finite values in topk_weights."
            )
            topk_weights_min = float(topk_weights.min().item())
            topk_weights_max = float(topk_weights.max().item())
            assert topk_weights_min >= 0.0, (
                "PPLX saw negative topk_weights. "
                "Check router logits or softmax stability."
            )
            assert topk_weights_max <= 1.0 + 1e-3, (
                "PPLX saw topk_weights outside the expected [0, 1] range. "
                "Check router logits or softmax stability."
            )
        # expert_map should be None because with expert map, -1 id is used for
        # non-local token; this causes error when casting ids to the
        # topk_indices_dtype() int32
        #
        if expert_map is not None:
            logger.warning_once(
                "The PPLX backend does not support expert mapping. "
                "The provided `expert_map` will be ignored."
            )
        expert_map = None  # noqa: F841

        # Is this always going to be a1.device?
        device = a1.device

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        repeat_cols = 4
        repeat_rows = 1 if quant_config.per_act_token_quant else a1.size(0)
        # TODO(bnell): always pass quant_config.a1_scale?
        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            (None if quant_config.per_act_token_quant else quant_config.a1_scale),
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
        )

        _validate_scale_shape(
            a1q, a1q_scale, quant_config.per_act_token_quant, quant_config.block_shape
        )
        if a1q_scale is not None:
            assert torch.isfinite(a1q_scale).all(), (
                "Found non-finite values in PPLX a1q_scale."
            )
        has_non_finite_a1q = _has_non_finite(a1q)
        if _pplx_debug_enabled() and has_non_finite_a1q:
            logger.warning(
                "PPLX_DEBUG_PREPARE found non-finite a1q values. a1=%s a1q=%s",
                _tensor_stats(a1),
                _tensor_stats(a1q),
            )

        orig_a_scale_block_shape: int | None = None

        if a1q_scale is not None:
            scalar_scales = a1q_scale.numel() == 1

            # pplx requires 2-d scales even for scalar scales
            if a1q_scale.dim() <= 1:
                assert scalar_scales
                a1q_scale = a1q_scale.view(1, 1)

            orig_a_scale_block_shape = a1q_scale.shape[-1]

            if not quant_config.is_block_quantized:
                # TODO (bnell): use group_broadcast instead?
                a1q_scale = a1q_scale.repeat(repeat_rows, repeat_cols)
                assert a1q_scale.shape[0] == a1q.shape[0], (
                    "PPLX expects per-token or per-tensor scales to have one "
                    "row per token after repeat."
                )
                assert a1q_scale.shape[1] == repeat_cols, (
                    "PPLX expects per-token or per-tensor scales to have "
                    f"{repeat_cols} columns after repeat."
                )

        assert a1q_scale is None or a1q_scale.ndim == 2, (
            f"{0 if a1q_scale is None else (a1q_scale.ndim, a1q_scale.shape)}"
        )
        if quant_config.is_block_quantized and a1q_scale is not None:
            block_shape = quant_config.block_shape
            assert block_shape is not None
            expected_scale_shape = (
                a1q.shape[0],
                cdiv(a1q.shape[1], block_shape[1]),
            )
            assert a1q_scale.shape == expected_scale_shape, (
                "PPLX block-quant scale shape mismatch: "
                f"{a1q_scale.shape} vs {expected_scale_shape}."
            )
            expected_stride = (expected_scale_shape[1], 1)
            assert a1q_scale.stride() == expected_stride, (
                "PPLX block-quant scale stride mismatch: "
                f"{a1q_scale.stride()} vs {expected_stride}."
            )

        expert_num_tokens = torch.empty(
            self.num_local_experts,
            dtype=torch.int32,
            device=device,
        )

        expert_x = torch.empty(
            (
                self.num_local_experts,
                self.max_num_tokens * self.num_dispatchers(),
                hidden_dim,
            ),
            dtype=a1q.dtype,
            device=device,
        )
        # Initialize to avoid propagating uninitialized padding into CUTLASS.
        expert_x.zero_()

        expert_x_scale: torch.Tensor | None = None
        if a1q.dtype.itemsize == 1:
            if quant_config.is_per_act_token or quant_config.is_per_tensor:
                # PPLX expects 4 FP32 scales per token for non-block quant.
                final_dim = repeat_cols
            else:
                # (M x K_tiles) -> (E x M x K_tiles)
                assert quant_config.block_shape is not None
                num_blocks = cdiv(expert_x.size(2), quant_config.block_shape[1])
                final_dim = num_blocks

            expert_x_scale_shape = (
                self.num_local_experts,
                expert_x.size(1),
                round_up(final_dim, repeat_cols),  # round up for alignment
            )

            expert_x_scale = torch.empty(
                expert_x_scale_shape,
                dtype=torch.float32,
                device=expert_x.device,
            )
            # Initialize scales to avoid undefined padding values.
            expert_x_scale.zero_()

        # This argument is optional, defaults to indices.size(0)
        # There's not much point setting this unless it is != indices.size(0)
        bound_m: torch.Tensor | None = None

        self.a2a.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=a1q,
            dp_x_scale=a1q_scale,
            indices=topk_ids,
            bound_m=bound_m,
            do_send=True,
            do_recv=False,
        )

        hook = lambda: self.a2a.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=a1q,
            dp_x_scale=a1q_scale,
            indices=topk_ids,
            bound_m=bound_m,
            do_send=False,
            do_recv=True,
        )

        return (
            hook,
            lambda: self._receiver(
                expert_num_tokens,
                expert_x,
                expert_x_scale,
                orig_a_scale_block_shape,
            ),
        )

    def _receiver(
        self,
        expert_num_tokens: torch.Tensor,
        expert_x: torch.Tensor,
        expert_x_scale: torch.Tensor | None,
        orig_a_scale_block_shape: int | None,
    ) -> mk.PrepareResultType:
        if expert_x_scale is not None:
            expert_x_scale = expert_x_scale[:, :, :orig_a_scale_block_shape]
            assert expert_x_scale.ndim == 3
            if not expert_x_scale.is_contiguous():
                expert_x_scale = expert_x_scale.contiguous()

        if expert_num_tokens.numel() != 0:
            expert_num_tokens_i64 = expert_num_tokens.to(torch.int64)
            expert_num_tokens_min = int(expert_num_tokens_i64.min().item())
            expert_num_tokens_max = int(expert_num_tokens_i64.max().item())
            assert expert_num_tokens_min >= 0, (
                "PPLX saw negative expert_num_tokens. "
                "Check dispatch indices or token counts."
            )
            max_tokens = self.max_num_tokens * self.num_dispatchers()
            assert expert_num_tokens_max <= max_tokens, (
                "PPLX saw expert_num_tokens out of range. "
                f"max={expert_num_tokens_max} limit={max_tokens}."
            )

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
        )

        return expert_x, expert_x_scale, expert_tokens_meta, None, None

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        hook, receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant=defer_input_quant,
        )
        hook()
        return receiver()

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the combine kernel."
        )

        # This argument is optional
        # There's not much point setting this unless it is != topk_ids.size(0)
        bound_m: torch.Tensor | None = None

        # TODO (bnell): fails in test_pplx_moe.py, figure out what's going on
        # num_tokens = output.size(0)  # M
        # assert topk_ids.size(0) == num_tokens, (
        #    f"{topk_ids.size(0)} == {num_tokens}")
        assert topk_ids.size() == topk_weights.size(), (
            f"{topk_ids.size()} == {topk_weights.size()}"
        )
        assert output.size(0) <= self.max_num_tokens, (
            f"{output.size(0)} <= {self.max_num_tokens}"
        )
        assert output.size(1) == fused_expert_output.size(-1)

        # For multi-GPU: ensure output buffer is zeroed before combine
        # to avoid accumulation issues in PPLX combine
        # PPLX combine may accumulate into output, so we need to zero it first
        output.zero_()

        # Set weights to 1 if we did them in dispatch. This is hacky.
        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        assert topk_ids.dtype == torch.uint32, (
            "PPLX combine expects topk_ids dtype torch.uint32. "
            "Check router indices dtype conversion for PPLX all2all."
        )
        assert topk_ids.is_contiguous(), (
            "PPLX combine expects topk_ids to be contiguous."
        )
        assert topk_ids.stride(-1) == 1, (
            f"PPLX combine expects topk_ids stride(-1) == 1, got {topk_ids.stride()}."
        )
        assert topk_ids.size() == topk_weights.size(), (
            f"{topk_ids.size()} == {topk_weights.size()}"
        )
        assert topk_weights.is_contiguous(), (
            "PPLX combine expects topk_weights to be contiguous."
        )
        assert topk_weights.stride(-1) == 1, (
            "PPLX combine expects topk_weights stride(-1) == 1, got "
            f"{topk_weights.stride()}."
        )
        topk_ids_u32 = topk_ids.view(dtype=torch.uint32)
        if _pplx_debug_enabled():
            _maybe_log_combine_first_context(
                stage="before_send",
                layer_id=self.layer_id,
                layer_name=self.layer_name,
                apply_router_weight_on_input=apply_router_weight_on_input,
                topk_ids=topk_ids_u32,
                topk_weights=topk_weights,
                fused_expert_output=fused_expert_output,
                output=output,
            )

        self.a2a.combine(
            out_tokens=output,
            indices=topk_ids_u32,
            weights=topk_weights,
            expert_y=fused_expert_output,
            bound_m=bound_m,
            do_send=True,
            do_recv=False,
        )

        # For multi-GPU: synchronize before recv to ensure send completes
        # This prevents data alignment and synchronization issues in PPLX combine
        from vllm.utils.torch_utils import current_stream

        def recv_combine():
            # Ensure send phase completes before recv
            current_stream().synchronize()
            output_before_recv = (
                output.detach().clone() if _pplx_debug_enabled() else None
            )
            self.a2a.combine(
                out_tokens=output,
                indices=topk_ids_u32,
                weights=topk_weights,
                expert_y=fused_expert_output,
                bound_m=bound_m,
                do_send=False,
                do_recv=True,
            )
            if _pplx_debug_enabled():
                _maybe_log_combine_first_context(
                    stage="after_recv",
                    layer_id=self.layer_id,
                    layer_name=self.layer_name,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    topk_ids=topk_ids_u32,
                    topk_weights=topk_weights,
                    fused_expert_output=fused_expert_output,
                    output=output,
                    output_before_recv=output_before_recv,
                )

        return recv_combine

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        receiver = self.finalize_async(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        receiver()
