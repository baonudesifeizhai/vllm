# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..fx_utils import is_func
from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

FP8_DTYPE = current_platform.fp8_dtype()

logger = init_logger(__name__)

_ASYNC_TP_MATCH_COUNT_ATTR = "_vllm_async_tp_match_count"


def _get_async_tp_match_count(graph: fx.Graph) -> int:
    return int(getattr(graph, _ASYNC_TP_MATCH_COUNT_ATTR, 0))


def _set_async_tp_match_count(graph: fx.Graph, count: int) -> None:
    setattr(graph, _ASYNC_TP_MATCH_COUNT_ATTR, count)


class BasePattern:
    def __init__(self, dtype: torch.dtype, device: str | None) -> None:
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    @staticmethod
    def wrap_trace_fn(trace_fn):
        def wrapped(*args, **kwargs):
            gm = trace_fn(*args, **kwargs)
            from torch._inductor.fx_passes.post_grad import view_to_reshape

            view_to_reshape(gm)
            return gm

        return wrapped


class GEMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        mul = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mm_weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [mul, mm_weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            mm = torch.ops.aten.mm.default(mul, mm_weight)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(mul: torch.Tensor, mm_weight: torch.Tensor) -> torch.Tensor:
            gemm_rs = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                mul,
                mm_weight,
                "sum",
                scatter_dim=0,
                group_name=self.tp.device_group.group_name,
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherGEMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

            return torch.ops.aten.mm.default(all_gather, weight)

        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                x,
                [weight],
                gather_dim=0,
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)
        return [input, mm_weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            scaled_mm = torch.ops.aten._scaled_mm.default(
                input,
                mat2=mat2,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                scale_result=None,
                out_dtype=self.dtype,
            )
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                scaled_mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        return [x, weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )

            return torch.ops.aten._scaled_mm.default(
                all_gather,
                mat2=weight,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=None,
                scale_result=None,
                out_dtype=self.dtype,
            )

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class CutlassScaledMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        cutlass_mm_output = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        return [input, mm_weight, scale_a, scale_b, cutlass_mm_output]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            cutlass_mm_output: torch.Tensor,
        ) -> torch.Tensor:
            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=cutlass_mm_output,
                a=input,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None,
            )

            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                cutlass_scaled_mm[1],
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(
            input: torch.Tensor,
            mat2: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            cutlass_mm_output: torch.Tensor,
        ) -> torch.Tensor:
            # Calculate output shape: input @ mat2 with scatter_dim reduced
            output_shape = [*input.shape[:-1], mat2.shape[1]]
            scatter_dim = 0
            gemm_rs = torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,  # orig_scatter_dim
                scatter_dim,  # scatter_dim_after_maybe_reshape
                self.tp.device_group.group_name,
                output_shape,
                None,  # bias
                None,  # result_scale
                self.dtype,  # out_dtype
                False,  # use_fast_accum
            )

            return gemm_rs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherCutlassScaledMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = (
            torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        s2 = weight.shape[1]
        output = torch.empty([s1, s2], device=self.device, dtype=self.dtype)

        return [x, weight, scale_a, scale_b, output]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x, dim=0, world_size=self.tp_size, group_name=self.tp.unique_name
            )

            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=output,
                a=all_gather,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None,
            )
            return cutlass_scaled_mm[1]

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class FlashInferBMMFP8ReduceScatterPattern(BasePattern):
    """Matches unsqueeze -> unsqueeze -> bmm_fp8 -> view -> reduce_scatter."""

    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        weight = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        scale_a = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        return [input, weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            output_shape = [*input.shape[:-1], weight.shape[1]]
            scatter_dim = 0
            return torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter(
                input,
                weight,
                scale_a,
                scale_b,
                "sum",
                scatter_dim,
                scatter_dim,
                self.tp.device_group.group_name,
                output_shape,
                self.dtype,
            )

        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            input_3d = input.unsqueeze(0)
            weight_3d = weight.unsqueeze(0)
            bmm_result = torch.ops.vllm.bmm_fp8.default(
                input_3d, weight_3d, scale_a, scale_b, self.dtype, "auto"
            )
            mm_result = bmm_result.view(input.shape[0], weight.shape[1])
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm_result,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            BasePattern.wrap_trace_fn(pm.fwd_only),
            pm_pass,
        )


class AllGatherFlashInferBMMFP8Pattern(BasePattern):
    """Matches all_gather -> unsqueeze -> unsqueeze -> bmm_fp8 -> view."""

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        scale_a = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        return [x, weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            ag_output, mm_output = (
                torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul(
                    x,
                    weight,
                    scale_a,
                    scale_b,
                    0,
                    self.tp.device_group.group_name,
                    self.dtype,
                )
            )
            return mm_output

        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            ag_3d = all_gather.unsqueeze(0)
            weight_3d = weight.unsqueeze(0)
            bmm_result = torch.ops.vllm.bmm_fp8.default(
                ag_3d, weight_3d, scale_a, scale_b, self.dtype, "auto"
            )
            # Match the real graph shape expression directly. Using
            # all_gather.shape[0] traces as a composed symbolic expression,
            # while the compiled graph keeps the gathered token count as the
            # bmm_result second dimension symbol.
            return bmm_result.view(bmm_result.shape[1], bmm_result.shape[2])

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            BasePattern.wrap_trace_fn(pm.fwd_only),
            pm_pass,
        )


class AsyncTPPass(VllmPatternMatcherPass):
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        # Enable symmetric memory for the TP process group
        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="async_tp_pass"
        )
        GEMMReduceScatterPattern(self.model_dtype, self.device).register(self.patterns)

        AllGatherGEMMPattern(self.model_dtype, self.device).register(self.patterns)

        # These fusions are enabled only for bfloat16 models because
        # `scaled_mm` or `cutlass_scaled_mm` with per-token (row-wise) scaling
        # only supports bfloat16 as the output dtype.
        if self.model_dtype == torch.bfloat16:
            ScaledMMReduceScatterPattern(self.model_dtype, self.device).register(
                self.patterns
            )
            AllGatherScaledMMPattern(self.model_dtype, self.device).register(
                self.patterns
            )

            CutlassScaledMMReduceScatterPattern(self.model_dtype, self.device).register(
                self.patterns
            )
            AllGatherCutlassScaledMMPattern(self.model_dtype, self.device).register(
                self.patterns
            )

            if hasattr(torch.ops.vllm, "bmm_fp8"):
                FlashInferBMMFP8ReduceScatterPattern(
                    self.model_dtype, self.device
                ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # This pass is applied on top of the sequence parallelism pass.
        # It inherits the same applicability condition as `SequenceParallelismPass`.
        # See `SequenceParallelismPass.is_applicable` for more details.
        if (
            not self.compilation_config.splitting_ops
            or self.compilation_config.use_inductor_graph_partition
        ):
            return True
        tp_size = get_tensor_model_parallel_world_size()
        return bool(compile_range.is_single_size() and compile_range.end % tp_size == 0)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        _set_async_tp_match_count(graph, self.matched_count)
        logger.debug("Early async_tp matched %s patterns", self.matched_count)


class LateAsyncTPAllGatherPass(VllmPatternMatcherPass):
    """
    Run late all-gather fusion after other graph-normalizing passes.

    The early AsyncTP pass already handles working reduce-scatter patterns.
    This late pass is intentionally limited to all-gather FlashInfer patterns
    that are more likely to match after downstream graph rewrites.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        self.group_name = self.tp.device_group.group_name

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        if (
            not self.compilation_config.splitting_ops
            or self.compilation_config.use_inductor_graph_partition
        ):
            return True
        tp_size = get_tensor_model_parallel_world_size()
        return bool(compile_range.is_single_size() and compile_range.end % tp_size == 0)

    @staticmethod
    def _is_view_or_reshape(node: fx.Node) -> bool:
        return is_func(node, torch.ops.aten.reshape.default) or is_func(
            node, torch.ops.aten.view.default
        )

    @staticmethod
    def _erase_if_unused(graph: fx.Graph, node: fx.Node | None) -> None:
        if node is not None and len(node.users) == 0:
            graph.erase_node(node)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = 0
        if self.model_dtype != torch.bfloat16 or not hasattr(torch.ops.vllm, "bmm_fp8"):
            total_matches = _get_async_tp_match_count(graph)
            logger.debug("Replaced %s patterns", total_matches)
            return

        for node in list(graph.nodes):
            if not is_func(node, torch.ops.vllm.bmm_fp8.default):
                continue

            if len(node.args) < 4:
                continue

            ag_3d = node.args[0]
            weight_3d = node.args[1]
            scale_a = node.args[2]
            scale_b = node.args[3]

            if not (
                isinstance(ag_3d, fx.Node)
                and isinstance(weight_3d, fx.Node)
                and is_func(ag_3d, torch.ops.aten.unsqueeze.default)
                and is_func(weight_3d, torch.ops.aten.unsqueeze.default)
                and ag_3d.args[1] == 0
                and weight_3d.args[1] == 0
            ):
                continue

            all_gather = ag_3d.args[0]
            weight = weight_3d.args[0]
            if not (
                isinstance(all_gather, fx.Node)
                and isinstance(weight, fx.Node)
                and is_func(all_gather, torch.ops.vllm.all_gather.default)
            ):
                continue

            reshape_users = [
                user for user in list(node.users) if self._is_view_or_reshape(user)
            ]
            if not reshape_users:
                continue

            if any(user.args[0] is not node for user in reshape_users):
                continue

            with graph.inserting_before(reshape_users[0]):
                fused_ag = graph.call_function(
                    torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default,
                    args=(
                        all_gather.args[0],
                        weight,
                        scale_a,
                        scale_b,
                        0,
                        self.group_name,
                        self.dtype,
                    ),
                )
                fused_mm_output = graph.call_function(
                    operator.getitem,
                    args=(fused_ag, 1),
                )

            fused_ag.meta = dict(node.meta)
            fused_mm_output.meta = dict(reshape_users[0].meta)

            for reshape_node in reshape_users:
                reshape_node.replace_all_uses_with(fused_mm_output)
                graph.erase_node(reshape_node)

            self._erase_if_unused(graph, node)
            self._erase_if_unused(graph, ag_3d)
            self._erase_if_unused(graph, weight_3d)
            self._erase_if_unused(graph, all_gather)

            self.matched_count += 1

        graph.lint()
        total_matches = _get_async_tp_match_count(graph) + self.matched_count
        _set_async_tp_match_count(graph, total_matches)
        logger.debug("Replaced %s patterns", total_matches)
