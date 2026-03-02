# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..inductor_pass import enable_fake_mode, get_pass_context
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

FP8_DTYPE = current_platform.fp8_dtype()

logger = init_logger(__name__)


class BasePattern:
    def __init__(self, dtype: torch.dtype, device: str | None) -> None:
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def _compute_ragged_split_sizes(self) -> list[int] | None:
        try:
            compile_range = get_pass_context().compile_range
        except AssertionError:
            return None
        if not compile_range.is_single_size():
            return None
        num_tokens = compile_range.end
        if num_tokens % self.tp_size == 0:
            return None
        base = num_tokens // self.tp_size
        remainder = num_tokens % self.tp_size
        return [base + (1 if rank < remainder else 0) for rank in range(self.tp_size)]


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


class Nvfp4MatmulReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        a_q = torch.empty([16, 16], device=self.device, dtype=torch.uint8)
        b_q = torch.empty([16, 16], device=self.device, dtype=torch.uint8)
        a_scale = torch.empty([128, 4], device=self.device, dtype=torch.float8_e4m3fn)
        b_scale = torch.empty([16, 4], device=self.device, dtype=torch.uint8)
        alpha = torch.empty([1], device=self.device, dtype=torch.float32)
        return [a_q, b_q, a_scale, b_scale, alpha]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            a_q: torch.Tensor,
            b_q: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            mm = torch.ops.vllm.matmul_nvf4_bf16.default(
                a_q, b_q, a_scale, b_scale, alpha
            )
            return torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        def replacement(
            a_q: torch.Tensor,
            b_q: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops._C.fused_nvf4_matmul_reduce_scatter(
                a_q,
                b_q,
                a_scale,
                b_scale,
                alpha,
                "sum",
                scatter_dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )

        split_sizes = self._compute_ragged_split_sizes()
        if split_sizes is None:
            return

        def pattern_v(
            a_q: torch.Tensor,
            b_q: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            mm = torch.ops.vllm.matmul_nvf4_bf16.default(
                a_q, b_q, a_scale, b_scale, alpha
            )
            return torch.ops.vllm.reduce_scatterv.default(
                mm,
                dim=0,
                sizes=split_sizes,
                group_name=self.tp.unique_name,
            )

        pm.register_replacement(
            pattern_v, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherNvfp4QuantizeMatmulPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        a_shard = torch.empty([8, 64], device=self.device, dtype=torch.bfloat16)
        hadamard_matrix = torch.empty(
            [64, 64], device=self.device, dtype=torch.bfloat16
        )
        act_global_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        b_q = torch.empty([16, 32], device=self.device, dtype=torch.uint8)
        b_scale = torch.empty([16, 4], device=self.device, dtype=torch.uint8)
        weight_global_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        return [
            a_shard,
            hadamard_matrix,
            act_global_scale,
            b_q,
            b_scale,
            weight_global_scale,
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            a_shard: torch.Tensor,
            hadamard_matrix: torch.Tensor,
            act_global_scale: torch.Tensor,
            b_q: torch.Tensor,
            b_scale: torch.Tensor,
            weight_global_scale: torch.Tensor,
        ) -> torch.Tensor:
            gathered = torch.ops.vllm.all_gather.default(
                a_shard,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            a_q, a_scale = torch.ops.vllm.fused_quantize_nv.default(
                gathered, hadamard_matrix, act_global_scale
            )
            alpha = 1 / (weight_global_scale * act_global_scale)
            return torch.ops.vllm.matmul_nvf4_bf16.default(
                a_q, b_q, a_scale, b_scale, alpha
            )

        def replacement(
            a_shard: torch.Tensor,
            hadamard_matrix: torch.Tensor,
            act_global_scale: torch.Tensor,
            b_q: torch.Tensor,
            b_scale: torch.Tensor,
            weight_global_scale: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops._C.fused_all_gather_quantize_nvf4_matmul(
                a_shard,
                hadamard_matrix,
                act_global_scale,
                b_q,
                b_scale,
                weight_global_scale,
                gather_dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )

        split_sizes = self._compute_ragged_split_sizes()
        if split_sizes is None:
            return

        def pattern_v(
            a_shard: torch.Tensor,
            hadamard_matrix: torch.Tensor,
            act_global_scale: torch.Tensor,
            b_q: torch.Tensor,
            b_scale: torch.Tensor,
            weight_global_scale: torch.Tensor,
        ) -> torch.Tensor:
            gathered = torch.ops.vllm.all_gatherv.default(
                a_shard,
                dim=0,
                sizes=split_sizes,
                group_name=self.tp.unique_name,
            )
            a_q, a_scale = torch.ops.vllm.fused_quantize_nv.default(
                gathered, hadamard_matrix, act_global_scale
            )
            alpha = 1 / (weight_global_scale * act_global_scale)
            return torch.ops.vllm.matmul_nvf4_bf16.default(
                a_q, b_q, a_scale, b_scale, alpha
            )

        pm.register_replacement(
            pattern_v, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class ScaledFp4QuantFlashinferMMReduceScatterPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        a = torch.empty([16, 64], device=self.device, dtype=torch.bfloat16)
        a_q = torch.empty([16, 32], device=self.device, dtype=torch.uint8)
        a_scale = torch.empty([128, 1], device=self.device, dtype=torch.int32)
        input_global_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        b_t = torch.empty([32, 16], device=self.device, dtype=torch.uint8)
        b_scale_t = torch.empty([4, 16], device=self.device, dtype=torch.float8_e4m3fn)
        alpha = torch.empty([1], device=self.device, dtype=torch.float32)
        return [a, a_q, a_scale, input_global_scale, b_t, b_scale_t, alpha]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            a: torch.Tensor,
            a_q: torch.Tensor,
            a_scale: torch.Tensor,
            input_global_scale: torch.Tensor,
            b_t: torch.Tensor,
            b_scale_t: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            quant_out = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.scaled_fp4_quant.default,
                output=a_q,
                input=a,
                output_scale=a_scale,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=True,
            )
            mm = torch.ops.vllm.flashinfer_mm_fp4.default(
                quant_out[1],
                b_t,
                quant_out[2].view(torch.float8_e4m3fn),
                b_scale_t,
                alpha,
                self.dtype,
                False,
                "cutlass",
            )
            return torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        def replacement(
            a: torch.Tensor,
            a_q: torch.Tensor,
            a_scale: torch.Tensor,
            input_global_scale: torch.Tensor,
            b_t: torch.Tensor,
            b_scale_t: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            del a_q, a_scale
            return torch.ops._C.fused_scaled_fp4_quant_flashinfer_mm_reduce_scatter(
                a,
                b_t,
                input_global_scale,
                b_scale_t,
                alpha,
                "sum",
                scatter_dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
                backend="cutlass",
            )

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )

        split_sizes = self._compute_ragged_split_sizes()
        if split_sizes is None:
            return

        def pattern_v(
            a: torch.Tensor,
            a_q: torch.Tensor,
            a_scale: torch.Tensor,
            input_global_scale: torch.Tensor,
            b_t: torch.Tensor,
            b_scale_t: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            quant_out = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.scaled_fp4_quant.default,
                output=a_q,
                input=a,
                output_scale=a_scale,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=True,
            )
            mm = torch.ops.vllm.flashinfer_mm_fp4.default(
                quant_out[1],
                b_t,
                quant_out[2].view(torch.float8_e4m3fn),
                b_scale_t,
                alpha,
                self.dtype,
                False,
                "cutlass",
            )
            return torch.ops.vllm.reduce_scatterv.default(
                mm,
                dim=0,
                sizes=split_sizes,
                group_name=self.tp.unique_name,
            )

        pm.register_replacement(
            pattern_v, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllGatherScaledFp4QuantFlashinferMMPattern(BasePattern):
    def get_inputs(self) -> list[torch.Tensor]:
        a_shard = torch.empty([8, 64], device=self.device, dtype=torch.bfloat16)
        a_q = torch.empty([16, 32], device=self.device, dtype=torch.uint8)
        a_scale = torch.empty([128, 1], device=self.device, dtype=torch.int32)
        input_global_scale = torch.empty([1], device=self.device, dtype=torch.float32)
        b_t = torch.empty([32, 16], device=self.device, dtype=torch.uint8)
        b_scale_t = torch.empty([4, 16], device=self.device, dtype=torch.float8_e4m3fn)
        alpha = torch.empty([1], device=self.device, dtype=torch.float32)
        return [a_shard, a_q, a_scale, input_global_scale, b_t, b_scale_t, alpha]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            a_shard: torch.Tensor,
            a_q: torch.Tensor,
            a_scale: torch.Tensor,
            input_global_scale: torch.Tensor,
            b_t: torch.Tensor,
            b_scale_t: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            gathered = torch.ops.vllm.all_gather.default(
                a_shard,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            quant_out = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.scaled_fp4_quant.default,
                output=a_q,
                input=gathered,
                output_scale=a_scale,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=True,
            )
            return torch.ops.vllm.flashinfer_mm_fp4.default(
                quant_out[1],
                b_t,
                quant_out[2].view(torch.float8_e4m3fn),
                b_scale_t,
                alpha,
                self.dtype,
                False,
                "cutlass",
            )

        def replacement(
            a_shard: torch.Tensor,
            a_q: torch.Tensor,
            a_scale: torch.Tensor,
            input_global_scale: torch.Tensor,
            b_t: torch.Tensor,
            b_scale_t: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            del a_q, a_scale
            return torch.ops._C.fused_all_gather_scaled_fp4_quant_flashinfer_mm(
                a_shard,
                b_t,
                input_global_scale,
                b_scale_t,
                alpha,
                gather_dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
                backend="cutlass",
            )

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )

        split_sizes = self._compute_ragged_split_sizes()
        if split_sizes is None:
            return

        def pattern_v(
            a_shard: torch.Tensor,
            a_q: torch.Tensor,
            a_scale: torch.Tensor,
            input_global_scale: torch.Tensor,
            b_t: torch.Tensor,
            b_scale_t: torch.Tensor,
            alpha: torch.Tensor,
        ) -> torch.Tensor:
            gathered = torch.ops.vllm.all_gatherv.default(
                a_shard,
                dim=0,
                sizes=split_sizes,
                group_name=self.tp.unique_name,
            )
            quant_out = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.scaled_fp4_quant.default,
                output=a_q,
                input=gathered,
                output_scale=a_scale,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=True,
            )
            return torch.ops.vllm.flashinfer_mm_fp4.default(
                quant_out[1],
                b_t,
                quant_out[2].view(torch.float8_e4m3fn),
                b_scale_t,
                alpha,
                self.dtype,
                False,
                "cutlass",
            )

        pm.register_replacement(
            pattern_v, replacement, self.get_inputs(), pm.fwd_only, pm_pass
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

        has_nvf4_fused_ops = (
            hasattr(torch.ops._C, "fused_nvf4_matmul_reduce_scatter")
            and hasattr(torch.ops._C, "fused_all_gather_quantize_nvf4_matmul")
            and hasattr(torch.ops.vllm, "fused_quantize_nv")
            and hasattr(torch.ops.vllm, "matmul_nvf4_bf16")
        )
        has_flashinfer_nvf4_fused_ops = (
            hasattr(torch.ops._C, "fused_scaled_fp4_quant_flashinfer_mm_reduce_scatter")
            and hasattr(torch.ops._C, "fused_all_gather_scaled_fp4_quant_flashinfer_mm")
            and hasattr(torch.ops._C, "scaled_fp4_quant")
            and hasattr(torch.ops.vllm, "flashinfer_mm_fp4")
        )
        if self.model_dtype == torch.bfloat16 and has_nvf4_fused_ops:
            Nvfp4MatmulReduceScatterPattern(self.model_dtype, self.device).register(
                self.patterns
            )
            AllGatherNvfp4QuantizeMatmulPattern(self.model_dtype, self.device).register(
                self.patterns
            )
        if self.model_dtype == torch.bfloat16 and has_flashinfer_nvf4_fused_ops:
            ScaledFp4QuantFlashinferMMReduceScatterPattern(
                self.model_dtype, self.device
            ).register(self.patterns)
            AllGatherScaledFp4QuantFlashinferMMPattern(
                self.model_dtype, self.device
            ).register(self.patterns)

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
        if not compile_range.is_single_size():
            return False
        if envs.VLLM_ENABLE_SP_RAGGED:
            return True
        tp_size = get_tensor_model_parallel_world_size()
        return bool(compile_range.end % tp_size == 0)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
