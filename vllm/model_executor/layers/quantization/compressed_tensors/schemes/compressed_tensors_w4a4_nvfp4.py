# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E501
    run_nvfp4_emulations,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    cutlass_fp4_supported,
    swizzle_blockscale,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A4Fp4"]


class CompressedTensorsW4A4Fp4(CompressedTensorsScheme):
    def __init__(self):
        self.backend = "none"
        if envs.VLLM_NVFP4_GEMM_BACKEND is None:
            if has_flashinfer():
                self.backend = "flashinfer-cutlass"
            elif cutlass_fp4_supported():
                self.backend = "cutlass"
        elif envs.VLLM_USE_FBGEMM:
            self.backend = "fbgemm"
            try:
                import fbgemm_gpu  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Backend fbgemm requires fbgemm.f4f4bf16 operator, "
                    "Please install with: pip install fbgemm-gpu-genai"
                ) from exc
        elif envs.VLLM_NVFP4_GEMM_BACKEND.startswith("flashinfer-"):
            self.backend = envs.VLLM_NVFP4_GEMM_BACKEND
            assert has_flashinfer(), f"FlashInfer is required for {self.backend}"
        elif envs.VLLM_NVFP4_GEMM_BACKEND == "cutlass":
            self.backend = "cutlass"
            assert cutlass_fp4_supported(), f"Cutlass is required for {self.backend}"

        if self.backend == "none":
            raise ValueError(
                "No valid NVFP4 GEMM backend found. "
                "Please check your platform capability."
            )

        logger.info_once(f"Using {self.backend} for NVFP4 GEMM")
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        if envs.VLLM_USE_NVFP4_CT_EMULATIONS:
            return 80
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        # Store original output_size_per_partition for weight loading
        # Alignment will be done in process_weights_after_loading
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer) -> None:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            align_dim_for_cutlass,
        )

        global_input_scale = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(global_input_scale, requires_grad=False)

        layer.weight_global_scale = Parameter(
            layer.weight_global_scale.max().to(torch.float32), requires_grad=False
        )

        # Ensure alignment for CUTLASS kernel
        # Note: We don't truncate weights here to avoid breaking mamba_mixer2
        # split operations. Instead, we handle alignment in apply_weights.
        # Check if backend uses CUTLASS (either "cutlass" or "flashinfer-cutlass")
        uses_cutlass = self.backend == "cutlass" or (
            self.backend.startswith("flashinfer-")
            and self.backend[len("flashinfer-") :] == "cutlass"
        )
        if uses_cutlass:
            weight_data = layer.weight_packed.data
            original_output_size = weight_data.shape[0]
            aligned_output_size = align_dim_for_cutlass(original_output_size)
            if aligned_output_size != original_output_size:
                logger.warning(
                    "[CompressedTensorsW4A4Fp4.process_weights_after_loading] "
                    "Weight output_size=%s is not aligned to 32. "
                    "Will handle alignment in apply_weights to avoid breaking "
                    "mamba_mixer2 split operations.",
                    original_output_size,
                )
                # Store aligned size for use in apply_weights
                layer._aligned_output_size = aligned_output_size
            else:
                layer._aligned_output_size = None

        if self.backend == "flashinfer-trtllm":
            # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
            # FlashInfer provides nvfp4_quantize to quantize + shuffle the
            # layout but we use our own quantization so we have to call
            # shuffles ourselves.
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight_packed.data
            weight_scale = layer.weight_scale.data

            epilogue_tile_m = 128
            weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
            weight_scale = (
                shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
                .reshape(weight_scale.shape)
                .view(torch.float8_e4m3fn)
            )

            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(weight, requires_grad=False)
        else:
            swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
            if self.backend == "fbgemm":
                swizzled_weight_scale = swizzled_weight_scale.view(-1).view(torch.uint8)
            layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(
                layer.weight_packed.data, requires_grad=False
            )

        layer.alpha = Parameter(
            1 / (layer.input_global_scale * layer.weight_global_scale),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if envs.VLLM_USE_NVFP4_CT_EMULATIONS:
            out = run_nvfp4_emulations(
                x=x,
                input_global_scale=layer.input_global_scale,
                weight=layer.weight_packed,
                weight_scale_swizzled=layer.weight_scale,
                weight_global_scale=layer.weight_global_scale,
            )
            if bias is not None:
                out = out + bias
            return out

        output_dtype = x.dtype
        original_output_size = layer.weight_packed.shape[0]

        # Handle alignment for CUTLASS kernel if needed
        # Check if backend uses CUTLASS (either "cutlass" or "flashinfer-cutlass")
        uses_cutlass = self.backend == "cutlass" or (
            self.backend.startswith("flashinfer-")
            and self.backend[len("flashinfer-") :] == "cutlass"
        )
        needs_alignment = False
        aligned_output_size = None
        if uses_cutlass and hasattr(layer, "_aligned_output_size"):
            aligned_output_size = layer._aligned_output_size
            if (
                aligned_output_size is not None
                and aligned_output_size != original_output_size
            ):
                needs_alignment = True
                # Truncate weight for CUTLASS kernel alignment
                weight_packed = layer.weight_packed[:aligned_output_size, :]
                weight_scale = layer.weight_scale[:aligned_output_size, :]
            else:
                weight_packed = layer.weight_packed
                weight_scale = layer.weight_scale
        else:
            weight_packed = layer.weight_packed
            weight_scale = layer.weight_scale

        # Always use original output size for output_shape to preserve dimensions
        # for downstream layers (e.g., mamba_mixer2)
        output_shape = [*x.shape[:-1], original_output_size]

        # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_blockscale = scaled_fp4_quant(x, layer.input_global_scale)

        mm_args = (
            x_fp4,
            weight_packed,
            x_blockscale,
            weight_scale,
            layer.alpha,
            output_dtype,
        )
        if self.backend.startswith("flashinfer-"):
            backend_name = self.backend[len("flashinfer-") :]
            out = flashinfer_scaled_fp4_mm(*mm_args, backend=backend_name)
        elif self.backend == "fbgemm":
            out = torch.ops.fbgemm.f4f4bf16(
                x_fp4,
                weight_packed,
                x_blockscale.view(-1).view(torch.uint8),
                weight_scale,
                layer.alpha,
                use_mx=False,
            ).to(output_dtype)
        else:
            assert self.backend == "cutlass"
            out = cutlass_scaled_fp4_mm(*mm_args)

        # If alignment was needed, pad output back to original size
        if needs_alignment and aligned_output_size < original_output_size:
            # Pad with zeros to restore original dimension
            pad_size = original_output_size - aligned_output_size
            logger.debug(
                "[CompressedTensorsW4A4Fp4.apply_weights] Padding output from %s "
                "to %s (pad_size=%s)",
                aligned_output_size,
                original_output_size,
                pad_size,
            )
            out = torch.nn.functional.pad(
                out, (0, pad_size), mode="constant", value=0.0
            )

        if bias is not None:
            out = out + bias

        # Ensure output shape matches expected dimensions
        result = out.view(*output_shape)
        if result.shape[-1] != original_output_size:
            logger.warning(
                "[CompressedTensorsW4A4Fp4.apply_weights] Output shape mismatch: "
                "expected last dim=%s, got %s, output_shape=%s, out.shape=%s",
                original_output_size,
                result.shape[-1],
                output_shape,
                out.shape,
            )
        return result
