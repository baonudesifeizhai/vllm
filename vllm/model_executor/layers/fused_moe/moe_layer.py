# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MoELayer: Unified MoE layer manager with clear separation of concerns.

This module provides a clean abstraction for managing the complete MoE layer,
separating the responsibilities of gate, shared experts, and routed experts.
"""

from enum import Enum
from typing import Literal

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.platforms import current_platform
from vllm.utils.torch_utils import aux_stream, current_stream

logger = init_logger(__name__)


class ExecutionMode(Enum):
    """Execution mode for MoE layer."""

    STANDARD = "standard"
    """Standard mode: only routed experts, no shared experts."""

    SHARED_SEQUENTIAL = "shared_sequential"
    """Sequential mode: run shared experts, then routed experts."""

    SHARED_OVERLAP_TP = "shared_overlap_tp"
    """TP overlap mode: run shared experts in separate stream with TP."""

    SHARED_OVERLAP_DP = "shared_overlap_dp"
    """DP overlap mode: run shared experts during dispatch/combine with DP."""


class MoELayer(nn.Module):
    """
    Unified MoE layer manager.

    This class manages the complete MoE layer with clear separation:
    - Gate/Router: computes router logits
    - Shared Experts: optional shared expert computation
    - Routed Experts: main expert computation via FusedMoE

    It replaces SharedFusedMoE with a cleaner design where:
    - FusedMoE only handles routed experts computation
    - MoELayer orchestrates the full layer execution
    - Execution modes are explicit and well-defined

    Args:
        gate: Router/gate module for computing router logits. If None, router_logits
            must be provided in forward().
        shared_experts: Optional shared expert module.
        routed_experts: FusedMoE instance for routed experts (should NOT have
            shared_experts set).
        execution_mode: Execution mode. If "auto", will be determined automatically.
    """

    def __init__(
        self,
        routed_experts: FusedMoE,
        gate: nn.Module | None = None,
        shared_experts: nn.Module | None = None,
        execution_mode: Literal["auto"] | ExecutionMode = "auto",
    ):
        super().__init__()
        self.gate = gate
        self.shared_experts = shared_experts
        self.routed_experts = routed_experts

        # Ensure routed_experts (FusedMoE) doesn't have shared_experts set
        # Note: FusedMoE.shared_experts property always returns None,
        # but we check anyway for clarity and to catch any misuse
        if (
            hasattr(self.routed_experts, "shared_experts")
            and self.routed_experts.shared_experts is not None
        ):
            logger.warning(
                "routed_experts has shared_experts set, "
                "but MoELayer manages shared experts. "
                "This should not happen with a clean FusedMoE instance."
            )

        # Determine execution mode
        if execution_mode == "auto":
            self.execution_mode = self._determine_execution_mode()
        else:
            self.execution_mode = execution_mode

        # Setup shared experts stream for TP overlap mode
        if self.execution_mode == ExecutionMode.SHARED_OVERLAP_TP:
            self.shared_experts_stream = aux_stream()
            if self.shared_experts_stream is None:
                logger.warning(
                    "Shared experts stream not available, "
                    "falling back to sequential mode"
                )
                self.execution_mode = ExecutionMode.SHARED_SEQUENTIAL
        else:
            self.shared_experts_stream = None

        logger.debug(
            "MoELayer initialized with execution_mode=%s",
            self.execution_mode.value,
        )

    def _has_separate_shared_experts(self) -> bool:
        """
        Check if shared experts should be run separately (not by modular kernel).

        Shared experts are separate when:
        - quant_method is NOT FusedMoEModularMethod
          (modular kernel handles it internally)
        - AND shared_experts is not None
        """
        return (
            not isinstance(self.routed_experts.quant_method, FusedMoEModularMethod)
            and self.shared_experts is not None
        )

    def _maybe_setup_shared_experts_stream(
        self,
        hidden_states: torch.Tensor,
        use_chunked_impl: bool = False,
    ) -> tuple[bool, torch.Tensor | None]:
        """
        Determine if shared experts should run in a separate stream and prepare clone.

        Returns:
            (use_stream, hidden_states_clone): Whether to use stream
            and the clone if needed.
        """
        has_separate = self._has_separate_shared_experts()

        use_shared_experts_stream = (
            current_platform.is_cuda()
            and has_separate
            and not use_chunked_impl
            and self.shared_experts_stream is not None
            and (
                hidden_states.shape[0]
                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            )
        )

        hidden_states_clone: torch.Tensor | None = None
        if use_shared_experts_stream:
            assert self.shared_experts_stream is not None

            # Clone BEFORE switching streams to avoid race condition
            # where routed_expert kernel may mutate hidden_states.
            hidden_states_clone = hidden_states.clone()

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html
            hidden_states_clone.record_stream(self.shared_experts_stream)

            # Mark sync start point for the separate shared experts
            # stream here since we want to run in parallel with the
            # router/gate (next op below)
            assert self.shared_experts_stream is not None
            self.shared_experts_stream.wait_stream(current_stream())

        return use_shared_experts_stream, hidden_states_clone

    def _should_disable_overlap(self) -> bool:
        """
        Determine if shared expert overlap should be disabled.

        This integrates the logic from SharedFusedMoE.use_overlapped:
        - EPLB with non-default backend (correctness issues)
        - FlashInfer with DP (no benefit)
        - Other conditions that require sequential execution
        """
        if self.shared_experts is None:
            return True

        # Check EPLB conditions
        if (
            hasattr(self.routed_experts, "enable_eplb")
            and self.routed_experts.enable_eplb
        ):
            backend = self.routed_experts.moe_parallel_config.all2all_backend
            # EPLB with non-default backend: disable overlap for correctness
            return backend != "allgather_reducescatter"

        # Check FlashInfer with DP
        # FlashInfer with DP: no benefit from overlap
        return (
            hasattr(self.routed_experts, "moe_config")
            and hasattr(
                self.routed_experts.moe_config,
                "use_flashinfer_cutlass_kernels",
            )
            and self.routed_experts.moe_config.use_flashinfer_cutlass_kernels
            and self.routed_experts.dp_size > 1
        )

    def _determine_execution_mode(self) -> ExecutionMode:
        """Determine the execution mode based on configuration."""
        if self.shared_experts is None:
            return ExecutionMode.STANDARD

        # Check if overlap should be disabled
        if self._should_disable_overlap():
            return ExecutionMode.SHARED_SEQUENTIAL

        # Get parallel config from routed_experts
        tp_size = self.routed_experts.tp_size
        dp_size = self.routed_experts.dp_size

        # Check if chunking is used
        # (chunked impl needs sequential at this level)
        use_chunked_impl = getattr(
            self.routed_experts, "use_dp_chunking", lambda: False
        )()
        if use_chunked_impl:
            # Chunking implies sequential processing of chunks
            return ExecutionMode.SHARED_SEQUENTIAL

        # Check if overlap is enabled
        # For TP: use separate stream (if separate shared experts)
        if (
            tp_size > 1
            and aux_stream() is not None
            and self._has_separate_shared_experts()
        ):
            return ExecutionMode.SHARED_OVERLAP_TP
        elif dp_size > 1:
            # Check if modular kernel supports shared expert overlap
            if isinstance(self.routed_experts.quant_method, FusedMoEModularMethod):
                # DP case: shared expert handled by modular kernel
                return ExecutionMode.SHARED_OVERLAP_DP
            else:
                # DP case without modular kernel: sequential
                return ExecutionMode.SHARED_SEQUENTIAL
        else:
            return ExecutionMode.SHARED_SEQUENTIAL

    def _run_gate(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run gate/router to get router logits."""
        if router_logits is not None:
            return router_logits

        if self.gate is None:
            raise ValueError("router_logits must be provided if gate is None")

        router_logits, _ = self.gate(hidden_states)
        return router_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MoE layer.

        Args:
            hidden_states: Input hidden states.
            router_logits: Optional pre-computed router logits. If None and gate
                is not None, will be computed from gate.

        Returns:
            Output hidden states. If shared_experts exists, returns tuple
            (shared_out, routed_out), otherwise returns routed_out only.
        """
        # Run gate if needed
        router_logits = self._run_gate(hidden_states, router_logits)

        # Dispatch to appropriate execution mode
        if self.execution_mode == ExecutionMode.STANDARD:
            return self._forward_standard(hidden_states, router_logits)
        elif self.execution_mode == ExecutionMode.SHARED_SEQUENTIAL:
            return self._forward_shared_sequential(hidden_states, router_logits)
        elif self.execution_mode == ExecutionMode.SHARED_OVERLAP_TP:
            return self._forward_shared_overlap_tp(hidden_states, router_logits)
        elif self.execution_mode == ExecutionMode.SHARED_OVERLAP_DP:
            return self._forward_shared_overlap_dp(hidden_states, router_logits)
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}")

    def _forward_standard(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        """Standard mode: only routed experts."""
        return self.routed_experts(hidden_states, router_logits)

    def _forward_shared_sequential(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequential mode: run shared experts, then routed experts."""
        # Run shared experts first
        shared_out = self.shared_experts(hidden_states)  # type: ignore

        # Reduce shared expert outputs if necessary
        if (
            self.routed_experts.reduce_results
            and get_tensor_model_parallel_world_size() > 1
            and self.routed_experts.must_reduce_shared_expert_outputs()
        ):
            shared_out = tensor_model_parallel_all_reduce(shared_out)

        # Run routed experts
        routed_out = self.routed_experts(hidden_states, router_logits)

        return shared_out, routed_out

    def _forward_shared_overlap_tp(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TP overlap mode: run shared experts in separate stream.

        Execution flow:
        1. Setup shared experts stream and clone hidden_states if needed
        2. Start routed experts in main stream
        3. Run shared experts in separate stream (parallel)
        4. Wait for both streams and return results

        Note: This method respects token threshold for stream usage.
        """
        # Check if we should use stream (with token threshold check)
        use_stream, hidden_states_clone = self._maybe_setup_shared_experts_stream(
            hidden_states, use_chunked_impl=False
        )

        if not use_stream or hidden_states_clone is None:
            # Fallback to sequential if stream not available or threshold exceeded
            return self._forward_shared_sequential(hidden_states, router_logits)

        # Start routed experts in main stream
        routed_out = self.routed_experts(hidden_states, router_logits)

        # Run shared experts in separate stream
        # Note: hidden_states_clone is already recorded to the stream
        # in _maybe_setup_shared_experts_stream
        assert self.shared_experts_stream is not None
        with torch.cuda.stream(self.shared_experts_stream):
            shared_out = self.shared_experts(hidden_states_clone)  # type: ignore

        # Wait for shared experts stream to complete
        current_stream().wait_stream(self.shared_experts_stream)

        # Reduce shared expert outputs if necessary
        if (
            self.routed_experts.reduce_results
            and get_tensor_model_parallel_world_size() > 1
            and self.routed_experts.must_reduce_shared_expert_outputs()
        ):
            shared_out = tensor_model_parallel_all_reduce(shared_out)

        return shared_out, routed_out

    def _forward_shared_overlap_dp(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DP overlap mode: run shared experts during dispatch/combine.

        In this mode, the modular kernel (FusedMoEModularMethod) handles
        shared expert computation during the dispatch/combine phase.
        We need to pass shared_experts to the modular kernel.
        """
        # For DP case with modular kernel, we need to set shared_experts
        # on the modular kernel so it can handle the overlap
        from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
            FusedMoEModularMethod,
        )

        # Set shared_experts on the modular kernel if not already set
        if (
            isinstance(self.routed_experts.quant_method, FusedMoEModularMethod)
            and hasattr(
                self.routed_experts.quant_method.fused_experts,
                "shared_experts",
            )
            and self.routed_experts.quant_method.fused_experts.shared_experts
            != self.shared_experts
        ):
            self.routed_experts.quant_method.fused_experts.shared_experts = (
                self.shared_experts
            )

        # Call routed_experts, which will handle the overlap internally
        # via modular kernel
        result = self.routed_experts(hidden_states, router_logits)

        # The result should be a tuple (shared_out, routed_out) in this mode
        if isinstance(result, tuple):
            return result
        else:
            # Fallback: if routed_experts doesn't return tuple, run shared separately
            logger.warning(
                "DP overlap mode expected tuple return from routed_experts, "
                "falling back to sequential"
            )
            return self._forward_shared_sequential(hidden_states, router_logits)
