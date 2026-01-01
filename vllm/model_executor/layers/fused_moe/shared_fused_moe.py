# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.moe_layer import MoELayer


class SharedFusedMoE(torch.nn.Module):
    """
    A FusedMoE operation that also computes the results of shared experts.

    This is a backward-compatible wrapper around MoELayer that maintains
    the original SharedFusedMoE API while using the cleaner MoELayer
    implementation internally.

    If an all2all communicator is being used the shared expert computation
    can be interleaved with the fused all2all dispatch communication step.
    """

    def __init__(
        self,
        shared_experts: torch.nn.Module | None,
        gate: torch.nn.Module | None = None,
        use_overlapped: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Create the routed experts (FusedMoE) instance
        self._routed_experts = FusedMoE(**kwargs)

        # Store original shared_experts and gate for property access
        self._shared_experts = shared_experts
        self._gate = gate

        # Disable shared expert overlap if:
        #   - we are using eplb with non-default backend, because of correctness issues
        #   - we are using flashinfer with DP, since there nothing to gain
        #   - we are using marlin kernels
        backend = self._routed_experts.moe_parallel_config.all2all_backend
        self.use_overlapped = (
            use_overlapped
            and not (
                (
                    self._routed_experts.enable_eplb
                    and backend != "allgather_reducescatter"
                )
                or (
                    self._routed_experts.moe_config.use_flashinfer_cutlass_kernels
                    and self._routed_experts.dp_size > 1
                )
            )
            and self._shared_experts is not None
        )

        # Determine which shared_experts and gate to pass to MoELayer
        # If use_overlapped is False, we handle shared_experts separately in forward
        moe_shared_experts = self._shared_experts if self.use_overlapped else None
        moe_gate = self._gate if self.use_overlapped else None

        # Create MoELayer to manage gate, shared experts, and routed experts
        self._moe_layer = MoELayer(
            routed_experts=self._routed_experts,
            gate=moe_gate,
            shared_experts=moe_shared_experts,
            execution_mode="auto",
        )

    @property
    def shared_experts(self) -> torch.nn.Module | None:
        """Return shared_experts if overlapped, otherwise None (handled separately)."""
        return self._shared_experts if self.use_overlapped else None

    @property
    def gate(self) -> torch.nn.Module | None:
        """Return gate if overlapped, otherwise None (handled separately)."""
        return self._gate if self.use_overlapped else None

    @property
    def is_internal_router(self) -> bool:
        return self.gate is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_overlapped:
            # Non-overlapped mode: run shared experts separately, then routed experts
            if self._shared_experts is not None:
                shared_out = self._shared_experts(hidden_states)

                # Reduce shared expert outputs if necessary, since the MLP
                # should have been created with reduce_results=False.
                if (
                    self._routed_experts.reduce_results
                    and get_tensor_model_parallel_world_size() > 1
                    and self._routed_experts.must_reduce_shared_expert_outputs()
                ):
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
            else:
                shared_out = None

            # Run routed experts only (no shared experts in MoELayer)
            fused_out = self._moe_layer(hidden_states, router_logits)
        else:
            shared_out, fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            # ensure early TP reduction of shared expert outputs when required
            if (
                shared_out is not None
                and self._routed_experts.reduce_results
                and get_tensor_model_parallel_world_size() > 1
                and self._routed_experts.must_reduce_shared_expert_outputs()
            ):
                shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out, fused_out
