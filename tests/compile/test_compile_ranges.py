# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test for issue #28868: When compiling with ranges, we should pass the range
information to Inductor.

This test verifies that compile ranges information is properly passed to Inductor
when using vLLM's compilation interface.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import fx

from vllm.compilation.compiler_interface import (
    InductorAdaptor,
    InductorStandaloneAdaptor,
)


def create_simple_graph() -> tuple[fx.GraphModule, list[torch.Tensor]]:
    """Create a simple FX graph for testing."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    z = graph.call_function(torch.add, (x, y))
    graph.output(z)
    gm = fx.GraphModule({}, graph)
    example_inputs = [torch.randn(2, 3), torch.randn(2, 3)]
    return gm, example_inputs


@pytest.mark.skipif(
    not hasattr(torch._inductor, "standalone_compile"),
    reason="standalone_compile not available in this PyTorch version",
)
def test_inductor_standalone_adaptor_passes_ranges():
    """Test that InductorStandaloneAdaptor passes ranges to standalone_compile."""
    adaptor = InductorStandaloneAdaptor(save_format="binary")
    adaptor.initialize_cache("", disable_cache=True)

    graph, example_inputs = create_simple_graph()
    compiler_config = {"ranges": {"s0": (1, 10), "s1": (2, 20)}}

    captured_kwargs = {}

    def capture_standalone_compile(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock()

    with patch(
        "torch._inductor.standalone_compile", side_effect=capture_standalone_compile
    ):
        adaptor.compile(graph, example_inputs, compiler_config, key="test_key")

    # Check if ranges were passed in options
    # Issue #28868: ranges should be passed to Inductor
    options = captured_kwargs.get("options", {})
    assert (
        "ranges" in options
        or "range_constraints" in options
        or "compile_ranges" in options
    ), (
        "Range information not passed to standalone_compile. "
        f"Captured kwargs: {captured_kwargs}"
    )


@pytest.mark.skipif(
    not (
        hasattr(torch._inductor, "compile_fx")
        and hasattr(getattr(torch._inductor, "compile_fx", None), "compile_fx")
    ),
    reason="compile_fx not available in this PyTorch version",
)
def test_inductor_adaptor_passes_ranges():
    """Test that InductorAdaptor passes ranges to compile_fx."""
    adaptor = InductorAdaptor()
    adaptor.initialize_cache("", disable_cache=True)

    graph, example_inputs = create_simple_graph()
    compiler_config = {
        "ranges": {"s0": (1, 10), "s1": (2, 20)},
        "force_disable_caches": True,
    }

    captured_kwargs = {}

    def capture_compile_fx(*args, **kwargs):
        captured_kwargs.update(kwargs)
        mock_result = MagicMock()
        mock_result.current_callable = MagicMock()
        mock_result.current_callable.__code__ = MagicMock()
        mock_result.current_callable.__code__.co_filename = "/tmp/test.py"
        mock_result._fx_graph_cache_key = "test_hash"
        return mock_result

    def hijack_compiled_fx_graph_hash(*args, **kwargs):
        return ("test_hash", {})

    with (
        patch("torch._inductor.compile_fx.compile_fx", side_effect=capture_compile_fx),
        patch(
            "torch._inductor.codecache.compiled_fx_graph_hash",
            side_effect=hijack_compiled_fx_graph_hash,
        ),
    ):
        adaptor.compile(graph, example_inputs, compiler_config, key="test_key")

    # Check if ranges were passed in config_patches
    # Issue #28868: ranges should be passed to Inductor
    config = captured_kwargs.get("config_patches", {})
    assert isinstance(config, dict) and (
        "ranges" in config
        or "range_constraints" in config
        or "compile_ranges" in config
    ), f"Range information not passed to compile_fx. Captured kwargs: {captured_kwargs}"
