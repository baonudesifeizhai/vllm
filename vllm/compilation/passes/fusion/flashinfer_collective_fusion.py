# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.fx.experimental.symbolic_shapes import statically_known_true

from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

from ..fx_utils import is_func

FLASHINFER_BMM_FP8_MIN_M = 64
FLASHINFER_BMM_FP8_DTYPE = torch.float8_e4m3fn
VIEW_LIKE_OPS = (
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.default,
)
LAYOUT_PRESERVING_OPS = (
    torch.ops.aten.contiguous.default,
    torch.ops.aten.clone.default,
)


def _get_node_arg(node: fx.Node, name: str, index: int) -> object:
    return node.kwargs.get(name, node.args[index] if len(node.args) > index else None)


def _is_view_like(node: fx.Node) -> bool:
    return any(is_func(node, op) for op in VIEW_LIKE_OPS)


def _is_passthrough(node: fx.Node) -> bool:
    return _is_view_like(node) or any(is_func(node, op) for op in LAYOUT_PRESERVING_OPS)


def _strip_view_like(node: fx.Node) -> fx.Node:
    while _is_view_like(node):
        parent = node.args[0]
        if not isinstance(parent, fx.Node):
            break
        node = parent
    return node


@dataclass(frozen=True)
class _UserPathState:
    node: fx.Node
    saw_slice_scatter: bool = False


def _walk_user_paths(
    start_nodes: list[fx.Node],
    *,
    track_slice_scatter: bool = False,
) -> list[_UserPathState]:
    worklist = [_UserPathState(node=user) for user in start_nodes]
    visited: set[_UserPathState] = set()
    reachable: list[_UserPathState] = []

    while worklist:
        state = worklist.pop()
        if state in visited:
            continue
        visited.add(state)
        reachable.append(state)

        if _is_passthrough(state.node):
            worklist.extend(
                _UserPathState(
                    node=child,
                    saw_slice_scatter=state.saw_slice_scatter,
                )
                for child in state.node.users
            )
            continue

        if track_slice_scatter and is_func(
            state.node, torch.ops.aten.slice_scatter.default
        ):
            worklist.extend(
                _UserPathState(node=child, saw_slice_scatter=True)
                for child in state.node.users
            )

    return reachable


def _node_shape(node: fx.Node) -> list[object] | None:
    val = node.meta.get("val")
    if hasattr(val, "shape"):
        return list(val.shape)
    return None


def _node_first_dim(node: fx.Node) -> object | None:
    shape = _node_shape(node)
    if shape:
        return shape[0]
    return None


def _node_ndim(node: fx.Node) -> int | None:
    shape = _node_shape(node)
    if shape is None:
        return None
    return len(shape)


def _dim_is_statically_lt(dim: int | torch.SymInt, threshold: int) -> bool:
    if isinstance(dim, int):
        return dim < threshold
    try:
        return bool(statically_known_true(dim < threshold))
    except Exception:
        return False


def _passes_min_m(node: fx.Node) -> bool:
    gemm_m = _node_first_dim(node)
    if gemm_m is None or not isinstance(gemm_m, int | torch.SymInt):
        return True
    return not _dim_is_statically_lt(gemm_m, FLASHINFER_BMM_FP8_MIN_M)


def _passes_min_m_after_reduce_scatter(node: fx.Node, world_size: int) -> bool:
    gemm_m = _node_first_dim(node)
    if gemm_m is None or not isinstance(gemm_m, int | torch.SymInt):
        return True
    return not _dim_is_statically_lt(gemm_m, FLASHINFER_BMM_FP8_MIN_M * world_size)


def _copy_replacement_meta(src: fx.Node, dst: fx.Node) -> None:
    dst.meta = {
        key: value for key, value in src.meta.items() if key != "eager_input_vals"
    }


def _arg_numel_is_one(arg: object) -> bool:
    if isinstance(arg, torch.Tensor):
        return arg.numel() == 1
    if isinstance(arg, fx.Node):
        val = arg.meta.get("val")
        if hasattr(val, "numel"):
            return val.numel() == 1
    return False


def _unwrap_bmm_fp8_arg_to_2d(arg: object) -> fx.Node | None:
    if not isinstance(arg, fx.Node):
        return None

    node = _strip_view_like(arg)
    if is_func(node, torch.ops.aten.unsqueeze.default):
        dim = _get_node_arg(node, "dim", 1)
        if dim != 0:
            return None
        src = _get_node_arg(node, "self", 0)
        if not isinstance(src, fx.Node):
            return None
        src = _strip_view_like(src)
        ndim = _node_ndim(src)
        if ndim is not None and ndim != 2:
            return None
        return src

    ndim = _node_ndim(node)
    if ndim is not None and ndim != 2:
        return None
    return node


@dataclass
class _CollectiveOp:
    input_node: fx.Node
    dim: object
    world_size: object
    group_name: object


@dataclass
class _BmmFp8Op:
    a_2d: fx.Node
    b_2d: fx.Node
    a_scale: object
    b_scale: object
    out_dtype: object
    backend: object


def _parse_collective_op(
    node: fx.Node,
    op,
) -> _CollectiveOp | None:
    if not is_func(node, op):
        return None

    input_node = _get_node_arg(node, "tensor", 0)
    dim = _get_node_arg(node, "dim", 1)
    world_size = _get_node_arg(node, "world_size", 2)
    group_name = _get_node_arg(node, "group_name", 3)
    if not isinstance(input_node, fx.Node):
        return None
    return _CollectiveOp(
        input_node=input_node,
        dim=dim,
        world_size=world_size,
        group_name=group_name,
    )


def _parse_all_gather(
    node: fx.Node,
) -> _CollectiveOp | None:
    return _parse_collective_op(node, torch.ops.vllm.all_gather.default)


def _parse_collective_group_name(collective: _CollectiveOp) -> str | None:
    if (
        collective.dim != 0
        or not isinstance(collective.world_size, int)
        or not isinstance(collective.group_name, str)
    ):
        return None
    return collective.group_name


def _parse_bmm_fp8(
    node: fx.Node,
) -> _BmmFp8Op | None:
    if not is_func(node, torch.ops.vllm.bmm_fp8.default):
        return None

    a = _get_node_arg(node, "A", 0)
    b = _get_node_arg(node, "B", 1)
    a_scale = _get_node_arg(node, "A_scale", 2)
    b_scale = _get_node_arg(node, "B_scale", 3)
    out_dtype = _get_node_arg(node, "dtype", 4)
    backend = _get_node_arg(node, "backend", 5)

    a_2d = _unwrap_bmm_fp8_arg_to_2d(a)
    b_2d = _unwrap_bmm_fp8_arg_to_2d(b)
    if a_2d is None or b_2d is None:
        return None
    return _BmmFp8Op(
        a_2d=a_2d,
        b_2d=b_2d,
        a_scale=a_scale,
        b_scale=b_scale,
        out_dtype=out_dtype,
        backend=backend,
    )


def _flashinfer_bmm_fp8_extra_check(match: pm.Match) -> bool:
    for node in match.nodes:
        if not is_func(node, torch.ops.vllm.bmm_fp8.default):
            continue
        parsed = _parse_bmm_fp8(node)
        return (
            parsed is not None
            and parsed.backend == "auto"
            and _arg_numel_is_one(parsed.a_scale)
            and _arg_numel_is_one(parsed.b_scale)
        )
    return False


def _find_match_node(match: pm.Match, op) -> fx.Node | None:
    for node in match.nodes:
        if is_func(node, op):
            return node
    return None


def _get_match_bmm_fp8(match: pm.Match) -> _BmmFp8Op | None:
    node = _find_match_node(match, torch.ops.vllm.bmm_fp8.default)
    if node is None:
        return None
    return _parse_bmm_fp8(node)


def _is_qkv_split(node: fx.Node) -> bool:
    if not is_func(node, torch.ops.aten.split_with_sizes.default):
        return False

    split_sizes = _get_node_arg(node, "split_sizes", 1)
    dim = _get_node_arg(node, "dim", 2)
    return (
        isinstance(split_sizes, (list, tuple))
        and len(split_sizes) == 3
        and dim in (-1, 1)
    )


def _classify_qkv_branch(node: fx.Node) -> str | None:
    if any(_is_qkv_split(state.node) for state in _walk_user_paths(list(node.users))):
        return "direct"
    if any(
        state.saw_slice_scatter and _is_qkv_split(state.node)
        for state in _walk_user_paths(list(node.users), track_slice_scatter=True)
    ):
        return "rotary"
    return None


class _FlashInferCollectivePatternBase:
    def __init__(self, dtype: torch.dtype, device: str | None) -> None:
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def empty_fp8(self, *shape: int) -> torch.Tensor:
        return torch.empty(*shape, dtype=FLASHINFER_BMM_FP8_DTYPE, device=self.device)

    def empty_fp32_scalar(self) -> torch.Tensor:
        return torch.empty([], dtype=torch.float32, device=self.device)


class FlashInferBMMFP8ReduceScatterPattern(_FlashInferCollectivePatternBase):
    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.empty_fp8(128, 16),
            self.empty_fp8(16, 16),
            self.empty_fp32_scalar(),
            self.empty_fp32_scalar(),
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            a_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            a = torch.ops.aten.unsqueeze.default(a_2d, 0)
            b = torch.ops.aten.unsqueeze.default(b_2d, 0)
            bmm = torch.ops.vllm.bmm_fp8.default(
                a, b, a_scale, b_scale, self.dtype, "auto"
            )
            output = torch.ops.aten.view.default(bmm, [a_2d.shape[0], b_2d.shape[1]])
            return torch.ops.vllm.reduce_scatter.default(
                output, 0, self.tp_size, self.tp.unique_name
            )

        def replacement(
            a_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            output_shape = [a_2d.shape[0], b_2d.shape[1]]
            return torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter.default(
                a_2d,
                b_2d,
                a_scale,
                b_scale,
                "sum",
                0,
                0,
                self.tp.unique_name,
                output_shape,
                self.dtype,
            )

        def extra_check(match: pm.Match) -> bool:
            parsed = _get_match_bmm_fp8(match)
            if parsed is None:
                return False
            return _flashinfer_bmm_fp8_extra_check(match) and (
                _passes_min_m_after_reduce_scatter(parsed.a_2d, self.tp_size)
            )

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=extra_check,
        )


class FlashInferAllGatherBMMFP8Pattern(_FlashInferCollectivePatternBase):
    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.empty_fp8(128, 16),
            self.empty_fp8(16, 16),
            self.empty_fp32_scalar(),
            self.empty_fp32_scalar(),
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            gathered = torch.ops.vllm.all_gather.default(
                a_shard_2d, 0, self.tp_size, self.tp.unique_name
            )
            a = torch.ops.aten.unsqueeze.default(gathered, 0)
            b = torch.ops.aten.unsqueeze.default(b_2d, 0)
            bmm = torch.ops.vllm.bmm_fp8.default(
                a, b, a_scale, b_scale, self.dtype, "auto"
            )
            return torch.ops.aten.view.default(bmm, [gathered.shape[0], b_2d.shape[1]])

        def replacement(
            a_shard_2d: torch.Tensor,
            b_2d: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default(
                a_shard_2d,
                b_2d,
                a_scale,
                b_scale,
                0,
                self.tp.unique_name,
                self.dtype,
            )

        def extra_check(match: pm.Match) -> bool:
            parsed = _get_match_bmm_fp8(match)
            if parsed is None:
                return False
            return _flashinfer_bmm_fp8_extra_check(match) and _passes_min_m(parsed.a_2d)

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=extra_check,
        )


class FlashInferAllGatherBMMFP8QKVPattern(_FlashInferCollectivePatternBase):
    def register(self, pm_pass: PatternMatcherPass) -> None:
        pattern = pm.CallFunction(
            torch.ops.aten.split_with_sizes.default,
            pm.CallFunction(
                torch.ops.aten.view.default,
                pm.CallFunction(
                    torch.ops.vllm.bmm_fp8.default,
                    pm.CallFunction(
                        torch.ops.aten.unsqueeze.default,
                        pm.CallFunction(
                            torch.ops.vllm.all_gather.default,
                            pm.KeywordArg("a_shard_2d"),
                            0,
                            self.tp_size,
                            self.tp.unique_name,
                        ),
                        0,
                    ),
                    pm.CallFunction(
                        torch.ops.aten.unsqueeze.default,
                        pm.KeywordArg("b_2d"),
                        0,
                    ),
                    pm.KeywordArg("a_scale"),
                    pm.KeywordArg("b_scale"),
                    self.dtype,
                    "auto",
                    _users=pm.MULTIPLE,
                ),
                pm.Ignored(),
                _users=pm.MULTIPLE,
            ),
            pm.Ignored(),
            pm.Ignored(),
            _users=pm.MULTIPLE,
        )

        def extra_check(match: pm.Match) -> bool:
            if not _flashinfer_bmm_fp8_extra_check(match):
                return False

            split_node = _find_match_node(
                match, torch.ops.aten.split_with_sizes.default
            )
            if split_node is None:
                return False

            ag_match = _match_ag_bmm_from_split(split_node)
            return ag_match is not None and _passes_min_m(ag_match.replace_nodes[0])

        @pm.register_graph_pattern(
            pattern,
            pass_dict=pm_pass,
            extra_check=extra_check,
        )
        def handler(
            match: pm.Match,
            a_shard_2d: fx.Node,
            b_2d: fx.Node,
            a_scale: object,
            b_scale: object,
        ) -> None:
            del a_shard_2d, b_2d, a_scale, b_scale
            split_node = _find_match_node(
                match, torch.ops.aten.split_with_sizes.default
            )
            if split_node is None:
                return

            ag_match = _match_ag_bmm_from_split(split_node)
            if ag_match is None or not _passes_min_m(ag_match.replace_nodes[0]):
                return

            _lower_ag_bmm(match.graph, ag_match)
            match.erase_nodes()


def register_flashinfer_bmm_fp8_collective_patterns(
    pm_pass: PatternMatcherPass,
    dtype: torch.dtype,
    device: str | None,
) -> None:
    FlashInferBMMFP8ReduceScatterPattern(dtype, device).register(pm_pass)
    FlashInferAllGatherBMMFP8Pattern(dtype, device).register(pm_pass)
    FlashInferAllGatherBMMFP8QKVPattern(dtype, device).register(pm_pass)


@dataclass
class _FP8CollectiveGemmMatch:
    replace_nodes: list[fx.Node]
    a_2d: fx.Node
    b_2d: fx.Node
    a_scale: object
    b_scale: object
    out_dtype: object
    group_name: str


def _match_ag_bmm_from_split(
    split_node: fx.Node,
) -> _FP8CollectiveGemmMatch | None:
    if not _is_qkv_split(split_node):
        return None

    split_input = _get_node_arg(split_node, "self", 0)
    if (
        not isinstance(split_input, fx.Node)
        or not _is_passthrough(split_input)
        or _node_ndim(split_input) != 2
    ):
        return None

    bmm_node = _strip_view_like(split_input)
    parsed_bmm = _parse_bmm_fp8(bmm_node)
    if parsed_bmm is None:
        return None

    ag_node = _strip_view_like(parsed_bmm.a_2d)
    parsed_ag = _parse_all_gather(ag_node)
    if parsed_ag is None:
        return None

    parsed_group_name = _parse_collective_group_name(parsed_ag)
    if parsed_group_name is None:
        return None

    replace_nodes = [split_input]
    sibling_targets = [
        user
        for user in bmm_node.users
        if user is not split_input
        and _is_passthrough(user)
        and _node_ndim(user) == 2
        and _node_shape(user) == _node_shape(split_input)
    ]
    if sibling_targets:
        if len(sibling_targets) != 1:
            return None

        branch_kinds = {
            _classify_qkv_branch(split_input),
            _classify_qkv_branch(sibling_targets[0]),
        }
        if branch_kinds != {"direct", "rotary"}:
            return None
        replace_nodes.append(sibling_targets[0])

    return _FP8CollectiveGemmMatch(
        replace_nodes=replace_nodes,
        a_2d=parsed_ag.input_node,
        b_2d=parsed_bmm.b_2d,
        a_scale=parsed_bmm.a_scale,
        b_scale=parsed_bmm.b_scale,
        out_dtype=parsed_bmm.out_dtype,
        group_name=parsed_group_name,
    )


def _first_node_in_graph(graph: fx.Graph, nodes: list[fx.Node]) -> fx.Node:
    node_order = {node: index for index, node in enumerate(graph.nodes)}
    return min(nodes, key=node_order.__getitem__)


def _lower_ag_bmm(graph: fx.Graph, match: _FP8CollectiveGemmMatch) -> None:
    replace_node = _first_node_in_graph(graph, match.replace_nodes)
    with graph.inserting_before(replace_node):
        replacement = graph.call_function(
            torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default,
            args=(
                match.a_2d,
                match.b_2d,
                match.a_scale,
                match.b_scale,
                0,
                match.group_name,
                match.out_dtype,
            ),
        )

    _copy_replacement_meta(replace_node, replacement)
    for node in match.replace_nodes:
        node.replace_all_uses_with(replacement)
    for node in match.replace_nodes:
        graph.erase_node(node)
