# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ..fx_utils import is_func

FLASHINFER_BMM_FP8_MIN_M = 64
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


def _walk_reachable_users(start_nodes: list[fx.Node]) -> list[fx.Node]:
    worklist = list(start_nodes)
    visited: set[fx.Node] = set()
    reachable: list[fx.Node] = []

    while worklist:
        user = worklist.pop()
        if user in visited:
            continue
        visited.add(user)
        reachable.append(user)

        if _is_passthrough(user):
            worklist.extend(user.users)

    return reachable


def _walk_reachable_users_with_slice_scatter_state(
    start_nodes: list[fx.Node],
) -> list[tuple[fx.Node, bool]]:
    worklist: list[tuple[fx.Node, bool]] = [(user, False) for user in start_nodes]
    visited: set[tuple[fx.Node, bool]] = set()
    reachable: list[tuple[fx.Node, bool]] = []

    while worklist:
        user, saw_slice_scatter = worklist.pop()
        state = (user, saw_slice_scatter)
        if state in visited:
            continue
        visited.add(state)
        reachable.append(state)

        if _is_passthrough(user):
            worklist.extend((child, saw_slice_scatter) for child in user.users)
            continue

        if is_func(user, torch.ops.aten.slice_scatter.default):
            worklist.extend((child, True) for child in user.users)

    return reachable


def _collect_first_passthrough_matches(
    start_nodes: list[fx.Node],
    predicate,
) -> list[fx.Node]:
    worklist = list(start_nodes)
    visited: set[fx.Node] = set()
    matches: list[fx.Node] = []

    while worklist:
        user = worklist.pop()
        if user in visited:
            continue
        visited.add(user)

        if not _is_passthrough(user):
            continue

        if predicate(user):
            matches.append(user)
            continue

        worklist.extend(user.users)

    return matches


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


def _copy_replacement_meta(src: fx.Node, dst: fx.Node) -> None:
    dst.meta = {
        key: value for key, value in src.meta.items() if key != "eager_input_vals"
    }


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


def _parse_reduce_scatter(
    node: fx.Node,
) -> _CollectiveOp | None:
    return _parse_collective_op(node, torch.ops.vllm.reduce_scatter.default)


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


@dataclass
class _FP8CollectiveGemmMatch:
    replace_nodes: list[fx.Node]
    a_2d: fx.Node
    b_2d: fx.Node
    a_scale: object
    b_scale: object
    out_dtype: object
    group_name: str


@dataclass
class _MatchedCollectiveUser:
    node: fx.Node
    collective: _CollectiveOp


class _FlashInferCollectiveFusionRewriter:
    def __init__(self, graph: fx.Graph) -> None:
        self.graph = graph
        self._node_order = {node: index for index, node in enumerate(graph.nodes)}

    def run(self) -> int:
        replaced = 0
        for node in list(self.graph.nodes):
            if not is_func(node, torch.ops.vllm.bmm_fp8.default):
                continue

            rs_match = self._match_bmm_rs(node)
            if rs_match is not None and _passes_min_m(rs_match.replace_nodes[0]):
                self._lower_bmm_rs(rs_match)
                replaced += 1
                continue

            ag_match = self._match_ag_bmm(node)
            if ag_match is None or not _passes_min_m(ag_match.replace_nodes[0]):
                continue

            self._lower_ag_bmm(ag_match)
            replaced += 1

        if replaced:
            self.graph.eliminate_dead_code()
            self.graph.lint()
        return replaced

    def _find_reduce_scatter_user(
        self,
        bmm_node: fx.Node,
    ) -> _MatchedCollectiveUser | None:
        rs_matches: list[_MatchedCollectiveUser] = []

        for user in _walk_reachable_users(list(bmm_node.users)):
            parsed_rs = _parse_reduce_scatter(user)
            if parsed_rs is not None:
                rs_matches.append(
                    _MatchedCollectiveUser(node=user, collective=parsed_rs)
                )

        if len(rs_matches) == 1:
            return rs_matches[0]
        return None

    def _is_qkv_split(self, node: fx.Node) -> bool:
        if not is_func(node, torch.ops.aten.split_with_sizes.default):
            return False

        split_sizes = _get_node_arg(node, "split_sizes", 1)
        dim = _get_node_arg(node, "dim", 2)
        return (
            isinstance(split_sizes, (list, tuple))
            and len(split_sizes) == 3
            and dim in (-1, 1)
        )

    def _classify_qkv_branch(self, node: fx.Node) -> str | None:
        if any(
            self._is_qkv_split(user) for user in _walk_reachable_users(list(node.users))
        ):
            return "direct"
        if any(
            saw_slice_scatter and self._is_qkv_split(user)
            for user, saw_slice_scatter in (
                _walk_reachable_users_with_slice_scatter_state(list(node.users))
            )
        ):
            return "rotary"
        return None

    def _find_ag_qkv_replace_targets(self, bmm_node: fx.Node) -> list[fx.Node] | None:
        replace_targets = [
            user
            for user in bmm_node.users
            if _is_passthrough(user) and _node_ndim(user) == 2
        ]
        if len(replace_targets) != 2:
            return None

        if _node_shape(replace_targets[0]) != _node_shape(replace_targets[1]):
            return None

        branch_kinds = [self._classify_qkv_branch(node) for node in replace_targets]
        if set(branch_kinds) == {"direct", "rotary"}:
            return replace_targets
        return None

    def _find_ag_single_replace_target(self, bmm_node: fx.Node) -> fx.Node | None:
        replace_targets = _collect_first_passthrough_matches(
            list(bmm_node.users),
            lambda node: _node_ndim(node) == 2,
        )

        if not replace_targets and _node_ndim(bmm_node) == 2:
            replace_targets = [bmm_node]

        if len(replace_targets) != 1:
            return None
        return replace_targets[0]

    def _find_ag_replace_targets(self, bmm_node: fx.Node) -> list[fx.Node] | None:
        qkv_targets = self._find_ag_qkv_replace_targets(bmm_node)
        if qkv_targets is not None:
            return qkv_targets

        target = self._find_ag_single_replace_target(bmm_node)
        if target is None:
            return None
        return [target]

    def _first_node_in_graph(self, nodes: list[fx.Node]) -> fx.Node:
        return min(nodes, key=self._node_order.__getitem__)

    def _match_bmm_rs(self, bmm_node: fx.Node) -> _FP8CollectiveGemmMatch | None:
        parsed_bmm = _parse_bmm_fp8(bmm_node)
        if parsed_bmm is None:
            return None

        rs_match = self._find_reduce_scatter_user(bmm_node)
        if rs_match is None:
            return None

        parsed_group_name = _parse_collective_group_name(rs_match.collective)
        if parsed_group_name is None:
            return None

        return _FP8CollectiveGemmMatch(
            replace_nodes=[rs_match.node],
            a_2d=parsed_bmm.a_2d,
            b_2d=parsed_bmm.b_2d,
            a_scale=parsed_bmm.a_scale,
            b_scale=parsed_bmm.b_scale,
            out_dtype=parsed_bmm.out_dtype,
            group_name=parsed_group_name,
        )

    def _match_ag_bmm(self, bmm_node: fx.Node) -> _FP8CollectiveGemmMatch | None:
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

        targets = self._find_ag_replace_targets(bmm_node)
        if targets is None:
            return None

        return _FP8CollectiveGemmMatch(
            replace_nodes=targets,
            a_2d=parsed_ag.input_node,
            b_2d=parsed_bmm.b_2d,
            a_scale=parsed_bmm.a_scale,
            b_scale=parsed_bmm.b_scale,
            out_dtype=parsed_bmm.out_dtype,
            group_name=parsed_group_name,
        )

    def _lower_bmm_rs(self, match: _FP8CollectiveGemmMatch) -> None:
        replace_node = match.replace_nodes[0]
        a_shape = _node_shape(match.a_2d)
        b_shape = _node_shape(match.b_2d)
        if a_shape is None or b_shape is None:
            return
        output_shape = [a_shape[0], b_shape[1]]

        with self.graph.inserting_before(replace_node):
            replacement = self.graph.call_function(
                torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter.default,
                args=(
                    match.a_2d,
                    match.b_2d,
                    match.a_scale,
                    match.b_scale,
                    "sum",
                    0,
                    0,
                    match.group_name,
                    output_shape,
                    match.out_dtype,
                ),
            )

        _copy_replacement_meta(replace_node, replacement)
        replace_node.replace_all_uses_with(replacement)
        self.graph.erase_node(replace_node)

    def _lower_ag_bmm(self, match: _FP8CollectiveGemmMatch) -> None:
        replace_node = self._first_node_in_graph(match.replace_nodes)
        with self.graph.inserting_before(replace_node):
            replacement = self.graph.call_function(
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
            self.graph.erase_node(node)


def rewrite_flashinfer_bmm_fp8_collective_fusion(graph: fx.Graph) -> int:
    return _FlashInferCollectiveFusionRewriter(graph).run()
