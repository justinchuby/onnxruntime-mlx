"""Op-correctness tests for the recurrent family (rust/src/ops/recurrent.rs) vs ORT CPU.

RNN / GRU / LSTM are unrolled over a STATICALLY-KNOWN sequence length into a fixed MLX graph. Each
claimed form (forward / reverse, with and without bias / initial state / peepholes) is parametrised
against ORT's CPU EP. Forms the handler leaves to ORT CPU are recorded as explicit skips:

  * dynamic (symbolic) seq_length  — the unroll needs a constant S;
  * a `sequence_lens` input        — variable-length masking;
  * non-default `activations`      — only RNN=Tanh / GRU=Sigmoid,Tanh / LSTM=Sigmoid,Tanh,Tanh
                                     are translatable (the STRINGS attr is not carried to the graph).
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

from _models import DataType, assert_matches_cpu, tensor

# Gate multiplier packed into W/R per op.
_GATES = {"RNN": 1, "GRU": 3, "LSTM": 4}


def _recurrent_model(
    op: str,
    *,
    seq: int,
    batch: int,
    input_size: int,
    hidden: int,
    direction: str,
    bias: bool,
    initial: bool,
    peephole: bool = False,
    linear_before_reset: int | None = None,
    input_forget: int | None = None,
    clip: float | None = None,
    outputs: tuple[str, ...] = ("Y", "Y_h", "Y_c"),
) -> tuple[bytes, dict[str, np.ndarray]]:
    """Build a single-node RNN/GRU/LSTM model plus random feeds."""
    num_dir = 2 if direction == "bidirectional" else 1
    gates = _GATES[op]
    rng = np.random.default_rng(hash((op, seq, direction, bias, initial, peephole)) & 0xFFFFFFFF)

    def rand(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    x = rand(seq, batch, input_size)
    w = rand(num_dir, gates * hidden, input_size)
    r = rand(num_dir, gates * hidden, hidden)

    in_values: list[ir.Value | None] = [
        tensor("X", DataType.FLOAT, [seq, batch, input_size]),
        tensor("W", DataType.FLOAT, [num_dir, gates * hidden, input_size]),
        tensor("R", DataType.FLOAT, [num_dir, gates * hidden, hidden]),
    ]
    feeds: dict[str, np.ndarray] = {"X": x, "W": w, "R": r}

    def add_optional(index: int, name: str, value: np.ndarray | None) -> None:
        while len(in_values) < index:
            in_values.append(None)
        if value is None:
            in_values.append(None)
            return
        in_values.append(tensor(name, DataType.FLOAT, list(value.shape)))
        feeds[name] = value

    add_optional(3, "B", rand(num_dir, 2 * gates * hidden) if bias else None)
    add_optional(4, "sequence_lens", None)  # never fed (claimed form has no sequence_lens)
    add_optional(5, "initial_h", rand(num_dir, batch, hidden) if initial else None)
    if op == "LSTM":
        add_optional(6, "initial_c", rand(num_dir, batch, hidden) if initial else None)
        add_optional(7, "P", rand(num_dir, 3 * hidden) if peephole else None)

    out_values: list[ir.Value] = []
    out_specs = {
        "Y": [seq, num_dir, batch, hidden],
        "Y_h": [num_dir, batch, hidden],
        "Y_c": [num_dir, batch, hidden],
    }
    for name in outputs:
        if name == "Y_c" and op != "LSTM":
            continue
        out_values.append(tensor(name, DataType.FLOAT, out_specs[name]))

    attrs: list[ir.Attr] = [
        ir.AttrInt64("hidden_size", hidden),
        ir.AttrString("direction", direction),
    ]
    if linear_before_reset is not None:
        attrs.append(ir.AttrInt64("linear_before_reset", linear_before_reset))
    if input_forget is not None:
        attrs.append(ir.AttrInt64("input_forget", input_forget))
    if clip is not None:
        attrs.append(ir.AttrFloat32("clip", clip))

    node = ir.node(
        op,
        in_values,
        attributes={attr.name: attr for attr in attrs},
        outputs=out_values,
    )
    graph = ir.Graph(
        [v for v in in_values if v is not None],
        out_values,
        nodes=[node],
        name=f"mlx_{op}",
        opset_imports={"": 17},
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString(), feeds


_DIRECTIONS = ["forward", "reverse"]
_FLAGS = [
    pytest.param(False, False, id="nobias-noinit"),
    pytest.param(True, False, id="bias-noinit"),
    pytest.param(True, True, id="bias-init"),
]


# --- RNN -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", _DIRECTIONS)
@pytest.mark.parametrize("bias,initial", _FLAGS)
def test_rnn(direction: str, bias: bool, initial: bool) -> None:
    model, feeds = _recurrent_model(
        "RNN",
        seq=5,
        batch=3,
        input_size=4,
        hidden=6,
        direction=direction,
        bias=bias,
        initial=initial,
        outputs=("Y", "Y_h"),
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


# --- GRU -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", _DIRECTIONS)
@pytest.mark.parametrize("bias,initial", _FLAGS)
@pytest.mark.parametrize("linear_before_reset", [0, 1])
def test_gru(direction: str, bias: bool, initial: bool, linear_before_reset: int) -> None:
    model, feeds = _recurrent_model(
        "GRU",
        seq=5,
        batch=3,
        input_size=4,
        hidden=6,
        direction=direction,
        bias=bias,
        initial=initial,
        linear_before_reset=linear_before_reset,
        outputs=("Y", "Y_h"),
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


# --- LSTM ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("direction", _DIRECTIONS)
@pytest.mark.parametrize("bias,initial", _FLAGS)
def test_lstm(direction: str, bias: bool, initial: bool) -> None:
    model, feeds = _recurrent_model(
        "LSTM",
        seq=5,
        batch=3,
        input_size=4,
        hidden=6,
        direction=direction,
        bias=bias,
        initial=initial,
        outputs=("Y", "Y_h", "Y_c"),
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("direction", _DIRECTIONS)
def test_lstm_peephole(direction: str) -> None:
    model, feeds = _recurrent_model(
        "LSTM",
        seq=4,
        batch=2,
        input_size=3,
        hidden=5,
        direction=direction,
        bias=True,
        initial=True,
        peephole=True,
        outputs=("Y", "Y_h", "Y_c"),
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


def test_lstm_input_forget() -> None:
    model, feeds = _recurrent_model(
        "LSTM",
        seq=4,
        batch=2,
        input_size=3,
        hidden=5,
        direction="forward",
        bias=True,
        initial=False,
        input_forget=1,
        outputs=("Y", "Y_h", "Y_c"),
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
def test_clip(op: str) -> None:
    model, feeds = _recurrent_model(
        op,
        seq=4,
        batch=2,
        input_size=3,
        hidden=5,
        direction="forward",
        bias=True,
        initial=False,
        clip=1.5,
        outputs=("Y", "Y_h"),
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


# --- Bidirectional (two directions concatenated along the num_directions axis) --------------------
@pytest.mark.parametrize("op", ["RNN", "GRU", "LSTM"])
def test_bidirectional(op: str) -> None:
    outs = ("Y", "Y_h", "Y_c") if op == "LSTM" else ("Y", "Y_h")
    model, feeds = _recurrent_model(
        op,
        seq=5,
        batch=3,
        input_size=4,
        hidden=6,
        direction="bidirectional",
        bias=True,
        initial=True,
        outputs=outs,
    )
    assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


# --- Forms deliberately left to ORT CPU (documented, not exercised on MLX) ------------------------
@pytest.mark.skip(reason="dynamic (symbolic) seq_length is left to ORT CPU — the unroll needs a static S")
def test_dynamic_seq_left_to_cpu() -> None:
    ...


@pytest.mark.skip(reason="sequence_lens (variable-length masking) is left to ORT CPU")
def test_sequence_lens_left_to_cpu() -> None:
    ...


@pytest.mark.skip(reason="non-default activations (e.g. Relu RNN) are left to ORT CPU")
def test_non_default_activations_left_to_cpu() -> None:
    ...
