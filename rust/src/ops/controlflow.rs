//! Control-flow op handlers: If, Scan, Loop. Faithful port of the C++ `ops/controlflow.cc`.
//!
//! Unlike ordinary ops these carry their computation as a nested subgraph (GraphProto) ATTRIBUTE — a
//! body ORT surfaces to a plugin EP via `Node_GetSubgraphs` (`ep::build_subgraphs` captures each body
//! as a `NodeDesc::subgraphs` entry). The MLX EP owns the control-flow node WHOLE (its body is
//! declined for independent offload in `ep::get_capability`) and realizes the control flow by
//! translating the body inline through `TranslationContext::run_subgraph`:
//!
//!   * If   — read the runtime `cond` host-side each forward and translate the taken branch only.
//!   * Scan — STATIC trip count (scan axis length known from the input shape). Unroll the body over
//!            axis 0, carrying state and stacking scan outputs. Forward, axis 0 only (MVP).
//!   * Loop — CONSTANT trip count M with a cond that is a pass-through of the loop cond input (the
//!            canonical `for i in range(M)` idiom). Unroll M times; carried-state-only (MVP).
//!
//! Anything outside these static/foldable forms is left unclaimed and runs on ORT's CPU control-flow
//! kernels (with the body ops still offloaded to MLX via the ordinary flat path).

use crate::engine::{MlxError, NodeDesc, SubgraphDesc, TensorRef, TranslationContext};
use crate::registry::{
    ClaimPredicate, GraphView, NodeView, OpHandler, OpRegistration, OpRegistry, K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::sys::ort;

// ---- shared helpers -----------------------------------------------------------------------------

/// Find a body subgraph by attribute name.
fn find_body<'a>(n: &'a NodeDesc, attr: &str) -> Option<&'a SubgraphDesc> {
    n.subgraphs.iter().find(|sg| sg.attr_name == attr)
}

/// Read a scalar bool from a foldable (initializer / ctx) node input.
fn read_host_bool(ctx: &TranslationContext, r: &TensorRef) -> Result<bool, MlxError> {
    let h = ctx.raw_host(r)?;
    if h.data.is_null() {
        return Ok(false);
    }
    Ok(unsafe { *(h.data as *const u8) } != 0)
}

/// Every node in a control-flow body must be MLX-translatable (recursively via the registry claim).
fn body_claimable(body: &GraphView) -> bool {
    body.all_nodes_claimable()
}

fn is_bool(node: &NodeView, i: usize) -> bool {
    matches!(node.input_info(i), Some(info)
        if info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)
}

fn is_int64(node: &NodeView, i: usize) -> bool {
    matches!(node.input_info(i), Some(info)
        if info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
}

/// Reject non-default (non-forward / non-axis-0) direction/axis attributes.
fn all_zero_ints_attr(node: &NodeView, name: &str) -> bool {
    let (present, v) = node.ints_attr(name);
    if !present {
        return true;
    }
    v.iter().all(|&x| x == 0)
}

// ---- If -----------------------------------------------------------------------------------------

fn if_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let cond = read_host_bool(ctx, &n.inputs[0])?;
    let attr = if cond { "then_branch" } else { "else_branch" };
    let branch = find_body(n, attr)
        .ok_or_else(|| "MLX If: missing branch subgraph".to_string())?
        .clone();
    let outs = ctx.run_subgraph(&branch, &[])?;
    if outs.len() != n.outputs.len() {
        return Err("MLX If: branch output arity mismatch".to_string());
    }
    for (i, o) in n.outputs.iter().enumerate() {
        ctx.bind(o, outs[i]);
    }
    Ok(())
}

fn if_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() == 0 {
        return false;
    }
    if !is_bool(node, 0) {
        return false;
    }
    let subs = node.subgraphs();
    if subs.len() != 2 {
        return false;
    }
    let (mut have_then, mut have_else) = (false, false);
    for (name, body) in &subs {
        match name.as_str() {
            "then_branch" => have_then = true,
            "else_branch" => have_else = true,
            _ => return false,
        }
        if !body.input_names().is_empty() {
            return false; // If branches take no formal inputs
        }
        if body.output_names().len() != node.num_outputs() {
            return false;
        }
        if !body_claimable(body) {
            return false;
        }
    }
    have_then && have_else
}

// ---- Scan ---------------------------------------------------------------------------------------

fn scan_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let num_scan = *n
        .ints
        .get("num_scan_inputs")
        .ok_or_else(|| "MLX Scan: missing num_scan_inputs".to_string())? as usize;
    let num_state = n.inputs.len() - num_scan;
    let body = find_body(n, "body")
        .ok_or_else(|| "MLX Scan: missing body subgraph".to_string())?
        .clone();

    let mut state: Vec<mlx::mlx_array> = Vec::with_capacity(num_state);
    for i in 0..num_state {
        state.push(ctx.resolve(&n.inputs[i])?);
    }
    let mut scans: Vec<mlx::mlx_array> = Vec::with_capacity(num_scan);
    for i in 0..num_scan {
        scans.push(ctx.resolve(&n.inputs[num_state + i])?);
    }

    let s0 = ctx.shape_of(scans[0]);
    if s0.is_empty() {
        return Err("MLX Scan: scan input is a scalar".to_string());
    }
    let trip = s0[0];

    let num_scan_out = body.output_names.len() as i64 - num_state as i64;
    if num_scan_out < 0 {
        return Err("MLX Scan: body output arity".to_string());
    }
    let num_scan_out = num_scan_out as usize;
    let mut collected: Vec<Vec<mlx::mlx_array>> = vec![Vec::new(); num_scan_out];

    for t in 0..trip {
        let mut bin: Vec<mlx::mlx_array> = Vec::with_capacity(num_state + num_scan);
        for i in 0..num_state {
            bin.push(state[i]);
        }
        for i in 0..num_scan {
            let shp = ctx.shape_of(scans[i]);
            let mut start = vec![0i32; shp.len()];
            let mut stop = shp.clone();
            start[0] = t;
            stop[0] = t + 1;
            let sl = ctx.slice(scans[i], &start, &stop)?;
            let sq = ctx.squeeze(sl, 0)?;
            bin.push(sq);
        }
        let bout = ctx.run_subgraph(&body, &bin)?;
        for i in 0..num_state {
            state[i] = bout[i];
        }
        for i in 0..num_scan_out {
            collected[i].push(bout[num_state + i]);
        }
    }

    for i in 0..num_state {
        ctx.bind(&n.outputs[i], state[i]);
    }
    for i in 0..num_scan_out {
        let stacked = ctx.stack(&collected[i], 0)?;
        ctx.bind(&n.outputs[num_state + i], stacked);
    }
    Ok(())
}

fn scan_claim(node: &NodeView) -> bool {
    let ninputs = node.num_inputs();
    let noutputs = node.num_outputs();
    let num_scan = node.int_attr("num_scan_inputs", -1);
    if num_scan <= 0 || (ninputs as i64) < num_scan {
        return false;
    }
    let num_state = ninputs as i64 - num_scan;
    if num_state < 0 {
        return false;
    }

    if !all_zero_ints_attr(node, "scan_input_directions")
        || !all_zero_ints_attr(node, "scan_output_directions")
        || !all_zero_ints_attr(node, "scan_input_axes")
        || !all_zero_ints_attr(node, "scan_output_axes")
    {
        return false;
    }

    // Every scan input must have a statically-known, non-empty scan axis (axis 0) for a static unroll.
    for i in num_state..ninputs as i64 {
        match node.input_info(i as usize) {
            Some(info) => {
                if info.shape.is_empty() || info.shape[0] < 1 {
                    return false;
                }
            }
            None => return false,
        }
    }

    let subs = node.subgraphs();
    if subs.len() != 1 || subs[0].0 != "body" {
        return false;
    }
    let body = &subs[0].1;
    if body.input_names().len() as i64 != num_state + num_scan {
        return false;
    }
    if (body.output_names().len() as i64) < num_state {
        return false;
    }
    if noutputs as i64 != body.output_names().len() as i64 {
        return false;
    }
    body_claimable(body)
}

// ---- Loop ---------------------------------------------------------------------------------------

/// True iff the body's cond output (body output 0) is a pass-through of the body's cond input (body
/// input 1): either a direct graph-output alias, or an Identity node copying it.
fn loop_cond_is_passthrough(body: &GraphView) -> bool {
    let bin = body.input_names();
    let bout = body.output_names();
    if bin.len() < 2 || bout.is_empty() {
        return false;
    }
    let cond_in = &bin[1];
    let cond_out = &bout[0];
    if cond_in.is_empty() || cond_out.is_empty() {
        return false;
    }
    if cond_in == cond_out {
        return true;
    }
    for node in body.nodes() {
        if node.op_type() != "Identity" {
            continue;
        }
        let ins = node.input_names();
        let outs = node.output_names();
        if ins.len() == 1 && outs.len() == 1 && &ins[0] == cond_in && &outs[0] == cond_out {
            return true;
        }
    }
    false
}

fn loop_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let body = find_body(n, "body")
        .ok_or_else(|| "MLX Loop: missing body subgraph".to_string())?
        .clone();
    let num_state = n.inputs.len() - 2; // inputs = [M, cond, state...]

    let trip_count: i64 = {
        let h = ctx.raw_host(&n.inputs[0])?;
        if h.data.is_null() {
            return Err("MLX Loop: null trip count".to_string());
        }
        unsafe { *(h.data as *const i64) }
    };
    let cond0 = read_host_bool(ctx, &n.inputs[1])?;
    let trip = if cond0 { trip_count } else { 0 };

    let mut state: Vec<mlx::mlx_array> = Vec::with_capacity(num_state);
    for i in 0..num_state {
        state.push(ctx.resolve(&n.inputs[2 + i])?);
    }

    for t in 0..trip {
        let iter = ctx.scalar_i64(t);
        let condin = ctx.scalar_bool(true);
        let mut bin: Vec<mlx::mlx_array> = Vec::with_capacity(2 + num_state);
        bin.push(iter);
        bin.push(condin);
        for i in 0..num_state {
            bin.push(state[i]);
        }
        let bout = ctx.run_subgraph(&body, &bin)?;
        // bout[0] = cond_out (pass-through, guaranteed true by claim); bout[1..] = carried state.
        for i in 0..num_state {
            state[i] = bout[1 + i];
        }
    }

    for i in 0..num_state {
        ctx.bind(&n.outputs[i], state[i]);
    }
    Ok(())
}

fn loop_claim(node: &NodeView) -> bool {
    let ninputs = node.num_inputs();
    if ninputs < 2 {
        return false; // require explicit M and cond (MVP)
    }
    let num_state = ninputs - 2;
    // M (int64 scalar) and cond (bool) must be present so we can read them host-side.
    if !is_int64(node, 0) || !is_bool(node, 1) {
        return false;
    }

    let subs = node.subgraphs();
    if subs.len() != 1 || subs[0].0 != "body" {
        return false;
    }
    let body = &subs[0].1;
    // body inputs = [iter_num, cond_in, state...]; carried-state-only (no scan outputs) in this MVP,
    // so body outputs = [cond_out, state...] and node outputs = [state...].
    if body.input_names().len() != 2 + num_state {
        return false;
    }
    if body.output_names().len() != 1 + num_state {
        return false;
    }
    if node.num_outputs() != num_state {
        return false;
    }
    if !loop_cond_is_passthrough(body) {
        return false;
    }
    body_claimable(body)
}

// ---- registration -------------------------------------------------------------------------------

fn reg(registry: &mut OpRegistry, op_type: &'static str, handler: OpHandler, claim: ClaimPredicate) {
    registry.register(OpRegistration {
        domain: "",
        op_type,
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

pub fn register(registry: &mut OpRegistry) {
    reg(registry, "If", if_op, if_claim);
    reg(registry, "Scan", scan_op, scan_claim);
    reg(registry, "Loop", loop_op, loop_claim);
}
