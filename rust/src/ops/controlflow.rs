//! Control-flow op handlers: If, Scan, Loop. Faithful port of the C++ `ops/controlflow.cc`.
//!
//! Unlike ordinary ops these carry their computation as a nested subgraph (GraphProto) ATTRIBUTE — a
//! body ORT surfaces to a plugin EP via `Node_GetSubgraphs` (`ep::build_subgraphs` captures each body
//! as a `NodeDesc::subgraphs` entry). The MLX EP owns the control-flow node WHOLE (its body is
//! declined for independent offload in `ep::get_capability`) and realizes the control flow by
//! translating the body inline through `TranslationContext::run_subgraph`:
//!
//!   * If — HOST-READABLE `cond` (graph input / initializer / outer-scope); read host-side and
//!     translate the taken branch only. A data-dependent `cond` produced by another node is declined
//!     (runs on ORT's CPU control-flow kernels).
//!   * Scan — SHAPE-SPECIALIZED trip count (the runtime scan-axis extent is fixed while tracing a
//!     shape-keyed closure). Unroll the body over each input's configured axis/direction, carrying
//!     state and stacking each output along its configured axis/direction.
//!   * Loop — CONSTANT trip count M with a cond that is a pass-through of the loop cond input (the
//!     canonical `for i in range(M)` idiom). Unroll M times; carried-state-only (MVP).
//!
//! Anything outside these static/foldable forms is left unclaimed and runs on ORT's CPU control-flow
//! kernels (with the body ops still offloaded to MLX via the ordinary flat path).

use crate::engine::{MlxError, NodeDesc, SubgraphDesc, TensorRef, TranslationContext};
use crate::ops::selective_scan;
use crate::registry::{
    ClaimPredicate, ClaimResult, GraphView, K_ANY_OPSET, NodeView, OpHandler, OpRegistration,
    OpRegistry,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

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

/// Every node in a control-flow body must be MLX-translatable (recursively via the registry claim),
/// and the body must be free of float64 (see `GraphView::body_uses_float64`).
fn body_claimable(body: &GraphView) -> bool {
    !body.body_uses_float64() && body.all_nodes_claimable()
}

/// Name the first body node the registry refuses, for a denial reason that can be acted on.
fn body_rejection(body: &GraphView) -> String {
    if body.body_uses_float64() {
        return "body carries float64, which MLX can only evaluate on a CPU stream — a \
                control-flow body is translated inside the parent's (GPU) plan, so it is left to \
                ORT CPU"
            .to_string();
    }
    match body.first_unclaimable_node() {
        Some((op_type, name, reason)) => {
            let label = if name.is_empty() {
                op_type
            } else {
                format!("{op_type} '{name}'")
            };
            format!("body contains an unclaimable operation: {label}: {reason}")
        }
        None => "body contains an unclaimable operation".to_string(),
    }
}

fn is_bool(node: &NodeView, i: usize) -> bool {
    matches!(node.input_info(i), Some(info)
        if info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)
}

fn is_int64(node: &NodeView, i: usize) -> bool {
    matches!(node.input_info(i), Some(info)
        if info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
}

fn normalized_axis(axis: i64, rank: usize) -> Option<usize> {
    let rank = rank as i64;
    let axis = if axis < 0 { axis + rank } else { axis };
    (0..rank).contains(&axis).then_some(axis as usize)
}

fn scan_attr(n: &NodeDesc, name: &str, len: usize) -> Result<Vec<i64>, MlxError> {
    match n.int_arrays.get(name) {
        Some(values) if values.len() == len => Ok(values.clone()),
        Some(values) => Err(format!(
            "MLX Scan: {name} has {} values, expected {len}",
            values.len()
        )),
        None => Ok(vec![0; len]),
    }
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

fn if_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() > 0,
        "expects 1 condition input and at least 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    require!(is_bool(node, 0), "condition input must have bool dtype");
    // The branch is selected host-side at translate time (`if_op` reads the cond via `raw_host`),
    // so the condition MUST be host-readable: a graph input, initializer, or outer-scope value.
    // A runtime intermediate (e.g. Phi-4's long-context `Greater(total_seq_len, 4096)` rotary-cache
    // selector) is not readable by `raw_host` — leave such If nodes to ORT's CPU control-flow
    // kernels (their body ops still offload to MLX via the ordinary flat path).
    require!(
        node.input_is_host_readable(0),
        "condition must be a graph input/initializer (data-dependent conditions run on CPU)"
    );
    let subs = node.subgraphs();
    require!(
        subs.len() == 2,
        "expects then_branch and else_branch subgraphs"
    );
    let (mut have_then, mut have_else) = (false, false);
    for (name, body) in &subs {
        match name.as_str() {
            "then_branch" => have_then = true,
            "else_branch" => have_else = true,
            _ => deny!("unsupported subgraph attribute {:?}", name),
        }
        require!(
            body.input_names().is_empty(),
            "{} must have no formal inputs",
            name
        );
        require!(
            body.output_names().len() == node.num_outputs(),
            "{} has {} outputs but the If node has {}",
            name,
            body.output_names().len(),
            node.num_outputs()
        );
        require!(
            body_claimable(body),
            "{} contains an unclaimable operation",
            name
        );
    }
    require!(
        have_then && have_else,
        "requires both then_branch and else_branch"
    );
    Ok(())
}

// ---- Scan ---------------------------------------------------------------------------------------

fn scan_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let num_scan =
        *n.ints
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

    let num_scan_out = body.output_names.len() as i64 - num_state as i64;
    if num_scan_out < 0 {
        return Err("MLX Scan: body output arity".to_string());
    }
    let num_scan_out = num_scan_out as usize;
    let input_axes = scan_attr(n, "scan_input_axes", num_scan)?;
    let input_directions = scan_attr(n, "scan_input_directions", num_scan)?;
    let output_axes = scan_attr(n, "scan_output_axes", num_scan_out)?;
    let output_directions = scan_attr(n, "scan_output_directions", num_scan_out)?;

    let scan_shapes: Vec<Vec<i32>> = scans.iter().map(|&scan| ctx.shape_of(scan)).collect();
    let scan_axes: Vec<usize> = input_axes
        .iter()
        .zip(&scan_shapes)
        .map(|(&axis, shape)| {
            normalized_axis(axis, shape.len()).ok_or_else(|| {
                format!(
                    "MLX Scan: input axis {axis} invalid for rank {}",
                    shape.len()
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let trip = scan_shapes[0][scan_axes[0]];
    if scan_shapes
        .iter()
        .zip(&scan_axes)
        .any(|(shape, &axis)| shape[axis] != trip)
    {
        return Err("MLX Scan: scan input sequence lengths differ".to_string());
    }

    if trip == 0 {
        for (o, &s) in n.outputs[..num_state].iter().zip(&state[..num_state]) {
            ctx.bind(o, s);
        }
        if num_scan_out == 0 {
            return Ok(());
        }
        // Infer each body output's element shape from one symbolic dummy iteration, then construct
        // the required zero-length accumulated output. The dummy graph is never bound or evaluated.
        let mut bin = state.clone();
        for (i, &scan) in scans.iter().enumerate() {
            let mut element_shape = scan_shapes[i].clone();
            element_shape.remove(scan_axes[i]);
            let dtype = ctx.dtype_of(scan);
            bin.push(ctx.zeros(&element_shape, dtype)?);
        }
        let templates = ctx.run_subgraph(&body, &bin)?;
        for i in 0..num_scan_out {
            let template = templates[num_state + i];
            let mut shape = ctx.shape_of(template);
            let axis = normalized_axis(output_axes[i], shape.len() + 1).ok_or_else(|| {
                format!(
                    "MLX Scan: output axis {} invalid for rank {}",
                    output_axes[i],
                    shape.len() + 1
                )
            })?;
            shape.insert(axis, 0);
            let dtype = ctx.dtype_of(template);
            let empty = ctx.zeros(&shape, dtype)?;
            ctx.bind(&n.outputs[num_state + i], empty);
        }
        return Ok(());
    }

    let mut collected: Vec<Vec<mlx::mlx_array>> = vec![Vec::new(); num_scan_out];

    // Fused Mamba-1 selective scan: replaces the whole unroll below with one custom Metal kernel
    // that keeps the running state in registers. Declines (falls through to the unroll) unless the
    // body matches exactly and the shapes/dtypes are covered. See `ops/selective_scan.rs`.
    //
    // Both outcomes are recorded on the node's trace path so a silent regression is detectable: if
    // the emitting graph changes shape and the pattern stops matching, the node shows up as
    // `op.composed` with a reason instead of quietly costing 7x more.
    let fusable_form = num_scan_out == 1
        && !selective_scan::disabled()
        && input_directions.iter().all(|&d| d == input_directions[0])
        && scan_axes.iter().all(|&ax| ax == 0)
        && output_axes[0] == 0;
    let decline_reason: &'static str = if !fusable_form {
        "Scan form (axes/directions/outputs) not fusable -> unrolled per timestep"
    } else {
        match selective_scan::match_body(&body, num_state, num_scan) {
            Some(m) => {
                let fused = selective_scan::emit(
                    ctx,
                    state[m.h_state],
                    state[m.a_state],
                    scans[m.dt],
                    scans[m.b],
                    scans[m.c],
                    scans[m.x],
                    input_directions[0] == 1,
                    output_directions[0] == 1,
                )?;
                match fused {
                    Some((h_final, y)) => {
                        let a_pass = state[m.a_state];
                        ctx.bind(&n.outputs[m.h_state], h_final);
                        ctx.bind(&n.outputs[m.a_state], a_pass);
                        ctx.bind(&n.outputs[m.out_y], y);
                        ctx.mark_fast(selective_scan::KERNEL_NAME);
                        return Ok(());
                    }
                    None => {
                        "selective-scan body matched but shape/dtype unsupported by the fused \
                         kernel -> unrolled"
                    }
                }
            }
            None => "Scan body is not a recognised Mamba-1 selective scan -> unrolled per timestep",
        }
    };

    for t in 0..trip {
        let mut bin: Vec<mlx::mlx_array> = Vec::with_capacity(num_state + num_scan);
        bin.extend_from_slice(&state[..num_state]);
        for (i, scan) in scans[..num_scan].iter().enumerate() {
            let shp = &scan_shapes[i];
            let axis = scan_axes[i];
            let index = if input_directions[i] == 0 {
                t
            } else {
                trip - 1 - t
            };
            let mut start = vec![0i32; shp.len()];
            let mut stop = shp.clone();
            start[axis] = index;
            stop[axis] = index + 1;
            let sl = ctx.slice(*scan, &start, &stop)?;
            let sq = ctx.squeeze(sl, axis as i32)?;
            bin.push(sq);
        }
        let bout = ctx.run_subgraph(&body, &bin)?;
        state[..num_state].copy_from_slice(&bout[..num_state]);
        for (coll, b) in collected
            .iter_mut()
            .zip(&bout[num_state..num_state + num_scan_out])
        {
            coll.push(*b);
        }
    }

    for (o, &s) in n.outputs[..num_state].iter().zip(&state[..num_state]) {
        ctx.bind(o, s);
    }
    for (i, coll) in collected.iter().enumerate() {
        if coll.is_empty() {
            return Err("MLX Scan: cannot stack an empty scan output".to_string());
        }
        let rank = ctx.shape_of(coll[0]).len() + 1;
        let axis = normalized_axis(output_axes[i], rank).ok_or_else(|| {
            format!(
                "MLX Scan: output axis {} invalid for rank {rank}",
                output_axes[i]
            )
        })?;
        let parts: Vec<mlx::mlx_array> = if output_directions[i] == 0 {
            coll.clone()
        } else {
            coll.iter().rev().copied().collect()
        };
        let stacked = ctx.stack(&parts, axis as i32)?;
        ctx.bind(&n.outputs[num_state + i], stacked);
    }
    // Recorded AFTER the unroll: translating the body dispatches through the registry, which
    // resets the per-node path mark, so a mark set before the loop would be clobbered.
    ctx.mark_composed(decline_reason);
    Ok(())
}

fn scan_claim(node: &NodeView) -> ClaimResult {
    let ninputs = node.num_inputs();
    let noutputs = node.num_outputs();
    let num_scan = node.int_attr("num_scan_inputs", -1);
    require!(
        num_scan > 0 && (ninputs as i64) >= num_scan,
        "num_scan_inputs must be positive and no greater than the {} inputs, got {}",
        ninputs,
        num_scan
    );
    let num_state = ninputs as i64 - num_scan;
    require!(num_state >= 0, "num_scan_inputs exceeds input count");

    let read_attr = |name: &str, len: usize| -> Result<Vec<i64>, String> {
        let (present, values) = node.ints_attr(name);
        if !present {
            return Ok(vec![0; len]);
        }
        if values.len() != len {
            return Err(format!(
                "{name} has {} values, expected {len}",
                values.len()
            ));
        }
        Ok(values)
    };
    let input_axes = read_attr("scan_input_axes", num_scan as usize)?;
    let input_directions = read_attr("scan_input_directions", num_scan as usize)?;
    require!(
        input_directions.iter().all(|&v| v == 0 || v == 1),
        "scan_input_directions values must be 0 or 1"
    );

    let mut trip = None;
    for (scan_index, i) in (num_state..ninputs as i64).enumerate() {
        let info = match node.input_info(i as usize) {
            Some(info) => info,
            None => deny!("scan input {} lacks tensor type/shape info", i),
        };
        let axis = match normalized_axis(input_axes[scan_index], info.shape.len()) {
            Some(axis) => axis,
            None => deny!(
                "scan input {} axis {} is invalid for rank {}",
                i,
                input_axes[scan_index],
                info.shape.len()
            ),
        };
        require!(
            info.shape[axis] == -1 || info.shape[axis] >= 1,
            "scan input {} axis {} must be non-empty or dynamic, got shape {:?}",
            i,
            input_axes[scan_index],
            info.shape
        );
        if info.shape[axis] >= 1 {
            if let Some(expected) = trip {
                require!(
                    info.shape[axis] == expected,
                    "scan inputs must have equal known sequence lengths, got {} and {}",
                    expected,
                    info.shape[axis]
                );
            } else {
                trip = Some(info.shape[axis]);
            }
        }
    }

    let subs = node.subgraphs();
    require!(
        subs.len() == 1 && subs[0].0 == "body",
        "requires exactly one body subgraph"
    );
    let body = &subs[0].1;
    require!(
        body.input_names().len() as i64 == num_state + num_scan,
        "body has {} inputs, expected {} carried-state plus scan inputs",
        body.input_names().len(),
        num_state + num_scan
    );
    require!(
        (body.output_names().len() as i64) >= num_state,
        "body has {} outputs but requires at least {} carried-state outputs",
        body.output_names().len(),
        num_state
    );
    require!(
        noutputs == body.output_names().len(),
        "Scan has {} outputs but body has {}",
        noutputs,
        body.output_names().len()
    );
    require!(body_claimable(body), "{}", body_rejection(body));
    let num_scan_out = body.output_names().len() - num_state as usize;
    let output_axes_present = node.ints_attr("scan_output_axes").0;
    let output_axes = read_attr("scan_output_axes", num_scan_out)?;
    let output_directions = read_attr("scan_output_directions", num_scan_out)?;
    require!(
        output_directions.iter().all(|&v| v == 0 || v == 1),
        "scan_output_directions values must be 0 or 1"
    );
    if output_axes_present {
        for (i, &axis) in output_axes.iter().enumerate() {
            let output_index = num_state as usize + i;
            let info = match node.output_info(output_index) {
                Some(info) => info,
                None => deny!("scan output {} lacks tensor type/shape info", output_index),
            };
            require!(
                normalized_axis(axis, info.shape.len()).is_some(),
                "scan output {} axis {} is invalid for rank {}",
                output_index,
                axis,
                info.shape.len()
            );
        }
    }
    Ok(())
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
        bin.extend_from_slice(&state[..num_state]);
        let bout = ctx.run_subgraph(&body, &bin)?;
        // bout[0] = cond_out (pass-through, guaranteed true by claim); bout[1..] = carried state.
        state[..num_state].copy_from_slice(&bout[1..1 + num_state]);
    }

    for (o, &s) in n.outputs[..num_state].iter().zip(&state[..num_state]) {
        ctx.bind(o, s);
    }
    Ok(())
}

fn loop_claim(node: &NodeView) -> ClaimResult {
    const UNSUPPORTED: &str = "Loop: only static carried-state loops (constant trip-count + passthrough cond, no scan outputs) are unrolled; scan outputs / dynamic control stay on CPU";
    let ninputs = node.num_inputs();
    require!(ninputs >= 2, "{}", UNSUPPORTED);
    let num_state = ninputs - 2;
    require!(is_int64(node, 0) && is_bool(node, 1), "{}", UNSUPPORTED);

    let subs = node.subgraphs();
    require!(subs.len() == 1 && subs[0].0 == "body", "{}", UNSUPPORTED);
    let body = &subs[0].1;
    require!(body.input_names().len() == 2 + num_state, "{}", UNSUPPORTED);
    require!(
        body.output_names().len() == 1 + num_state,
        "{}",
        UNSUPPORTED
    );
    require!(node.num_outputs() == num_state, "{}", UNSUPPORTED);
    require!(loop_cond_is_passthrough(body), "{}", UNSUPPORTED);
    require!(body_claimable(body), "{}", UNSUPPORTED);
    Ok(())
}

// ---- registration -------------------------------------------------------------------------------

fn reg(
    registry: &mut OpRegistry,
    op_type: &'static str,
    handler: OpHandler,
    claim: ClaimPredicate,
) {
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
