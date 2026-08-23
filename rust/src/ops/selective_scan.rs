//! Fused Mamba-1 selective scan.
//!
//! The generic `Scan` handler in `controlflow.rs` fully unrolls the body: one MLX subgraph copy
//! per timestep. For a Mamba-1 selective scan that is disastrous. In the RE-USE / SEMamba
//! speech-enhancement model there are 120 `Scan` nodes with sequence lengths of 442 (frequency
//! axis) and 256 (time axis), so unrolling emits on the order of half a million MLX nodes and the
//! recurrence becomes launch-bound: ~14 tiny kernels per timestep, each moving only a few MB.
//!
//! This module recognises that specific body and replaces the whole unroll with ONE custom Metal
//! kernel. Each thread owns a single `(batch, d_inner)` pair and keeps its `d_state` accumulator in
//! registers, so the running state is never materialised to device memory at all.
//!
//! Why not an associative / `cumsum`-based scan? Two independent reasons, both measured on the
//! real checkpoint rather than assumed:
//!
//!   * The textbook identity `h_t = exp(L_t) * cumsum(dBx_s * exp(-L_s))`, `L = cumsum(dt*A)`, is
//!     unusable here. On the real weights `dt` reaches 2.78 and `A` reaches -46.4, so a SINGLE
//!     step's log-decay reaches -129 — already past the fp32 `exp` range — and `exp(-L)` overflows
//!     to `inf`. Chunking does not rescue it: the worst in-chunk `|L|` is 1063 even at chunk 16.
//!   * A division-free associative (Hillis-Steele) scan IS numerically sound, but it has to
//!     materialise the `(chunk, batch, d_inner, d_state)` state, a `d_state`-fold (16x) bandwidth
//!     amplification. Measured on the real shapes it runs 4.7-6.3x SLOWER than the existing unroll
//!     and 36-49x slower than this kernel. The recurrence is bandwidth-bound, not depth-bound, so
//!     trading sequential depth for memory traffic loses.
//!
//! Any body that does not match this exact pattern, or that uses a shape/dtype the kernel does not
//! cover, falls through to the generic unroll unchanged.
//!
//! `T` and `B` are Metal template arguments, so MLX compiles and process-globally caches a pipeline
//! for each distinct sequence/batch shape. This is ideal for the fixed-shape RE-USE graph, but a
//! variable-length deployment pays a one-time Metal compile for every new shape.

use crate::engine::{MlxError, NodeDesc, Src, SubgraphDesc, TensorRef, TranslationContext};
use crate::sys::mlx;
use crate::sys::ort;
use std::collections::HashMap;
use std::ffi::CString;

const F32: mlx::mlx_dtype = mlx::mlx_dtype__MLX_FLOAT32;

/// Trace/profile name for the fused path, so a silent fallback is detectable.
pub const KERNEL_NAME: &str = "mlx_selective_scan";

/// Kill-switch: set to a non-empty value other than `0` to force the generic unroll. Used by tests
/// to compare the fused kernel against the unrolled reference in-process.
pub const NO_FUSE_ENV: &str = "ONNXRUNTIME_EP_MLX_NO_SELECTIVE_SCAN";

/// True when the fused selective scan is disabled by environment.
pub fn disabled() -> bool {
    std::env::var_os(NO_FUSE_ENV)
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

/// Largest `d_state` we keep in per-thread registers. Mamba-1 uses 16; the cap keeps the kernel
/// from spilling into local memory (which would erase the win) on an unexpected model.
const MAX_D_STATE: i32 = 32;

/// Which body input / output slot plays each role in a matched selective scan.
pub struct SelectiveScan {
    /// Index into the Scan node's STATE inputs holding the running state `h`.
    pub h_state: usize,
    /// Index into the Scan node's STATE inputs holding the pass-through `a_neg`.
    pub a_state: usize,
    /// Indices into the Scan node's SCAN inputs.
    pub dt: usize,
    pub b: usize,
    pub c: usize,
    pub x: usize,
    /// Index into the Scan node's outputs for the per-step readout `y`.
    pub out_y: usize,
}

// ---- structural matching -------------------------------------------------------------------------

/// Statically-known int64 list behind a node input: a body initializer, or a `Constant` node's
/// tensor attribute. Returns `None` for anything computed at run time, which declines the match.
fn static_ints(body_consts: &HashMap<&str, &NodeDesc>, r: &TensorRef) -> Option<Vec<i64>> {
    if r.source == Src::Absent || r.name.is_empty() {
        return None;
    }
    let read =
        |data: &[u8], count: usize, dtype: ort::ONNXTensorElementDataType| -> Option<Vec<i64>> {
            if dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 {
                return None;
            }
            let byte_count = count.checked_mul(std::mem::size_of::<i64>())?;
            if data.len() < byte_count {
                return None;
            }
            Some(
                (0..count)
                    .map(|i| i64::from_le_bytes(data[i * 8..i * 8 + 8].try_into().unwrap()))
                    .collect(),
            )
        };
    if let Some(init) = &r.init {
        if init.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            || init.data.is_null()
        {
            return None;
        }
        let byte_count = init.count.checked_mul(std::mem::size_of::<i64>())?;
        let bytes = unsafe { std::slice::from_raw_parts(init.data as *const u8, byte_count) };
        return read(bytes, init.count, init.dtype);
    }
    let node = body_consts.get(r.name.as_str())?;
    let t = node.tensors.get("value")?;
    read(&t.data, t.count, t.dtype)
}

struct Matcher<'a> {
    /// Body value name -> the node that produces it.
    producer: HashMap<&'a str, &'a NodeDesc>,
    /// Body value name -> the `Constant` node that produces it (subset of `producer`).
    consts: HashMap<&'a str, &'a NodeDesc>,
    /// Body graph input names, in declaration order.
    inputs: &'a [String],
}

impl<'a> Matcher<'a> {
    fn new(body: &'a SubgraphDesc) -> Self {
        let mut producer = HashMap::new();
        let mut consts = HashMap::new();
        for n in &body.nodes {
            for o in &n.outputs {
                if !o.name.is_empty() {
                    producer.insert(o.name.as_str(), n);
                    if n.op_type == "Constant" {
                        consts.insert(o.name.as_str(), n);
                    }
                }
            }
        }
        Matcher {
            producer,
            consts,
            inputs: &body.input_names,
        }
    }

    /// The node producing `name`, if it has op type `op`.
    fn def(&self, name: &str, op: &str) -> Option<&'a NodeDesc> {
        let n = self.producer.get(name)?;
        (n.op_type == op && n.domain.is_empty()).then_some(*n)
    }

    /// Position of `name` in the body's input list.
    fn input_slot(&self, name: &str) -> Option<usize> {
        self.inputs.iter().position(|i| i == name)
    }

    /// Match `Unsqueeze(<body input>, axes)` and return that input's slot, checking the axes match.
    fn unsqueeze_of_input(&self, name: &str, want_axes: &[i64]) -> Option<usize> {
        let n = self.def(name, "Unsqueeze")?;
        if n.inputs.len() < 2 {
            // opset<13 keeps axes as an attribute.
            let axes = n.int_arrays.get("axes")?;
            if axes.as_slice() != want_axes {
                return None;
            }
        } else {
            let axes = static_ints(&self.consts, &n.inputs[1])?;
            if axes.as_slice() != want_axes {
                return None;
            }
        }
        self.input_slot(&n.inputs[0].name)
    }

    /// Operands of a binary elementwise node, as names.
    fn binary(&self, name: &str, op: &str) -> Option<(&'a str, &'a str)> {
        let n = self.def(name, op)?;
        (n.inputs.len() == 2).then(|| (n.inputs[0].name.as_str(), n.inputs[1].name.as_str()))
    }
}

/// Recognise the Mamba-1 selective-scan body:
///
/// ```text
///   dt_col   = Unsqueeze(dt_t, [-1])
///   dA       = Exp(Mul(dt_col, Unsqueeze(a_neg, [0])))
///   dBx      = Mul(Mul(dt_col, Unsqueeze(x_t, [-1])), Unsqueeze(b_t, [1]))
///   h_out    = Add(Mul(dA, h_in), dBx)
///   a_out    = Identity(a_neg)
///   y_t      = ReduceSum(Mul(h_out, Unsqueeze(c_t, [1])), [-1], keepdims=0)
/// ```
///
/// `Mul`/`Add` operands are accepted in either order. Returns `None` (decline, keep unrolling) for
/// anything else.
pub fn match_body(body: &SubgraphDesc, num_state: usize, num_scan: usize) -> Option<SelectiveScan> {
    if num_state != 2 || num_scan != 4 || body.output_names.len() != 3 {
        return None;
    }
    let m = Matcher::new(body);
    if m.inputs.len() != 6 {
        return None;
    }

    // --- y_t = ReduceSum(Mul(h_out, Unsqueeze(c_t,[1])), [-1], keepdims=0) ---
    let (out_y, y_name) = (2usize, body.output_names[2].as_str());
    debug_assert_eq!(out_y, num_state, "y is the first scan output");
    let rs = m.def(y_name, "ReduceSum")?;
    if rs.ints.get("keepdims").copied().unwrap_or(1) != 0 {
        return None;
    }
    let rs_axes = if rs.inputs.len() >= 2 {
        static_ints(&m.consts, &rs.inputs[1])?
    } else {
        rs.int_arrays.get("axes")?.clone()
    };
    // The reduction must be over the last (d_state) axis of a rank-3 value.
    if rs_axes.len() != 1 || !(rs_axes[0] == -1 || rs_axes[0] == 2) {
        return None;
    }
    let (ym0, ym1) = m.binary(&rs.inputs[0].name, "Mul")?;

    // --- h_out = Add(Mul(dA, h_in), dBx) ---
    let h_name = body.output_names[0].as_str();
    let (add0, add1) = m.binary(h_name, "Add")?;

    // The ReduceSum must consume the NEW state (post-update), not the carry.
    let c_side = if ym0 == h_name {
        ym1
    } else if ym1 == h_name {
        ym0
    } else {
        return None;
    };
    let c = m.unsqueeze_of_input(c_side, &[1])?;

    // One Add operand is Mul(dA, h_in); the other is dBx.
    let mut decay: Option<(usize, usize)> = None; // (dt slot, a slot)
    let mut h_state: Option<usize> = None;
    let mut dbx_name: Option<&str> = None;
    for (cand, other) in [(add0, add1), (add1, add0)] {
        let Some((p, q)) = m.binary(cand, "Mul") else {
            continue;
        };
        for (exp_side, state_side) in [(p, q), (q, p)] {
            let Some(e) = m.def(exp_side, "Exp") else {
                continue;
            };
            let Some((e0, e1)) = m.binary(&e.inputs[0].name, "Mul") else {
                continue;
            };
            // Mul(dt_col, Unsqueeze(a_neg,[0])) in either order.
            for (dt_side, a_side) in [(e0, e1), (e1, e0)] {
                let Some(dt) = m.unsqueeze_of_input(dt_side, &[-1]) else {
                    continue;
                };
                let Some(a) = m.unsqueeze_of_input(a_side, &[0]) else {
                    continue;
                };
                if let Some(hs) = m.input_slot(state_side) {
                    decay = Some((dt, a));
                    h_state = Some(hs);
                    dbx_name = Some(other);
                }
            }
        }
    }
    let (dt_slot, a_slot) = decay?;
    let h_state_slot = h_state?;
    let dbx = dbx_name?;

    // --- dBx = Mul(Mul(dt_col, Unsqueeze(x_t,[-1])), Unsqueeze(b_t,[1])) ---
    let (d0, d1) = m.binary(dbx, "Mul")?;
    let mut xb: Option<(usize, usize)> = None;
    for (inner, b_side) in [(d0, d1), (d1, d0)] {
        let Some(b) = m.unsqueeze_of_input(b_side, &[1]) else {
            continue;
        };
        let Some((i0, i1)) = m.binary(inner, "Mul") else {
            continue;
        };
        for (dt_side, x_side) in [(i0, i1), (i1, i0)] {
            // Must reuse the SAME dt_col as the decay term.
            if m.unsqueeze_of_input(dt_side, &[-1]) != Some(dt_slot) {
                continue;
            }
            if let Some(x) = m.unsqueeze_of_input(x_side, &[-1]) {
                xb = Some((x, b));
            }
        }
    }
    let (x_slot, b_slot) = xb?;

    // --- a_out = Identity(a_neg) ---
    let ident = m.def(body.output_names[1].as_str(), "Identity")?;
    if m.input_slot(&ident.inputs[0].name)? != a_slot {
        return None;
    }

    // State slots come first in the body's input list, scan slots after.
    // The Scan node's state outputs are positional. The recognised body emits h then pass-through
    // a, so accepting swapped input slots would bind the final states in the wrong order.
    if h_state_slot != 0 || a_slot != 1 || num_state < 2 {
        return None;
    }
    let scan_slot = |s: usize| -> Option<usize> {
        (s >= num_state && s < num_state + num_scan).then(|| s - num_state)
    };
    let (dt, b, c, x) = (
        scan_slot(dt_slot)?,
        scan_slot(b_slot)?,
        scan_slot(c)?,
        scan_slot(x_slot)?,
    );
    // All four scan roles must be distinct inputs.
    let mut seen = [dt, b, c, x];
    seen.sort_unstable();
    if seen.windows(2).any(|w| w[0] == w[1]) {
        return None;
    }

    Some(SelectiveScan {
        h_state: h_state_slot,
        a_state: a_slot,
        dt,
        b,
        c,
        x,
        out_y,
    })
}

// (scan-slot mapping is inlined in `match_body`.)

// ---- fused kernel ----------------------------------------------------------------------------

/// Metal source. One thread per `(batch, d_inner)`; the `d_state` accumulator lives in registers,
/// so the running state never touches device memory. `precise::exp` matches the ONNX `Exp` the
/// unrolled path emits; an extreme negative argument simply underflows to 0, which is exactly what
/// the sequential fp32 reference does and is the benign direction.
const KERNEL_SRC: &str = r#"
    uint gid = thread_position_in_grid.x;
    uint BD = (uint)B * (uint)D;
    if (gid >= BD) { return; }
    uint bi = gid / (uint)D;
    uint di = gid % (uint)D;

    float h[NSTATE];
    float aloc[NSTATE];
    uint hbase = bi * (uint)D * (uint)NSTATE + di * (uint)NSTATE;
    for (uint n = 0; n < (uint)NSTATE; ++n) {
        h[n] = h0[hbase + n];
        aloc[n] = a[di * (uint)NSTATE + n];
    }

    for (uint i = 0; i < (uint)T; ++i) {
        uint t = IN_REVERSE ? ((uint)T - 1u - i) : i;
        uint o = OUT_REVERSE ? ((uint)T - 1u - i) : i;
        uint od = t * BD + bi * (uint)D + di;
        uint ob = t * (uint)B * (uint)NSTATE + bi * (uint)NSTATE;
        float dtv = dt[od];
        float dxv = dtv * x[od];
        float acc = 0.0f;
        for (uint n = 0; n < (uint)NSTATE; ++n) {
            float dA = precise::exp(dtv * aloc[n]);
            h[n] = dA * h[n] + dxv * bmat[ob + n];
            acc += h[n] * cmat[ob + n];
        }
        y[o * BD + bi * (uint)D + di] = acc;
    }

    for (uint n = 0; n < (uint)NSTATE; ++n) { hout[hbase + n] = h[n]; }
"#;

struct VecString(mlx::mlx_vector_string);

impl VecString {
    fn new(items: &[&str]) -> Result<Self, MlxError> {
        let raw = unsafe { mlx::mlx_vector_string_new() };
        for it in items {
            let s = CString::new(*it).map_err(|_| "MLX SelectiveScan: bad name".to_string())?;
            unsafe { mlx::mlx_vector_string_append_value(raw, s.as_ptr()) };
        }
        Ok(VecString(raw))
    }
}

impl Drop for VecString {
    fn drop(&mut self) {
        unsafe { mlx::mlx_vector_string_free(self.0) };
    }
}

/// Emit the fused kernel. Returns `Ok(None)` when this particular instance is not covered (wrong
/// rank, dtype or `d_state`), so the caller can fall back to the generic unroll.
///
/// `h0` is the incoming state `(batch, d_inner, d_state)`, `a` is `(d_inner, d_state)`, and `dt`,
/// `b`, `c`, `x` are time-major `(seq, batch, ...)`. Produces `(h_final, y)`.
#[allow(clippy::too_many_arguments)]
pub fn emit(
    ctx: &mut TranslationContext,
    h0: mlx::mlx_array,
    a: mlx::mlx_array,
    dt: mlx::mlx_array,
    b: mlx::mlx_array,
    c: mlx::mlx_array,
    x: mlx::mlx_array,
    in_reverse: bool,
    out_reverse: bool,
) -> Result<Option<(mlx::mlx_array, mlx::mlx_array)>, MlxError> {
    for arr in [h0, a, dt, b, c, x] {
        if ctx.dtype_of(arr) != F32 {
            return Ok(None);
        }
    }
    let (sh_h0, sh_a) = (ctx.shape_of(h0), ctx.shape_of(a));
    let (sh_dt, sh_b, sh_c, sh_x) = (
        ctx.shape_of(dt),
        ctx.shape_of(b),
        ctx.shape_of(c),
        ctx.shape_of(x),
    );
    if sh_h0.len() != 3 || sh_a.len() != 2 || sh_dt.len() != 3 || sh_x.len() != 3 {
        return Ok(None);
    }
    if sh_b.len() != 3 || sh_c.len() != 3 {
        return Ok(None);
    }
    let (t, batch, d_inner, d_state) = (sh_dt[0], sh_dt[1], sh_dt[2], sh_a[1]);
    if d_state <= 0 || d_state > MAX_D_STATE || t <= 0 || batch <= 0 || d_inner <= 0 {
        return Ok(None);
    }
    if sh_a[0] != d_inner
        || sh_h0 != [batch, d_inner, d_state]
        || sh_x != [t, batch, d_inner]
        || sh_b != [t, batch, d_state]
        || sh_c != [t, batch, d_state]
    {
        return Ok(None);
    }

    let name = CString::new("onnxrt_mlx_selective_scan").unwrap();
    let src = CString::new(KERNEL_SRC).unwrap();
    let header = CString::new("").unwrap();
    let in_names = VecString::new(&["h0", "a", "dt", "bmat", "cmat", "x"])?;
    let out_names = VecString::new(&["hout", "y"])?;

    let kernel = unsafe {
        mlx::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            in_names.0,
            out_names.0,
            src.as_ptr(),
            header.as_ptr(),
            /* ensure_row_contiguous */ true,
            /* atomic_outputs */ false,
        )
    };
    if kernel.ctx.is_null() {
        return Ok(None);
    }
    let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };

    let mut rc = 0;
    let mut set_int = |k: &str, v: i32| {
        let key = CString::new(k).unwrap();
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, key.as_ptr(), v)
        };
    };
    set_int("T", t);
    set_int("B", batch);
    set_int("D", d_inner);
    set_int("NSTATE", d_state);
    set_int("IN_REVERSE", in_reverse as i32);
    set_int("OUT_REVERSE", out_reverse as i32);

    let h_shape = [batch, d_inner, d_state];
    let y_shape = [t, batch, d_inner];
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_add_output_arg(config, h_shape.as_ptr(), 3, F32)
    };
    rc |= unsafe {
        mlx::mlx_fast_metal_kernel_config_add_output_arg(config, y_shape.as_ptr(), 3, F32)
    };

    // One thread per (batch, d_inner) pair.
    let threads = batch.saturating_mul(d_inner);
    let group = threads.min(256).max(1);
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config, threads, 1, 1) };
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_thread_group(config, group, 1, 1) };

    let mut inputs = crate::mlx::VectorArray::new();
    for arr in [h0, a, dt, b, c, x] {
        inputs.append(arr);
    }
    let mut outputs = crate::mlx::VectorArray::new();
    if rc == 0 {
        rc = unsafe {
            mlx::mlx_fast_metal_kernel_apply(
                outputs.as_mut_ptr(),
                kernel,
                inputs.as_raw(),
                config,
                ctx.stream(),
            )
        };
    }
    unsafe {
        mlx::mlx_fast_metal_kernel_config_free(config);
        mlx::mlx_fast_metal_kernel_free(kernel);
    }
    if rc != 0 || outputs.size() != 2 {
        return Ok(None);
    }
    let hout = ctx.keep(outputs.get(0));
    let yout = ctx.keep(outputs.get(1));
    Ok(Some((hout, yout)))
}
