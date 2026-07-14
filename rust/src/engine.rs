//! Engine core: the plan description (`NodeDesc`/`TensorRef`/`OutRef`) and the `TranslationContext`
//! a handler uses to Resolve inputs, Bind outputs, and emit MLX ops.
//!
//! This is a faithful port of the C++ `mlx_engine.h` / `mlx_backend.cc` translation core, restricted
//! to the wave-1 (eager, single-`mlx_eval` boundary) path: no compiled-decode fast-path, no
//! control-flow subgraphs. Those are called out as next-wave work in the README.

use std::collections::HashMap;
use std::os::raw::c_void;

use crate::mlx::{self, Array, VectorArray};
use crate::sys::mlx as mlxsys;
use crate::sys::ort;

/// Where a node input resolves from (mirrors the C++ `Src`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Src {
    CtxInput,
    Initializer,
    Intermediate,
    Absent,
}

/// A constant initializer payload surfaced at compile time (session-owned storage).
#[derive(Clone)]
pub struct InitData {
    pub data: *const c_void,
    pub shape: Vec<i64>,
    pub dtype: ort::ONNXTensorElementDataType,
    /// Element count of the initializer (kept for weight-repack handlers in the next wave).
    #[allow(dead_code)]
    pub count: usize,
}

/// A single node input reference.
#[derive(Clone)]
pub struct TensorRef {
    pub name: String,
    pub source: Src,
    pub ctx_index: usize,
    /// True when a CtxInput is a hoisted constant initializer (wrapped/cached once).
    pub constant: bool,
    pub init: Option<InitData>,
}

impl TensorRef {
    pub fn absent() -> Self {
        TensorRef {
            name: String::new(),
            source: Src::Absent,
            ctx_index: 0,
            constant: false,
            init: None,
        }
    }
}

/// A single node output reference.
#[derive(Clone)]
pub struct OutRef {
    pub name: String,
    /// A subgraph boundary output routed to `KernelContext_GetOutput(ctx_index)`.
    pub external: bool,
    pub ctx_index: usize,
    pub otype: ort::ONNXTensorElementDataType,
}

/// One ONNX node with just the metadata the MLX translator needs.
#[derive(Clone)]
pub struct NodeDesc {
    pub op_type: String,
    pub domain: String,
    pub since_version: i32,
    pub ints: HashMap<String, i64>,
    pub floats: HashMap<String, f32>,
    pub int_arrays: HashMap<String, Vec<i64>>,
    pub float_arrays: HashMap<String, Vec<f32>>,
    pub strings: HashMap<String, String>,
    pub inputs: Vec<TensorRef>,
    pub outputs: Vec<OutRef>,
}

impl NodeDesc {
    pub fn new(op_type: String, domain: String, since_version: i32) -> Self {
        NodeDesc {
            op_type,
            domain,
            since_version,
            ints: HashMap::new(),
            floats: HashMap::new(),
            int_arrays: HashMap::new(),
            float_arrays: HashMap::new(),
            strings: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }
}

/// Persistent per-subgraph MLX state: the topo-ordered nodes plus the persistent cache of
/// wrapped/repacked constant arrays (keyed by name, reused across runs — freed with the plan).
pub struct Plan {
    pub nodes: Vec<NodeDesc>,
    pub cache: HashMap<String, Array>,
}

impl Plan {
    pub fn new(nodes: Vec<NodeDesc>) -> Self {
        Plan {
            nodes,
            cache: HashMap::new(),
        }
    }
}

/// ONNX tensor element type -> MLX dtype (faithful port of `MlxDtypeFromOnnx`). Unknown types fall
/// back to fp32 so a stray dtype never crashes the wrap.
pub fn mlx_dtype_from_onnx(t: ort::ONNXTensorElementDataType) -> mlxsys::mlx_dtype {
    use ort::*;
    #[allow(non_upper_case_globals)]
    match t {
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => mlxsys::mlx_dtype__MLX_FLOAT32,
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => {
            mlxsys::mlx_dtype__MLX_FLOAT16
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => {
            mlxsys::mlx_dtype__MLX_BFLOAT16
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => {
            mlxsys::mlx_dtype__MLX_FLOAT64
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => mlxsys::mlx_dtype__MLX_INT8,
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => mlxsys::mlx_dtype__MLX_INT16,
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => mlxsys::mlx_dtype__MLX_INT32,
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => mlxsys::mlx_dtype__MLX_INT64,
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => mlxsys::mlx_dtype__MLX_UINT8,
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => {
            mlxsys::mlx_dtype__MLX_UINT16
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => {
            mlxsys::mlx_dtype__MLX_UINT32
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => {
            mlxsys::mlx_dtype__MLX_UINT64
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => mlxsys::mlx_dtype__MLX_BOOL,
        _ => mlxsys::mlx_dtype__MLX_FLOAT32,
    }
}

/// A translation/runtime error (mirrors the C++ `MlxError`, caught in `RunPlan`).
pub type MlxError = String;

/// Per-Compute execution context: builds the MLX graph for one forward pass, evals once, copies out.
/// Handlers receive `&mut TranslationContext` and use Resolve/Bind + the MLX op helpers.
pub struct TranslationContext<'a> {
    plan: &'a mut Plan,
    ort_api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
    stream: mlxsys::mlx_stream,
    /// name -> raw handle (borrowed; owned by `arena` or the plan cache).
    env: HashMap<String, mlxsys::mlx_array>,
    /// All arrays produced this run; freed together on drop (RAII, no per-site frees).
    arena: Vec<Array>,
}

impl<'a> TranslationContext<'a> {
    pub fn new(
        plan: &'a mut Plan,
        ort_api: *const ort::OrtApi,
        kctx: *mut ort::OrtKernelContext,
        stream: mlxsys::mlx_stream,
    ) -> Self {
        TranslationContext {
            plan,
            ort_api,
            kctx,
            stream,
            env: HashMap::new(),
            arena: Vec::new(),
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn stream(&self) -> mlxsys::mlx_stream {
        self.stream
    }

    /// Register a freshly produced array for teardown at end of run; returns its raw handle.
    pub fn keep(&mut self, a: Array) -> mlxsys::mlx_array {
        let raw = a.as_raw();
        self.arena.push(a);
        raw
    }

    /// Bind a node output name to a produced MLX array (visible to downstream nodes and CopyOut).
    pub fn bind(&mut self, o: &OutRef, a: mlxsys::mlx_array) {
        self.env.insert(o.name.clone(), a);
    }

    /// Resolve a node input to a raw MLX array handle (intermediate env / wrapped ctx input /
    /// cached-or-wrapped initializer). Faithful port of `TranslationContext::Resolve`.
    pub fn resolve(&mut self, r: &TensorRef) -> Result<mlxsys::mlx_array, MlxError> {
        match r.source {
            Src::Intermediate => self
                .env
                .get(&r.name)
                .copied()
                .ok_or_else(|| format!("MLX: missing intermediate {}", r.name)),
            Src::CtxInput => {
                if r.constant {
                    if let Some(a) = self.plan.cache.get(&r.name) {
                        return Ok(a.as_raw());
                    }
                } else if let Some(a) = self.env.get(&r.name) {
                    return Ok(*a);
                }
                let (data, shape, dtype) = self.read_ctx_input(r.ctx_index)?;
                let ishape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
                let arr = Array::from_data(data, &ishape, mlx_dtype_from_onnx(dtype));
                if r.constant {
                    let raw = arr.as_raw();
                    self.plan.cache.insert(r.name.clone(), arr);
                    Ok(raw)
                } else {
                    let raw = self.keep(arr);
                    self.env.insert(r.name.clone(), raw);
                    Ok(raw)
                }
            }
            Src::Initializer => {
                if let Some(a) = self.plan.cache.get(&r.name) {
                    return Ok(a.as_raw());
                }
                let init = r
                    .init
                    .as_ref()
                    .ok_or_else(|| format!("MLX: initializer {} has no data", r.name))?;
                let ishape: Vec<i32> = init.shape.iter().map(|&d| d as i32).collect();
                let arr = Array::from_data(init.data, &ishape, mlx_dtype_from_onnx(init.dtype));
                let raw = arr.as_raw();
                self.plan.cache.insert(r.name.clone(), arr);
                Ok(raw)
            }
            Src::Absent => Err("MLX: absent input".to_string()),
        }
    }

    /// Read an ORT kernel-context input tensor: (data ptr, shape, element type).
    fn read_ctx_input(
        &self,
        index: usize,
    ) -> Result<(*const c_void, Vec<i64>, ort::ONNXTensorElementDataType), MlxError> {
        unsafe {
            let api = &*self.ort_api;
            let mut val: *const ort::OrtValue = std::ptr::null();
            let st = (api.KernelContext_GetInput.unwrap())(self.kctx, index, &mut val);
            if !st.is_null() || val.is_null() {
                return Err(format!("MLX: KernelContext_GetInput({index}) failed"));
            }
            let mut info: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            (api.GetTensorTypeAndShape.unwrap())(val, &mut info);
            let mut nd: usize = 0;
            (api.GetDimensionsCount.unwrap())(info, &mut nd);
            let mut dims = vec![0i64; nd];
            if nd > 0 {
                (api.GetDimensions.unwrap())(info, dims.as_mut_ptr(), nd);
            }
            let mut etype: ort::ONNXTensorElementDataType = 0;
            (api.GetTensorElementType.unwrap())(info, &mut etype);
            (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info);

            let mut data: *const c_void = std::ptr::null();
            (api.GetTensorData.unwrap())(val, &mut data);
            Ok((data, dims, etype))
        }
    }

    // ---- MLX op helpers (each keeps and returns the raw result) --------------------------------

    /// Apply a unary `mlx_*(res, a, stream)` op.
    pub fn unary(
        &mut self,
        op: unsafe extern "C" fn(*mut mlxsys::mlx_array, mlxsys::mlx_array, mlxsys::mlx_stream) -> i32,
        a: mlxsys::mlx_array,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let mut res = Array::new();
        let mut raw = res.as_raw();
        let rc = unsafe { op(&mut raw, a, self.stream) };
        // The op may replace the handle; re-wrap whatever it produced.
        res = Array::from_raw(raw);
        if rc != 0 {
            return Err("mlx unary op failed".to_string());
        }
        Ok(self.keep(res))
    }

    /// Apply a binary `mlx_*(res, a, b, stream)` op.
    pub fn binary(
        &mut self,
        op: unsafe extern "C" fn(
            *mut mlxsys::mlx_array,
            mlxsys::mlx_array,
            mlxsys::mlx_array,
            mlxsys::mlx_stream,
        ) -> i32,
        a: mlxsys::mlx_array,
        b: mlxsys::mlx_array,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let mut res = Array::new();
        let mut raw = res.as_raw();
        let rc = unsafe { op(&mut raw, a, b, self.stream) };
        res = Array::from_raw(raw);
        if rc != 0 {
            return Err("mlx binary op failed".to_string());
        }
        Ok(self.keep(res))
    }

    /// `astype(a, t)` — cast to another dtype.
    pub fn astype(
        &mut self,
        a: mlxsys::mlx_array,
        t: mlxsys::mlx_dtype,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let mut res = Array::new();
        let mut raw = res.as_raw();
        let rc = unsafe { mlxsys::mlx_astype(&mut raw, a, t, self.stream) };
        res = Array::from_raw(raw);
        if rc != 0 {
            return Err("mlx_astype failed".to_string());
        }
        Ok(self.keep(res))
    }

    /// `zeros_like(a)`.
    pub fn zeros_like(&mut self, a: mlxsys::mlx_array) -> Result<mlxsys::mlx_array, MlxError> {
        let mut res = Array::new();
        let mut raw = res.as_raw();
        let rc = unsafe { mlxsys::mlx_zeros_like(&mut raw, a, self.stream) };
        res = Array::from_raw(raw);
        if rc != 0 {
            return Err("mlx_zeros_like failed".to_string());
        }
        Ok(self.keep(res))
    }

    /// Softmax over the last axis (precise), used by the Softmax handler.
    pub fn softmax_last_axis(
        &mut self,
        a: mlxsys::mlx_array,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let mut res = Array::new();
        let mut raw = res.as_raw();
        let rc = unsafe { mlxsys::mlx_softmax_axis(&mut raw, a, -1, true, self.stream) };
        res = Array::from_raw(raw);
        if rc != 0 {
            return Err("mlx_softmax_axis failed".to_string());
        }
        Ok(self.keep(res))
    }

    /// Translate every node, cast+collect boundary outputs, one `mlx_eval`, copy each output back
    /// across the ORT boundary. Faithful port of `ExecuteEager`.
    pub fn execute(&mut self) -> Result<(), MlxError> {
        let nodes = std::mem::take(&mut self.plan.nodes);
        let mut result = Ok(());
        for node in &nodes {
            if let Err(e) = crate::registry::translate(self, node) {
                result = Err(e);
                break;
            }
        }
        if result.is_ok() {
            result = self.finish_boundary(&nodes);
        }
        // Restore the plan's node list (we only borrowed it).
        self.plan.nodes = nodes;
        result
    }

    fn finish_boundary(&mut self, nodes: &[NodeDesc]) -> Result<(), MlxError> {
        // Collect boundary outputs, cast each to its ORT output dtype BEFORE eval so copy-out is a
        // straight typed memcpy, then eval the whole graph in one shot.
        let mut outs = VectorArray::new();
        let mut ext: Vec<(OutRef, mlxsys::mlx_array)> = Vec::new();
        for node in nodes {
            for o in &node.outputs {
                if o.external {
                    if let Some(&a) = self.env.get(&o.name) {
                        let casted = self.astype(a, mlx_dtype_from_onnx(o.otype))?;
                        outs.append(casted);
                        ext.push((o.clone(), casted));
                    }
                }
            }
        }
        // The single synchronous `mlx_eval` boundary: with tracing on this is wrapped
        // in the `mlx.eval` (cat "gpu") span, whose CPU wall time is the GPU-inclusive
        // time of the whole fused subgraph (MLX blocks here until the GPU work lands).
        // Sample GPU-memory counters just before and after so the curve shows the eval.
        let tr = crate::trace::tracer();
        tr.sample_gpu_counters();
        {
            let _eval = tr.eval_region();
            mlx::eval(&outs)?;
        }
        tr.sample_gpu_counters();
        for (o, a) in &ext {
            self.copy_out(o, *a)?;
        }
        Ok(())
    }

    /// Create the ORT output tensor with the MLX result shape and memcpy on unified memory.
    fn copy_out(&self, o: &OutRef, a: mlxsys::mlx_array) -> Result<(), MlxError> {
        let arr = std::mem::ManuallyDrop::new(Array::from_raw(a)); // borrow, do not free
        let shape = arr.shape();
        let count: usize = shape.iter().map(|&d| d as usize).product::<usize>().max(0);
        let itemsize = arr.itemsize();
        unsafe {
            let api = &*self.ort_api;
            let mut out: *mut ort::OrtValue = std::ptr::null_mut();
            let st = (api.KernelContext_GetOutput.unwrap())(
                self.kctx,
                o.ctx_index,
                shape.as_ptr(),
                shape.len(),
                &mut out,
            );
            if !st.is_null() || out.is_null() {
                return Err(format!("MLX: KernelContext_GetOutput({}) failed", o.ctx_index));
            }
            let mut dst: *mut c_void = std::ptr::null_mut();
            (api.GetTensorMutableData.unwrap())(out, &mut dst);
            let src = arr.data_bytes();
            if !src.is_null() && !dst.is_null() {
                std::ptr::copy_nonoverlapping(src, dst as *mut u8, count * itemsize);
            }
        }
        Ok(())
    }
}
