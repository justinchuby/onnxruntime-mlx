//! The (domain, op_type, [min,max] opset) -> { handler, claim predicate } registry — the single
//! source of truth for which ops the MLX EP can translate. Both the claim-time membership check
//! (GetCapability) and the run-time translator dispatch through the SAME table, so "claimed" and
//! "translatable" can never disagree (faithful port of `op_registry.{h,cc}`).

use std::os::raw::c_char;
use std::sync::OnceLock;

use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::sys::ort;

/// A translation handler: reads a NodeDesc, emits MLX ops through the context, binds the outputs.
pub type OpHandler = fn(&mut TranslationContext, &NodeDesc) -> Result<(), MlxError>;

/// A claim-time predicate: given the concrete ONNX node, decide whether MLX can translate it exactly
/// (dtypes / shapes / attributes / input form). The (domain, op_type, opset) key is matched first.
pub type ClaimPredicate = fn(&NodeView) -> bool;

/// Sentinel for an unbounded opset endpoint.
pub const K_ANY_OPSET: i32 = -1;

/// One registry entry: match (domain, op_type) with since_version in [min_opset, max_opset].
pub struct OpRegistration {
    pub domain: &'static str,
    pub op_type: &'static str,
    pub min_opset: i32,
    pub max_opset: i32,
    pub handler: OpHandler,
    pub claim: ClaimPredicate,
}

/// The opset-aware (domain, op) -> entry table (process-wide singleton).
pub struct OpRegistry {
    table: Vec<OpRegistration>,
}

impl OpRegistry {
    fn new() -> Self {
        OpRegistry { table: Vec::new() }
    }

    pub fn register(&mut self, entry: OpRegistration) {
        self.table.push(entry);
    }

    /// The matching entry for (domain, op_type, since_version), or None.
    pub fn find_entry(
        &self,
        domain: &str,
        op_type: &str,
        since_version: i32,
    ) -> Option<&OpRegistration> {
        self.table.iter().find(|e| {
            e.domain == domain
                && e.op_type == op_type
                && (e.min_opset == K_ANY_OPSET || since_version >= e.min_opset)
                && (e.max_opset == K_ANY_OPSET || since_version <= e.max_opset)
        })
    }
}

fn registry() -> &'static OpRegistry {
    static REGISTRY: OnceLock<OpRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut r = OpRegistry::new();
        register_builtin_ops(&mut r);
        r
    })
}

/// Populate the table with every built-in op module (wave-1: elementwise + math).
fn register_builtin_ops(registry: &mut OpRegistry) {
    crate::ops::elementwise::register(registry);
    crate::ops::math::register(registry);
}

/// Run-time dispatch: find the handler for a node and translate it.
pub fn translate(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let handler = registry()
        .find_entry(&n.domain, &n.op_type, n.since_version)
        .map(|e| e.handler)
        .ok_or_else(|| {
            format!(
                "MLX: no translation for op {}::{}",
                if n.domain.is_empty() { "ai.onnx" } else { &n.domain },
                n.op_type
            )
        })?;
    handler(ctx, n)
}

/// Claim-time node predicate consulted from GetCapability. True iff the registry has a matching
/// (domain, op, opset) entry AND that entry's claim predicate accepts this concrete node.
pub fn claimable(node: &NodeView) -> bool {
    match registry().find_entry(&node.domain(), &node.op_type(), node.since_version()) {
        Some(entry) => (entry.claim)(node),
        None => false,
    }
}

// ---- Claim-time node view -----------------------------------------------------------------------

/// A light read-only view over an `OrtNode` used by claim predicates (mirrors `Ort::ConstNode` plus
/// the `op_claim.h` helpers). All FFI is confined here.
pub struct NodeView {
    api: *const ort::OrtApi,
    node: *const ort::OrtNode,
}

/// Tensor element type + shape of a node value slot; `None` for an omitted optional / non-tensor.
pub struct SlotInfo {
    pub dtype: ort::ONNXTensorElementDataType,
    pub shape: Vec<i64>,
}

impl NodeView {
    pub fn new(api: *const ort::OrtApi, node: *const ort::OrtNode) -> Self {
        NodeView { api, node }
    }

    fn api(&self) -> &ort::OrtApi {
        unsafe { &*self.api }
    }

    fn cstr(&self, p: *const c_char) -> String {
        if p.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
        }
    }

    pub fn op_type(&self) -> String {
        unsafe {
            let mut p: *const c_char = std::ptr::null();
            (self.api().Node_GetOperatorType.unwrap())(self.node, &mut p);
            self.cstr(p)
        }
    }

    pub fn domain(&self) -> String {
        unsafe {
            let mut p: *const c_char = std::ptr::null();
            (self.api().Node_GetDomain.unwrap())(self.node, &mut p);
            self.cstr(p)
        }
    }

    pub fn since_version(&self) -> i32 {
        unsafe {
            let mut v: i32 = 0;
            (self.api().Node_GetSinceVersion.unwrap())(self.node, &mut v);
            v
        }
    }

    pub fn num_inputs(&self) -> usize {
        unsafe {
            let mut n: usize = 0;
            (self.api().Node_GetNumInputs.unwrap())(self.node, &mut n);
            n
        }
    }

    pub fn num_outputs(&self) -> usize {
        unsafe {
            let mut n: usize = 0;
            (self.api().Node_GetNumOutputs.unwrap())(self.node, &mut n);
            n
        }
    }

    fn inputs_raw(&self) -> Vec<*const ort::OrtValueInfo> {
        let n = self.num_inputs();
        let mut v: Vec<*const ort::OrtValueInfo> = vec![std::ptr::null(); n];
        if n > 0 {
            unsafe { (self.api().Node_GetInputs.unwrap())(self.node, v.as_mut_ptr(), n) };
        }
        v
    }

    fn outputs_raw(&self) -> Vec<*const ort::OrtValueInfo> {
        let n = self.num_outputs();
        let mut v: Vec<*const ort::OrtValueInfo> = vec![std::ptr::null(); n];
        if n > 0 {
            unsafe { (self.api().Node_GetOutputs.unwrap())(self.node, v.as_mut_ptr(), n) };
        }
        v
    }

    fn slot_info(&self, vi: *const ort::OrtValueInfo) -> Option<SlotInfo> {
        if vi.is_null() {
            return None;
        }
        unsafe {
            let api = self.api();
            let mut ti: *const ort::OrtTypeInfo = std::ptr::null();
            let st = (api.GetValueInfoTypeInfo.unwrap())(vi, &mut ti);
            if !st.is_null() || ti.is_null() {
                return None;
            }
            let mut onnx_type: ort::ONNXType = 0;
            (api.GetOnnxTypeFromTypeInfo.unwrap())(ti, &mut onnx_type);
            if onnx_type != ort::ONNXType_ONNX_TYPE_TENSOR {
                return None;
            }
            let mut tsi: *const ort::OrtTensorTypeAndShapeInfo = std::ptr::null();
            (api.CastTypeInfoToTensorInfo.unwrap())(ti, &mut tsi);
            if tsi.is_null() {
                return None;
            }
            let mut dtype: ort::ONNXTensorElementDataType = 0;
            (api.GetTensorElementType.unwrap())(tsi, &mut dtype);
            let mut nd: usize = 0;
            (api.GetDimensionsCount.unwrap())(tsi, &mut nd);
            let mut dims = vec![0i64; nd];
            if nd > 0 {
                (api.GetDimensions.unwrap())(tsi, dims.as_mut_ptr(), nd);
            }
            Some(SlotInfo { dtype, shape: dims })
        }
    }

    /// Element type + shape of input `i` (None if omitted / non-tensor).
    pub fn input_info(&self, i: usize) -> Option<SlotInfo> {
        let ins = self.inputs_raw();
        ins.get(i).and_then(|&vi| self.slot_info(vi))
    }

    /// Element type + shape of output `i` (None if omitted / non-tensor).
    pub fn output_info(&self, i: usize) -> Option<SlotInfo> {
        let outs = self.outputs_raw();
        outs.get(i).and_then(|&vi| self.slot_info(vi))
    }

    /// Read a scalar INT attribute by name, or `default` when absent / of another type.
    pub fn int_attr(&self, name: &str, default: i64) -> i64 {
        unsafe {
            let api = self.api();
            let cname = match std::ffi::CString::new(name) {
                Ok(c) => c,
                Err(_) => return default,
            };
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st =
                (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() || attr.is_null() {
                return default;
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            if atype != ort::OrtOpAttrType_ORT_OP_ATTR_INT {
                return default;
            }
            let mut value: i64 = default;
            let mut out_len: usize = 0;
            let st = (api.ReadOpAttr.unwrap())(
                attr,
                ort::OrtOpAttrType_ORT_OP_ATTR_INT,
                &mut value as *mut i64 as *mut std::os::raw::c_void,
                std::mem::size_of::<i64>(),
                &mut out_len,
            );
            if !st.is_null() {
                return default;
            }
            value
        }
    }
}

// ---- Shared claim helpers (port of op_claim.h) --------------------------------------------------

use ort::*;

/// Float dtypes the dtype-generic MLX paths handle: fp32, fp16, bf16.
pub fn is_mlx_float(t: ort::ONNXTensorElementDataType) -> bool {
    t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
}

pub fn is_signed_integer(t: ort::ONNXTensorElementDataType) -> bool {
    t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
}

/// Strict elementwise-or-trailing-suffix broadcast (rejects mismatched non-suffix shapes). A scalar
/// operand is allowed only via `scalar_or_suffix`.
pub fn suffix_broadcast(a: &[i64], b: &[i64]) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    // The longer shape's trailing dims must match the shorter shape's dims (numpy suffix rule),
    // requiring equal or 1 on the broadcast side.
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let off = long.len() - short.len();
    for i in 0..short.len() {
        let l = long[off + i];
        let s = short[i];
        if l != s && s != 1 && l != 1 {
            return false;
        }
    }
    true
}

/// Lenient variant that also accepts a genuine scalar operand (empty shape).
pub fn scalar_or_suffix_broadcast(a: &[i64], b: &[i64]) -> bool {
    if a.is_empty() || b.is_empty() {
        return true;
    }
    suffix_broadcast(a, b)
}
