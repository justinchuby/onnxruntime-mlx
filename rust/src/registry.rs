//! The (domain, op_type, [min,max] opset) -> { handler, claim predicate } registry — the single
//! source of truth for which ops the MLX EP can translate. Both the claim-time membership check
//! (GetCapability) and the run-time translator dispatch through the SAME table, so "claimed" and
//! "translatable" can never disagree (faithful port of `op_registry.{h,cc}`).

use std::borrow::Cow;
use std::os::raw::c_char;
use std::sync::LazyLock;

use half::{bf16, f16};

use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::sys::ort;

/// A translation handler: reads a NodeDesc, emits MLX ops through the context, binds the outputs.
pub type OpHandler = fn(&mut TranslationContext, &NodeDesc) -> Result<(), MlxError>;

/// The outcome of a claim predicate: `Ok(())` claims the node; `Err(reason)` declines it and carries
/// a colocated, human-readable explanation of WHY (surfaced verbatim by the tracer's "claiming view"
/// and by `ONNXRUNTIME_EP_MLX_CLAIM_DEBUG`). Because the reason travels WITH the decision, every decline is
/// guaranteed to have one and it can never drift out of sync with the predicate's actual logic —
/// there is no separate reason table to maintain.
pub type ClaimResult = Result<(), Cow<'static, str>>;

/// A claim-time predicate: given the concrete ONNX node, decide whether MLX can translate it exactly
/// (dtypes / shapes / attributes / input form) and, when it cannot, say why. The (domain, op_type,
/// opset) key is matched first.
pub type ClaimPredicate = fn(&NodeView) -> ClaimResult;

/// The strongest compiled shape mode an op lowering may enter for one concrete node.
///
/// `Shapeless` is an affirmative promise that the handler's shapeless trace emits only MLX
/// primitives implementing `Primitive::output_shapes`. `ShapeKeyedOnly` carries the exact reason
/// that promise cannot be made while still allowing a shape-keyed trace. `EagerOnly` is stricter:
/// no compiled route currently preserves that concrete form's semantics (and unsupported forms
/// use it conservatively too). Reasons stay at the registration boundary so partitioning and every
/// compiled execution route share one authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileShapeSafety {
    Shapeless,
    ShapeKeyedOnly { reason: &'static str },
    EagerOnly { reason: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompilePartitionClass {
    Shapeless,
    ShapeKeyed,
    Eager,
}

pub const MLX_PAD_SHAPE_REASON: &str =
    "emits MLX Pad, which lacks Primitive::output_shapes in MLX 0.32.2";
pub const MLX_RANDOM_BITS_SHAPE_REASON: &str =
    "emits MLX RandomBits, which lacks Primitive::output_shapes in MLX 0.32.2";
pub const MLX_SCAN_SHAPE_REASON: &str =
    "emits MLX Scan, which lacks Primitive::output_shapes in MLX 0.32.2";
pub const MLX_SCATTER_SHAPE_REASON: &str =
    "emits MLX Scatter, which lacks Primitive::output_shapes in MLX 0.32.2";
pub const MLX_SLICE_SHAPE_REASON: &str =
    "emits MLX Slice, which lacks Primitive::output_shapes in MLX 0.32.2";
pub const MLX_SPLIT_SHAPE_REASON: &str =
    "emits MLX Split, which lacks Primitive::output_shapes in MLX 0.32.2";
pub const MLX_VIEW_SHAPE_REASON: &str =
    "emits MLX View, which lacks Primitive::output_shapes in MLX 0.32.2";

impl CompileShapeSafety {
    pub const fn allows_shapeless(self) -> bool {
        matches!(self, Self::Shapeless)
    }

    pub const fn allows_shape_keyed(self) -> bool {
        !matches!(self, Self::EagerOnly { .. })
    }

    pub const fn partition_class(self) -> CompilePartitionClass {
        match self {
            Self::Shapeless => CompilePartitionClass::Shapeless,
            Self::ShapeKeyedOnly { .. } => CompilePartitionClass::ShapeKeyed,
            Self::EagerOnly { .. } => CompilePartitionClass::Eager,
        }
    }
}

/// Per-node classifier for lowerings whose primitive emission depends on attributes or static
/// input geometry. This preserves shapeless decode for safe forms without letting an unsafe branch
/// silently inherit a permissive default.
pub type CompileShapeClassifier = fn(&NodeView) -> CompileShapeSafety;

/// Decline the current claim predicate with a colocated reason (`format!`-style args, including
/// inline captures like `deny!("rank {rank} unsupported")`). Use inside a `-> ClaimResult` function.
/// The reason string is only built on the decline path, so its allocation never touches a claim
/// that succeeds.
#[macro_export]
macro_rules! deny {
    ($($fmt:tt)+) => {
        return ::core::result::Result::Err(::std::borrow::Cow::Owned(format!($($fmt)+)))
    };
}

/// Guard a claim requirement: if `$cond` is false, decline with the given reason (`format!`-style,
/// inline captures allowed). `require!(rank <= 4, "rank {rank} > 4 unsupported")`.
#[macro_export]
macro_rules! require {
    ($cond:expr, $($fmt:tt)+) => {
        // Bind to a bool first so the macro negates `!ok` (a bool), not `!(a < b)` directly — the
        // latter trips clippy::neg_cmp_op_on_partial_ord in every caller that passes a float
        // comparison. (An `#[allow]` on the `if` would be an unstable expression attribute, E0658.)
        {
            let ok: bool = $cond;
            if !ok {
                return ::core::result::Result::Err(::std::borrow::Cow::Owned(format!($($fmt)+)));
            }
        }
    };
}

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
    table: Vec<RegisteredOp>,
}

enum CompileShapeRule {
    Always(CompileShapeSafety),
    Classified(CompileShapeClassifier),
}

struct RegisteredOp {
    entry: OpRegistration,
    compile_shape_rule: CompileShapeRule,
}

impl OpRegistry {
    fn new() -> Self {
        OpRegistry { table: Vec::new() }
    }

    /// Register a lowering that deliberately opts in to shapeless compilation.
    pub fn register_shapeless(&mut self, entry: OpRegistration) {
        self.table.push(RegisteredOp {
            entry,
            compile_shape_rule: CompileShapeRule::Always(CompileShapeSafety::Shapeless),
        });
    }

    /// Register a lowering that always requires a shape-keyed trace.
    pub fn register_shape_keyed(&mut self, entry: OpRegistration, reason: &'static str) {
        assert!(
            !reason.is_empty(),
            "shape-keyed registration for {}::{} requires a reason",
            entry.domain,
            entry.op_type
        );
        self.table.push(RegisteredOp {
            entry,
            compile_shape_rule: CompileShapeRule::Always(CompileShapeSafety::ShapeKeyedOnly {
                reason,
            }),
        });
    }

    /// Register a lowering that must execute eagerly because it reads runtime host state.
    pub fn register_eager(&mut self, entry: OpRegistration, reason: &'static str) {
        assert!(
            !reason.is_empty(),
            "eager registration for {}::{} requires a reason",
            entry.domain,
            entry.op_type
        );
        self.table.push(RegisteredOp {
            entry,
            compile_shape_rule: CompileShapeRule::Always(CompileShapeSafety::EagerOnly { reason }),
        });
    }

    /// Register a lowering whose concrete node form decides whether shapeless compilation is safe.
    pub fn register_shape_classified(
        &mut self,
        entry: OpRegistration,
        classifier: CompileShapeClassifier,
    ) {
        self.table.push(RegisteredOp {
            entry,
            compile_shape_rule: CompileShapeRule::Classified(classifier),
        });
    }

    fn find_entry_index(&self, domain: &str, op_type: &str, since_version: i32) -> Option<usize> {
        self.table.iter().position(|registered| {
            let e = &registered.entry;
            e.domain == domain
                && e.op_type == op_type
                && (e.min_opset == K_ANY_OPSET || since_version >= e.min_opset)
                && (e.max_opset == K_ANY_OPSET || since_version <= e.max_opset)
        })
    }

    /// The matching entry for (domain, op_type, since_version), or None.
    pub fn find_entry(
        &self,
        domain: &str,
        op_type: &str,
        since_version: i32,
    ) -> Option<&OpRegistration> {
        self.find_entry_index(domain, op_type, since_version)
            .map(|index| &self.table[index].entry)
    }

    fn compile_shape_safety(&self, node: &NodeView) -> CompileShapeSafety {
        let domain = node.domain();
        let domain = if domain == "ai.onnx" { "" } else { &domain };
        let op_type = node.op_type();
        let Some(index) = self.find_entry_index(domain, &op_type, node.since_version()) else {
            return CompileShapeSafety::ShapeKeyedOnly {
                reason: "no matching registered lowering; shapeless compile denied conservatively",
            };
        };
        let safety = match self.table[index].compile_shape_rule {
            CompileShapeRule::Always(safety) => safety,
            CompileShapeRule::Classified(classifier) => classifier(node),
        };
        if let CompileShapeSafety::ShapeKeyedOnly { reason }
        | CompileShapeSafety::EagerOnly { reason } = safety
        {
            assert!(
                !reason.is_empty(),
                "shape classifier for {}::{} returned an empty reason",
                domain,
                op_type
            );
        }
        safety
    }
}

static REGISTRY: LazyLock<OpRegistry> = LazyLock::new(|| {
    let mut r = OpRegistry::new();
    register_builtin_ops(&mut r);
    r
});

fn registry() -> &'static OpRegistry {
    &REGISTRY
}

/// Compile-shape capability declared by the registered lowering for one concrete node.
pub fn compile_shape_safety(node: &NodeView) -> CompileShapeSafety {
    registry().compile_shape_safety(node)
}

/// Populate the table with every built-in op module (wave-1: elementwise + math).
fn register_builtin_ops(registry: &mut OpRegistry) {
    crate::ops::elementwise::register(registry);
    crate::ops::image::register(registry);
    crate::ops::math::register(registry);
    crate::ops::reduction::register(registry);
    crate::ops::shape::register(registry);
    crate::ops::matmul::register(registry);
    // signal/random/recurrent/ssm/misc/controlflow
    crate::ops::signal::register(registry);
    crate::ops::random::register(registry);
    crate::ops::recurrent::register(registry);
    crate::ops::ssm::register(registry);
    crate::ops::misc::register(registry);
    crate::ops::controlflow::register(registry);
    // norm+attention
    crate::ops::norm::register_norm(registry);
    crate::ops::onnx_ml_linear::register(registry);
    crate::ops::onnx_ml_preprocess::register(registry);
    crate::ops::attention::register_attention(registry);
    // conv+vision
    crate::ops::conv::register_conv(registry);
    crate::ops::vision::register_vision(registry);
    crate::ops::quant::register(registry); // quant
    crate::ops::stragglers::register(registry); // stragglers
}

/// Run-time dispatch: find the handler for a node and translate it.
pub fn translate(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let handler = registry()
        .find_entry(&n.domain, &n.op_type, n.since_version)
        .map(|e| e.handler)
        .ok_or_else(|| {
            format!(
                "MLX: no translation for op {}::{}",
                if n.domain.is_empty() {
                    "ai.onnx"
                } else {
                    &n.domain
                },
                n.op_type
            )
        })?;
    // Bracket the handler so it can declare (via ctx.mark_fast/mark_composed) which path it took;
    // the tracer then surfaces composed (fallback) paths prominently. Near-zero cost when tracing
    // is off: `op_timer_start`/`record_op_path` early-return on the atomic enable flag.
    let tr = crate::trace::tracer();
    let start = tr.op_timer_start();
    ctx.reset_path_mark();
    let r = handler(ctx, n);
    if r.is_ok() {
        let mark = ctx.take_path_mark();
        tr.record_op_path(n, start, mark);
        // Per-op detail span (rich Args always; fine mode also times a per-op eval).
        ctx.trace_node(&n.op_type, n, start);
    }
    r
}

/// The claim decision for `node`, WITH its reason on decline. `Ok(())` = MLX can translate it
/// exactly; `Err(reason)` = it falls back to CPU, and `reason` explains why (either "no registry
/// entry for this (domain, op, opset)" or the colocated message the matching claim predicate
/// returned). This is the single source of truth: `claimable` is just `.is_ok()`, and the tracer /
/// `ONNXRUNTIME_EP_MLX_CLAIM_DEBUG` surface the `Err` string directly — no separate reason table to drift.
pub fn claim_decision(node: &NodeView) -> ClaimResult {
    match registry().find_entry(&node.domain(), &node.op_type(), node.since_version()) {
        Some(entry) => (entry.claim)(node),
        None => {
            let op = node.op_type();
            let dom = node.domain();
            let where_ = if dom.is_empty() {
                String::new()
            } else {
                format!(" (domain {dom})")
            };
            Err(Cow::Owned(format!(
                "no MLX handler for {op}{where_} at opset {} — add a claim+handler in rust/src/ops/ and register it",
                node.since_version()
            )))
        }
    }
}

/// A node's op type, qualified by its domain when it has one.
///
/// `Attention`, `MatMulNBits`, and others exist in both the default ONNX domain
/// and `com.microsoft` with different contracts, so a diagnostic that prints the
/// bare op type cannot say which one it declined — and grouping counts by that
/// bare name silently merges two different ops into one line.
///
/// The default domain (empty, or the explicit `ai.onnx`) stays unqualified,
/// because writing `ai.onnx.Softmax` would be noise on the common case.
pub fn qualified_op_name(node: &NodeView) -> String {
    let op = node.op_type();
    let domain = node.domain();
    if domain.is_empty() || domain == "ai.onnx" {
        op
    } else {
        format!("{domain}.{op}")
    }
}

/// Claim-time node predicate consulted from GetCapability. True iff `claim_decision` succeeds.
pub fn claimable(node: &NodeView) -> bool {
    claim_decision(node).is_ok()
}

/// A short ORT element-type name for diagnostics (usable by claim predicates when building reasons).
pub fn ort_dtype_name(t: ort::ONNXTensorElementDataType) -> &'static str {
    #[allow(non_upper_case_globals)]
    match t {
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => "fp32",
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => "fp16",
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => "bf16",
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => "fp64",
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => "int8",
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => "int16",
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => "int32",
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => "int64",
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => "uint8",
        ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => "bool",
        _ => "other",
    }
}

// ---- Claim-time node view -----------------------------------------------------------------------

/// A light read-only view over an `OrtNode` used by claim predicates (mirrors `Ort::ConstNode` plus
/// the `op_claim.h` helpers). All FFI is confined here.
pub struct NodeView {
    api: *const ort::OrtApi,
    node: *const ort::OrtNode,
}

/// Tensor element type + shape of a node value slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotInfo {
    pub dtype: ort::ONNXTensorElementDataType,
    pub shape: Vec<i64>,
}

/// ONNX value kind exposed by the plugin-EP graph API.
///
/// `KernelContext_GetOutput` only allocates tensors, so container-producing nodes must never be
/// claimed at an EP boundary. The contained kind is retained to make those refusals precise and to
/// allow consumers of graph-input optionals to validate their actual tensor payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueKind {
    Tensor(SlotInfo),
    Sequence(Box<ValueKind>),
    Optional(Box<ValueKind>),
    Other,
}

impl ValueKind {
    pub fn description(&self) -> &'static str {
        match self {
            Self::Tensor(_) => "tensor",
            Self::Sequence(_) => "sequence",
            Self::Optional(_) => "optional",
            Self::Other => "non-tensor container",
        }
    }

    pub fn tensor(&self) -> Option<&SlotInfo> {
        match self {
            Self::Tensor(info) => Some(info),
            _ => None,
        }
    }

    pub fn optional_tensor(&self) -> Option<&SlotInfo> {
        match self {
            Self::Optional(inner) => inner.tensor(),
            _ => None,
        }
    }
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

    /// The node's graph name (e.g. `/model.22/dfl/Softmax`), for locating a specific node.
    pub fn name(&self) -> String {
        unsafe {
            let mut p: *const c_char = std::ptr::null();
            (self.api().Node_GetName.unwrap())(self.node, &mut p);
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

    unsafe fn type_info_kind(&self, ti: *const ort::OrtTypeInfo) -> ValueKind {
        unsafe {
            if ti.is_null() {
                return ValueKind::Other;
            }
            let api = self.api();
            let mut onnx_type: ort::ONNXType = 0;
            let st = (api.GetOnnxTypeFromTypeInfo.unwrap())(ti, &mut onnx_type);
            if !st.is_null() {
                self.release_status(st);
                return ValueKind::Other;
            }
            match onnx_type {
                t if t == ort::ONNXType_ONNX_TYPE_TENSOR => {
                    let mut tsi: *const ort::OrtTensorTypeAndShapeInfo = std::ptr::null();
                    let st = (api.CastTypeInfoToTensorInfo.unwrap())(ti, &mut tsi);
                    if !st.is_null() || tsi.is_null() {
                        self.release_status(st);
                        return ValueKind::Other;
                    }
                    let mut dtype: ort::ONNXTensorElementDataType = 0;
                    let st = (api.GetTensorElementType.unwrap())(tsi, &mut dtype);
                    if !st.is_null() {
                        self.release_status(st);
                        return ValueKind::Other;
                    }
                    let mut nd: usize = 0;
                    let st = (api.GetDimensionsCount.unwrap())(tsi, &mut nd);
                    if !st.is_null() {
                        self.release_status(st);
                        return ValueKind::Other;
                    }
                    let mut shape = vec![0i64; nd];
                    if nd > 0 {
                        let st = (api.GetDimensions.unwrap())(tsi, shape.as_mut_ptr(), nd);
                        if !st.is_null() {
                            self.release_status(st);
                            return ValueKind::Other;
                        }
                    }
                    ValueKind::Tensor(SlotInfo { dtype, shape })
                }
                t if t == ort::ONNXType_ONNX_TYPE_SEQUENCE => {
                    let mut sequence: *const ort::OrtSequenceTypeInfo = std::ptr::null();
                    let st = (api.CastTypeInfoToSequenceTypeInfo.unwrap())(ti, &mut sequence);
                    if !st.is_null() || sequence.is_null() {
                        self.release_status(st);
                        return ValueKind::Other;
                    }
                    let mut element: *mut ort::OrtTypeInfo = std::ptr::null_mut();
                    let st = (api.GetSequenceElementType.unwrap())(sequence, &mut element);
                    if !st.is_null() || element.is_null() {
                        self.release_status(st);
                        return ValueKind::Other;
                    }
                    let kind = self.type_info_kind(element);
                    (api.ReleaseTypeInfo.unwrap())(element);
                    ValueKind::Sequence(Box::new(kind))
                }
                t if t == ort::ONNXType_ONNX_TYPE_OPTIONAL => {
                    let mut optional: *const ort::OrtOptionalTypeInfo = std::ptr::null();
                    let st = (api.CastTypeInfoToOptionalTypeInfo.unwrap())(ti, &mut optional);
                    if !st.is_null() || optional.is_null() {
                        self.release_status(st);
                        return ValueKind::Other;
                    }
                    let mut contained: *mut ort::OrtTypeInfo = std::ptr::null_mut();
                    let st = (api.GetOptionalContainedTypeInfo.unwrap())(optional, &mut contained);
                    if !st.is_null() || contained.is_null() {
                        self.release_status(st);
                        return ValueKind::Other;
                    }
                    let kind = self.type_info_kind(contained);
                    (api.ReleaseTypeInfo.unwrap())(contained);
                    ValueKind::Optional(Box::new(kind))
                }
                _ => ValueKind::Other,
            }
        }
    }

    fn value_kind(&self, vi: *const ort::OrtValueInfo) -> Option<ValueKind> {
        if vi.is_null() {
            return None;
        }
        unsafe {
            let api = self.api();
            let mut ti: *const ort::OrtTypeInfo = std::ptr::null();
            let st = (api.GetValueInfoTypeInfo.unwrap())(vi, &mut ti);
            if !st.is_null() || ti.is_null() {
                self.release_status(st);
                return None;
            }
            Some(self.type_info_kind(ti))
        }
    }

    /// Element type + shape of input `i` (None if omitted / non-tensor).
    pub fn input_info(&self, i: usize) -> Option<SlotInfo> {
        let ins = self.inputs_raw();
        ins.get(i)
            .and_then(|&vi| self.value_kind(vi))
            .and_then(|kind| kind.tensor().cloned())
    }

    /// Element type + shape of output `i` (None if omitted / non-tensor).
    pub fn output_info(&self, i: usize) -> Option<SlotInfo> {
        let outs = self.outputs_raw();
        outs.get(i)
            .and_then(|&vi| self.value_kind(vi))
            .and_then(|kind| kind.tensor().cloned())
    }

    /// Full ONNX kind of input `i`, including sequence and optional containment.
    pub fn input_kind(&self, i: usize) -> Option<ValueKind> {
        let ins = self.inputs_raw();
        ins.get(i).and_then(|&vi| self.value_kind(vi))
    }

    /// Full ONNX kind of output `i`, including sequence and optional containment.
    pub fn output_kind(&self, i: usize) -> Option<ValueKind> {
        let outs = self.outputs_raw();
        outs.get(i).and_then(|&vi| self.value_kind(vi))
    }

    /// Release a non-null `OrtStatus` returned on an error/not-found path (the OrtApi allocates a
    /// status object even for "attribute not found", which the caller owns and must free).
    #[inline]
    unsafe fn release_status(&self, st: *mut ort::OrtStatus) {
        if !st.is_null() {
            unsafe { (self.api().ReleaseStatus.unwrap())(st) };
        }
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
            let st = (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return default;
            }
            if attr.is_null() {
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
                self.release_status(st);
                return default;
            }
            value
        }
    }

    /// Read a scalar FLOAT attribute by name, or `default` when absent / of another type
    /// (mirrors `FloatAttribute`).
    pub fn float_attr(&self, name: &str, default: f32) -> f32 {
        unsafe {
            let api = self.api();
            let cname = match std::ffi::CString::new(name) {
                Ok(c) => c,
                Err(_) => return default,
            };
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st = (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return default;
            }
            if attr.is_null() {
                return default;
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            if atype != ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT {
                return default;
            }
            let mut value: f32 = default;
            let mut out_len: usize = 0;
            let st = (api.ReadOpAttr.unwrap())(
                attr,
                ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT,
                &mut value as *mut f32 as *mut std::os::raw::c_void,
                std::mem::size_of::<f32>(),
                &mut out_len,
            );
            if !st.is_null() {
                self.release_status(st);
                return default;
            }
            value
        }
    }

    /// True iff output `i` is present (declared, non-null value info with a non-empty name).
    pub fn output_present(&self, i: usize) -> bool {
        let outs = self.outputs_raw();
        match outs.get(i) {
            Some(&vi) if !vi.is_null() => {
                let mut p: *const c_char = std::ptr::null();
                unsafe { (self.api().GetValueInfoName.unwrap())(vi, &mut p) };
                !p.is_null() && unsafe { !std::ffi::CStr::from_ptr(p).to_bytes().is_empty() }
            }
            _ => false,
        }
    }

    /// Read an INTS attribute. Returns `(present, values)`: `present` is whether the node carries a
    /// genuine INTS attribute of that name (mirrors `IntsAttribute`).
    pub fn ints_attr(&self, name: &str) -> (bool, Vec<i64>) {
        unsafe {
            let api = self.api();
            let cname = match std::ffi::CString::new(name) {
                Ok(c) => c,
                Err(_) => return (false, Vec::new()),
            };
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st = (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return (false, Vec::new());
            }
            if attr.is_null() {
                return (false, Vec::new());
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            if atype != ort::OrtOpAttrType_ORT_OP_ATTR_INTS {
                return (false, Vec::new());
            }
            let read = api.ReadOpAttr.unwrap();
            let mut needed: usize = 0;
            let st0 = read(attr, atype, std::ptr::null_mut(), 0, &mut needed);
            self.release_status(st0);
            if needed == 0 {
                return (true, Vec::new());
            }
            let count = needed / std::mem::size_of::<i64>();
            let mut buf = vec![0i64; count];
            let mut out: usize = 0;
            let st = read(
                attr,
                atype,
                buf.as_mut_ptr() as *mut std::os::raw::c_void,
                needed,
                &mut out,
            );
            if st.is_null() {
                (true, buf)
            } else {
                self.release_status(st);
                (false, Vec::new())
            }
        }
    }

    /// Read a STRING attribute, or `default` when absent / of another type.
    pub fn string_attr(&self, name: &str, default: &str) -> String {
        unsafe {
            let api = self.api();
            let cname = match std::ffi::CString::new(name) {
                Ok(c) => c,
                Err(_) => return default.to_string(),
            };
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st = (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return default.to_string();
            }
            if attr.is_null() {
                return default.to_string();
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            if atype != ort::OrtOpAttrType_ORT_OP_ATTR_STRING {
                return default.to_string();
            }
            let read = api.ReadOpAttr.unwrap();
            let mut needed: usize = 0;
            let st0 = read(attr, atype, std::ptr::null_mut(), 0, &mut needed);
            self.release_status(st0);
            if needed == 0 {
                return String::new();
            }
            let mut buf = vec![0u8; needed];
            let mut out: usize = 0;
            let st = read(
                attr,
                atype,
                buf.as_mut_ptr() as *mut std::os::raw::c_void,
                needed,
                &mut out,
            );
            if !st.is_null() {
                self.release_status(st);
                return default.to_string();
            }
            buf.truncate(out.min(needed));
            String::from_utf8(buf).unwrap_or_else(|_| default.to_string())
        }
    }

    /// Read a scalar FLOAT attribute. Returns `None` when absent or of another type (so callers can
    /// distinguish "absent" from a genuine value, needed for optional-seed validation).
    pub fn float_attr_opt(&self, name: &str) -> Option<f32> {
        unsafe {
            let api = self.api();
            let cname = std::ffi::CString::new(name).ok()?;
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st = (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return None;
            }
            if attr.is_null() {
                return None;
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            if atype != ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT {
                return None;
            }
            let mut value: f32 = 0.0;
            let mut out_len: usize = 0;
            let st = (api.ReadOpAttr.unwrap())(
                attr,
                ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT,
                &mut value as *mut f32 as *mut std::os::raw::c_void,
                std::mem::size_of::<f32>(),
                &mut out_len,
            );
            if !st.is_null() {
                self.release_status(st);
                return None;
            }
            Some(value)
        }
    }

    /// Read a STRINGS attribute as a `Vec<String>` (null-separated buffer per the ORT ABI). Returns
    /// `None` when absent or of another type. Used to validate recurrent `activations` against the
    /// per-op defaults at claim time.
    pub fn strings_attr(&self, name: &str) -> Option<Vec<String>> {
        unsafe {
            let api = self.api();
            let cname = std::ffi::CString::new(name).ok()?;
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st = (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return None;
            }
            if attr.is_null() {
                return None;
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            if atype != ort::OrtOpAttrType_ORT_OP_ATTR_STRINGS {
                return None;
            }
            let read = api.ReadOpAttr.unwrap();
            let mut needed: usize = 0;
            let st0 = read(attr, atype, std::ptr::null_mut(), 0, &mut needed);
            self.release_status(st0);
            if needed == 0 {
                return Some(Vec::new());
            }
            let mut buf = vec![0u8; needed];
            let mut out: usize = 0;
            let st = read(
                attr,
                atype,
                buf.as_mut_ptr() as *mut std::os::raw::c_void,
                needed,
                &mut out,
            );
            if !st.is_null() {
                self.release_status(st);
                return None;
            }
            buf.truncate(out.min(needed));
            // Null-separated concatenation of C strings.
            let mut result = Vec::new();
            for part in buf.split(|&b| b == 0) {
                if !part.is_empty() {
                    result.push(String::from_utf8_lossy(part).into_owned());
                }
            }
            Some(result)
        }
    }

    /// True iff the node carries a genuine (non-UNDEFINED) attribute of `name`.
    pub fn has_attr(&self, name: &str) -> bool {
        unsafe {
            let api = self.api();
            let cname = match std::ffi::CString::new(name) {
                Ok(c) => c,
                Err(_) => return false,
            };
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st = (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return false;
            }
            if attr.is_null() {
                return false;
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            atype != ort::OrtOpAttrType_ORT_OP_ATTR_UNDEFINED
        }
    }

    /// Element dtype of a TENSOR-valued attribute, or `None` when `name` is absent or not a tensor.
    ///
    /// Lets a claim reject a fill value it could not rebuild in an MLX dtype before the node is
    /// taken, rather than discovering it during translation.
    pub fn attr_tensor_dtype(&self, name: &str) -> Option<ort::ONNXTensorElementDataType> {
        unsafe {
            let api = self.api();
            let cname = std::ffi::CString::new(name).ok()?;
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st = (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return None;
            }
            if attr.is_null() {
                return None;
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            if atype != ort::OrtOpAttrType_ORT_OP_ATTR_TENSOR {
                return None;
            }
            let mut value: *mut ort::OrtValue = std::ptr::null_mut();
            let st = (api.OpAttr_GetTensorAttributeAsOrtValue.unwrap())(attr, &mut value);
            if !st.is_null() {
                self.release_status(st);
                return None;
            }
            if value.is_null() {
                return None;
            }
            let mut info: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            let st = (api.GetTensorTypeAndShape.unwrap())(value, &mut info);
            if !st.is_null() {
                self.release_status(st);
                (api.ReleaseValue.unwrap())(value);
                return None;
            }
            if info.is_null() {
                (api.ReleaseValue.unwrap())(value);
                return None;
            }
            let mut etype: ort::ONNXTensorElementDataType = 0;
            let st = (api.GetTensorElementType.unwrap())(info, &mut etype);
            let ok = st.is_null();
            if !ok {
                self.release_status(st);
            }
            (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info);
            (api.ReleaseValue.unwrap())(value);
            ok.then_some(etype)
        }
    }

    /// The raw attribute type of `name` (ORT_OP_ATTR_UNDEFINED when absent). Lets a claim match a
    /// specific attribute form (e.g. Constant's value_int/value_float/value_ints/value_floats).
    pub fn attr_type(&self, name: &str) -> ort::OrtOpAttrType {
        unsafe {
            let api = self.api();
            let cname = match std::ffi::CString::new(name) {
                Ok(c) => c,
                Err(_) => return ort::OrtOpAttrType_ORT_OP_ATTR_UNDEFINED,
            };
            let mut attr: *const ort::OrtOpAttr = std::ptr::null();
            let st = (api.Node_GetAttributeByName.unwrap())(self.node, cname.as_ptr(), &mut attr);
            if !st.is_null() {
                self.release_status(st);
                return ort::OrtOpAttrType_ORT_OP_ATTR_UNDEFINED;
            }
            if attr.is_null() {
                return ort::OrtOpAttrType_ORT_OP_ATTR_UNDEFINED;
            }
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            atype
        }
    }

    /// The int64 value of a constant scalar (rank-0 or [1]) INT64 initializer input `i`, or None.
    /// Mirrors the C++ `IsConstScalarI64` used by OneHot/Trilu claim predicates.
    pub fn const_scalar_i64(&self, i: usize) -> Option<i64> {
        let info = self.input_info(i)?;
        if info.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 {
            return None;
        }
        if !(info.shape.is_empty() || info.shape == [1]) {
            return None;
        }
        if !self.is_constant_initializer(i) {
            return None;
        }
        let ins = self.inputs_raw();
        let vi = *ins.get(i)?;
        if vi.is_null() {
            return None;
        }
        unsafe {
            let api = self.api();
            let mut val: *const ort::OrtValue = std::ptr::null();
            let st = (api.ValueInfo_GetInitializerValue.unwrap())(vi, &mut val);
            if !st.is_null() || val.is_null() {
                if !st.is_null() {
                    self.release_status(st);
                }
                return None;
            }
            let mut data: *mut std::os::raw::c_void = std::ptr::null_mut();
            let st = (api.GetTensorMutableData.unwrap())(val as *mut ort::OrtValue, &mut data);
            if !st.is_null() || data.is_null() {
                if !st.is_null() {
                    self.release_status(st);
                }
                return None;
            }
            Some(*(data as *const i64))
        }
    }

    /// True iff input `i` is present (non-null value info with a non-empty name).
    pub fn input_present(&self, i: usize) -> bool {
        let ins = self.inputs_raw();
        match ins.get(i) {
            Some(&vi) if !vi.is_null() => {
                let mut p: *const c_char = std::ptr::null();
                unsafe { (self.api().GetValueInfoName.unwrap())(vi, &mut p) };
                !p.is_null() && unsafe { !std::ffi::CStr::from_ptr(p).to_bytes().is_empty() }
            }
            _ => false,
        }
    }

    /// True iff input `i` is a constant initializer (readable at translate time).
    pub fn is_constant_initializer(&self, i: usize) -> bool {
        let ins = self.inputs_raw();
        let vi = match ins.get(i) {
            Some(&vi) if !vi.is_null() => vi,
            _ => return false,
        };
        unsafe {
            let mut is_const = false;
            let st = (self.api().ValueInfo_IsConstantInitializer.unwrap())(vi, &mut is_const);
            if !st.is_null() {
                self.release_status(st);
                return false;
            }
            is_const
        }
    }

    /// True iff input `i` is a tensor(int64) constant initializer.
    pub fn is_const_int64(&self, i: usize) -> bool {
        matches!(self.input_info(i), Some(info)
            if info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            && self.is_constant_initializer(i)
    }

    /// True iff input `i` has no producer node — i.e. it is a graph input, an initializer, or an
    /// outer-scope value. Such values are readable host-side at translate time (`raw_host` handles
    /// the `Initializer` / `CtxInput` sources). An input WITH a producer is a runtime intermediate
    /// `raw_host` cannot read.
    pub fn input_is_host_readable(&self, i: usize) -> bool {
        let ins = self.inputs_raw();
        let vi = match ins.get(i) {
            Some(&vi) if !vi.is_null() => vi,
            _ => return false,
        };
        unsafe {
            let mut producer: *const ort::OrtNode = std::ptr::null();
            let st = (self.api().ValueInfo_GetValueProducer.unwrap())(
                vi,
                &mut producer,
                std::ptr::null_mut(),
            );
            if !st.is_null() {
                self.release_status(st);
                return false;
            }
            producer.is_null()
        }
    }

    /// Read the int64 values of a constant-initializer input `i` AT CLAIM TIME. Returns None when the
    /// input is not a readable int64 constant initializer (→ node left to CPU).
    pub fn read_const_int64(&self, i: usize) -> Option<Vec<i64>> {
        if !self.is_const_int64(i) {
            return None;
        }
        let ins = self.inputs_raw();
        let vi = *ins.get(i)?;
        unsafe {
            let api = self.api();
            let mut value: *const ort::OrtValue = std::ptr::null();
            let st = (api.ValueInfo_GetInitializerValue.unwrap())(vi, &mut value);
            if !st.is_null() {
                self.release_status(st);
                return None;
            }
            if value.is_null() {
                return None;
            }
            let mut info: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            (api.GetTensorTypeAndShape.unwrap())(value, &mut info);
            let mut count: usize = 0;
            (api.GetTensorShapeElementCount.unwrap())(info, &mut count);
            (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info);
            let mut data: *const std::os::raw::c_void = std::ptr::null();
            (api.GetTensorData.unwrap())(value, &mut data);
            if data.is_null() {
                return if count == 0 { Some(Vec::new()) } else { None };
            }
            Some(std::slice::from_raw_parts(data as *const i64, count).to_vec())
        }
    }

    pub fn read_const_f32(&self, i: usize) -> Option<Vec<f32>> {
        if !matches!(self.input_info(i), Some(info)
            if info.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            || !self.is_constant_initializer(i)
        {
            return None;
        }
        let ins = self.inputs_raw();
        let vi = *ins.get(i)?;
        unsafe {
            let api = self.api();
            let mut value: *const ort::OrtValue = std::ptr::null();
            let st = (api.ValueInfo_GetInitializerValue.unwrap())(vi, &mut value);
            if !st.is_null() {
                self.release_status(st);
                return None;
            }
            if value.is_null() {
                return None;
            }
            let mut info: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            (api.GetTensorTypeAndShape.unwrap())(value, &mut info);
            let mut count: usize = 0;
            (api.GetTensorShapeElementCount.unwrap())(info, &mut count);
            (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info);
            let mut data: *const std::os::raw::c_void = std::ptr::null();
            (api.GetTensorData.unwrap())(value, &mut data);
            if data.is_null() {
                return if count == 0 { Some(Vec::new()) } else { None };
            }
            Some(std::slice::from_raw_parts(data as *const f32, count).to_vec())
        }
    }

    /// Positional input names of the node (empty string for an omitted optional slot).
    pub fn input_names(&self) -> Vec<String> {
        self.inputs_raw()
            .iter()
            .map(|&vi| {
                if vi.is_null() {
                    return String::new();
                }
                let mut p: *const c_char = std::ptr::null();
                unsafe { (self.api().GetValueInfoName.unwrap())(vi, &mut p) };
                self.cstr(p)
            })
            .collect()
    }

    /// Positional output names of the node.
    pub fn output_names(&self) -> Vec<String> {
        self.outputs_raw()
            .iter()
            .map(|&vi| {
                if vi.is_null() {
                    return String::new();
                }
                let mut p: *const c_char = std::ptr::null();
                unsafe { (self.api().GetValueInfoName.unwrap())(vi, &mut p) };
                self.cstr(p)
            })
            .collect()
    }

    /// The node's body subgraphs (If/Scan/Loop) as `(attribute_name, GraphView)` pairs. Empty for an
    /// ordinary op. The returned graph handles are borrowed (tied to the node's lifetime).
    pub fn subgraphs(&self) -> Vec<(String, GraphView)> {
        unsafe {
            let api = self.api();
            let mut num: usize = 0;
            let st = (api.Node_GetNumSubgraphs.unwrap())(self.node, &mut num);
            if !st.is_null() {
                self.release_status(st);
                return Vec::new();
            }
            if num == 0 {
                return Vec::new();
            }
            let mut graphs: Vec<*const ort::OrtGraph> = vec![std::ptr::null(); num];
            let mut names: Vec<*const c_char> = vec![std::ptr::null(); num];
            let st = (api.Node_GetSubgraphs.unwrap())(
                self.node,
                graphs.as_mut_ptr(),
                num,
                names.as_mut_ptr(),
            );
            if !st.is_null() {
                self.release_status(st);
                return Vec::new();
            }
            (0..num)
                .map(|i| (self.cstr(names[i]), GraphView::new(self.api, graphs[i])))
                .collect()
        }
    }

    /// True iff input `i` is a tensor(int32|int64) constant initializer (a shape/size parameter the
    /// vision handlers read at translate time). Port of the C++ `IsConstIntTensor`.
    pub fn is_const_int_tensor(&self, i: usize) -> bool {
        matches!(self.input_info(i), Some(info) if is_int_index(info.dtype))
            && self.is_constant_initializer(i)
    }

    /// Read the int32/int64 values of a constant-initializer input `i` AT CLAIM TIME, widened to
    /// int64. Returns None when the input is not a readable int32/int64 constant initializer. Port of
    /// the C++ `ReadConstIntAtClaim`.
    pub fn read_const_ints_any(&self, i: usize) -> Option<Vec<i64>> {
        let dtype = self.input_info(i)?.dtype;
        if !self.is_const_int_tensor(i) {
            return None;
        }
        let ins = self.inputs_raw();
        let vi = *ins.get(i)?;
        unsafe {
            let api = self.api();
            let mut value: *const ort::OrtValue = std::ptr::null();
            let st = (api.ValueInfo_GetInitializerValue.unwrap())(vi, &mut value);
            if !st.is_null() {
                self.release_status(st);
                return None;
            }
            if value.is_null() {
                return None;
            }
            let mut info: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            (api.GetTensorTypeAndShape.unwrap())(value, &mut info);
            let mut count: usize = 0;
            (api.GetTensorShapeElementCount.unwrap())(info, &mut count);
            (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info);
            let mut data: *const std::os::raw::c_void = std::ptr::null();
            (api.GetTensorData.unwrap())(value, &mut data);
            if data.is_null() {
                return if count == 0 { Some(Vec::new()) } else { None };
            }
            if dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 {
                Some(std::slice::from_raw_parts(data as *const i64, count).to_vec())
            } else {
                Some(
                    std::slice::from_raw_parts(data as *const i32, count)
                        .iter()
                        .map(|&v| v as i64)
                        .collect(),
                )
            }
        }
    }

    /// Read a scalar constant initializer as f64 for Range claim-time shape validation.
    pub fn read_const_scalar_f64(&self, i: usize) -> Option<f64> {
        if !self.is_constant_initializer(i) {
            return None;
        }
        let dtype = self.input_info(i)?.dtype;
        let ins = self.inputs_raw();
        let vi = *ins.get(i)?;
        unsafe {
            let api = self.api();
            let mut value: *const ort::OrtValue = std::ptr::null();
            let st = (api.ValueInfo_GetInitializerValue.unwrap())(vi, &mut value);
            if !st.is_null() {
                self.release_status(st);
                return None;
            }
            if value.is_null() {
                return None;
            }
            let mut info: *mut ort::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            (api.GetTensorTypeAndShape.unwrap())(value, &mut info);
            let mut count: usize = 0;
            (api.GetTensorShapeElementCount.unwrap())(info, &mut count);
            (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info);
            if count != 1 {
                return None;
            }
            let mut data: *const std::os::raw::c_void = std::ptr::null();
            (api.GetTensorData.unwrap())(value, &mut data);
            if data.is_null() {
                return None;
            }
            match dtype {
                t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
                    Some(*(data as *const i16) as f64)
                }
                t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
                    Some(*(data as *const i32) as f64)
                }
                t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
                    Some(*(data as *const i64) as f64)
                }
                t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
                    Some(*(data as *const f32) as f64)
                }
                t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => {
                    Some(f16::from_bits(*(data as *const u16)).to_f64())
                }
                t if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => {
                    Some(bf16::from_bits(*(data as *const u16)).to_f64())
                }
                _ => None,
            }
        }
    }
}

/// A light read-only view over an `OrtGraph` (a control-flow body), used by control-flow claim
/// predicates to inspect body nodes/inputs/outputs. All FFI is confined here.
pub struct GraphView {
    api: *const ort::OrtApi,
    graph: *const ort::OrtGraph,
}

impl GraphView {
    pub fn new(api: *const ort::OrtApi, graph: *const ort::OrtGraph) -> Self {
        GraphView { api, graph }
    }

    fn api(&self) -> &ort::OrtApi {
        unsafe { &*self.api }
    }

    fn name(&self, p: *const c_char) -> String {
        if p.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
        }
    }

    /// The body's nodes as claim-time views.
    pub fn nodes(&self) -> Vec<NodeView> {
        unsafe {
            let api = self.api();
            let mut num: usize = 0;
            (api.Graph_GetNumNodes.unwrap())(self.graph, &mut num);
            if num == 0 {
                return Vec::new();
            }
            let mut nodes: Vec<*const ort::OrtNode> = vec![std::ptr::null(); num];
            (api.Graph_GetNodes.unwrap())(self.graph, nodes.as_mut_ptr(), num);
            nodes
                .into_iter()
                .map(|n| NodeView::new(self.api, n))
                .collect()
        }
    }

    fn value_names(
        &self,
        count_fn: unsafe extern "C" fn(*const ort::OrtGraph, *mut usize) -> *mut ort::OrtStatus,
        get_fn: unsafe extern "C" fn(
            *const ort::OrtGraph,
            *mut *const ort::OrtValueInfo,
            usize,
        ) -> *mut ort::OrtStatus,
    ) -> Vec<String> {
        unsafe {
            let mut num: usize = 0;
            count_fn(self.graph, &mut num);
            if num == 0 {
                return Vec::new();
            }
            let mut vis: Vec<*const ort::OrtValueInfo> = vec![std::ptr::null(); num];
            get_fn(self.graph, vis.as_mut_ptr(), num);
            vis.into_iter()
                .map(|vi| {
                    if vi.is_null() {
                        return String::new();
                    }
                    let mut p: *const c_char = std::ptr::null();
                    (self.api().GetValueInfoName.unwrap())(vi, &mut p);
                    self.name(p)
                })
                .collect()
        }
    }

    /// The body's formal input names.
    pub fn input_names(&self) -> Vec<String> {
        self.value_names(
            self.api().Graph_GetNumInputs.unwrap(),
            self.api().Graph_GetInputs.unwrap(),
        )
    }

    /// The body's formal output names.
    pub fn output_names(&self) -> Vec<String> {
        self.value_names(
            self.api().Graph_GetNumOutputs.unwrap(),
            self.api().Graph_GetOutputs.unwrap(),
        )
    }

    /// Every node in this body is MLX-translatable (recursively via the registry claim). A body with
    /// no nodes (e.g. a pure alias branch) is trivially claimable.
    pub fn all_nodes_claimable(&self) -> bool {
        self.first_unclaimable_node().is_none()
    }

    /// The first node in this graph the registry refuses, as `(op_type, name, reason)`.
    ///
    /// `all_nodes_claimable` only answers yes/no, which makes a rejected control-flow body
    /// opaque: the parent reports that *something* inside is unclaimable without saying what.
    /// Returning the offending node lets the caller name it in its own denial reason.
    pub fn first_unclaimable_node(&self) -> Option<(String, String, String)> {
        self.nodes().iter().find_map(|n| match claim_decision(n) {
            Ok(()) => None,
            Err(reason) => Some((n.op_type(), n.name(), reason.into_owned())),
        })
    }

    /// Any node anywhere in this body (including nested bodies) carries a float64 tensor.
    ///
    /// A control-flow node is claimed as a whole and its body is translated inside the parent's
    /// plan, so the body never gets a cluster — or a stream — of its own. GetCapability's fp64
    /// colouring only sees the control-flow node itself, so a float64 tensor buried in a body would
    /// otherwise ride along on the GPU stream and hard-error inside MLX. Callers use this to decline
    /// such bodies outright.
    pub fn body_uses_float64(&self) -> bool {
        self.nodes().iter().any(|n| {
            node_uses_float64(n)
                || n.subgraphs()
                    .iter()
                    .any(|(_, body)| body.body_uses_float64())
        })
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

/// float64 (ONNX DOUBLE).
pub fn is_float64(t: ort::ONNXTensorElementDataType) -> bool {
    t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
}

/// Float dtypes an op may carry when it opts IN to the fp64 CPU-stream path: fp32/fp16/bf16 plus
/// float64.
///
/// This is deliberately a *separate* predicate from [`is_mlx_float`], for two independent reasons.
///
/// 1. MLX rejects float64 on the Metal backend outright, so an fp64 node is only translatable on an
///    MLX CPU stream; widening `is_mlx_float` (used by ~everything) would admit fp64 into
///    GPU-stream subgraphs and hard-error at run time.
///
/// 2. **MLX's CPU float64 support is partial, and silently so.** Several primitives keep the
///    `float64` dtype on the result while computing in float32 and widening back — the array claims
///    to be double precision and is not. Measured on MLX 0.32.2 (see the `mlx_float64_primitives`
///    test, which re-checks this at build time):
///
///    | exact in fp64 | silently float32-accurate |
///    |---|---|
///    | `log` `sqrt` `rsqrt` `tanh` `expm1` `power` `abs` `neg` `sign` `floor` `ceil` `round` `reciprocal` | `exp` `sin` `cos` `erf` `sigmoid` `logaddexp` |
///    | `add` `subtract` `multiply` `divide` `maximum` `minimum` | |
///    | `sum` `prod` `mean` `matmul` | `logsumexp` `softmax` |
///
/// So an op opts in only when *every* primitive its handler emits is in the left column. Claiming a
/// right-column op would make the EP quietly return float32-accurate answers for a float64 model,
/// which is worse than ORT's honest `NOT_IMPLEMENTED`. That is why `Sigmoid`, `Erf`, `Gelu`, `Exp`,
/// `Softplus`, `Mish`, `Swish` and the trig family stay fp32-only while `Log`, `Sqrt`, `Tanh`,
/// `Relu`, `Elu`/`Selu`/`Celu` (expm1), `Clip`, `MatMul`, exact reductions and all data movement do
/// carry fp64.
pub fn is_mlx_cpu_float(t: ort::ONNXTensorElementDataType) -> bool {
    is_mlx_float(t) || is_float64(t)
}

/// True when any tensor slot of `node` (input or output) is float64.
///
/// Drives two things: the cluster-boundary rule in GetCapability (an fp64 node never merges with a
/// non-fp64 node) and the CPU-stream selection for the resulting plan.
pub fn node_uses_float64(node: &NodeView) -> bool {
    (0..node.num_inputs()).any(|i| node.input_info(i).is_some_and(|s| is_float64(s.dtype)))
        || (0..node.num_outputs()).any(|i| node.output_info(i).is_some_and(|s| is_float64(s.dtype)))
}

pub fn is_signed_integer(t: ort::ONNXTensorElementDataType) -> bool {
    t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
}

pub fn is_unsigned_integer(t: ort::ONNXTensorElementDataType) -> bool {
    t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
}

/// The most relaxed dtype set the MLX Metal backend can carry: bool, all int/uint widths (8-64),
/// and fp16/bf16/fp32. EXCLUDES float64/complex/string/fp8. Port of `IsMlxSupportedType`.
pub fn is_mlx_supported(t: ort::ONNXTensorElementDataType) -> bool {
    is_mlx_float(t)
        || is_signed_integer(t)
        || is_unsigned_integer(t)
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
}

/// Numeric (non-bool) MLX types: the reductions/argmin-max/cumsum dtype set.
pub fn is_mlx_numeric(t: ort::ONNXTensorElementDataType) -> bool {
    is_mlx_float(t) || is_signed_integer(t) || is_unsigned_integer(t)
}

/// Dtypes the pure data-movement ops carry end-to-end (every case CopyOut can memcpy). Port of
/// `IsMovableType`.
///
/// float64 is included: these ops never do arithmetic, `copy_out_raw_delta` sizes every memcpy from
/// the MLX array's own `itemsize`, and MLX's CPU backend carries fp64 through gather/reshape/concat
/// unchanged. GetCapability isolates the resulting fp64 nodes into their own cluster, which runs on
/// an MLX CPU stream.
pub fn is_movable(t: ort::ONNXTensorElementDataType) -> bool {
    is_mlx_float(t)
        || is_float64(t)
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
}

/// int32/int64 — the gather/scatter index dtype.
pub fn is_int_index(t: ort::ONNXTensorElementDataType) -> bool {
    t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
}

/// MLX-compatible Range element dtypes.
pub fn is_range_type(t: ort::ONNXTensorElementDataType) -> bool {
    t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
}

/// Strict elementwise-or-trailing-suffix broadcast (rejects mismatched non-suffix shapes). A scalar
/// operand is allowed only via `scalar_or_suffix`.
///
/// A negative extent is ORT's encoding for an unknown (symbolic) dimension, not a literal size, so
/// it is treated as compatible with anything. Comparing it as a value would reject ordinary
/// broadcasts such as `[-1, -1, 64]` against `[-1, -1, -1]` purely because the static shape is not
/// fully known. Concrete shapes are resolved when the closure is compiled shape-keyed, and a model
/// whose runtime extents genuinely do not broadcast would already have failed ORT shape inference.
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
        if l < 0 || s < 0 {
            continue;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_broadcast_treats_unknown_extents_as_compatible() {
        // ORT reports symbolic dims as -1; an unknown extent is not a literal
        // size and must not be compared as one.
        assert!(suffix_broadcast(&[-1, -1, 64], &[-1, -1, -1]));
        assert!(suffix_broadcast(&[-1, 256, 1], &[1, 256, 16]));
        assert!(suffix_broadcast(&[-1], &[8]));
        assert!(scalar_or_suffix_broadcast(&[-1, -1, 64], &[-1, -1, -1]));
    }

    #[test]
    fn float64_is_a_separate_opt_in_from_the_gpu_float_set() {
        use ort::*;
        const F64: ort::ONNXTensorElementDataType =
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        const F32: ort::ONNXTensorElementDataType =
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

        // `is_mlx_float` gates the GPU-stream paths and must NEVER admit float64: MLX rejects fp64
        // on Metal outright, so widening it would turn a claim into a run-time hard error.
        assert!(!is_mlx_float(F64));
        assert!(!is_mlx_supported(F64));
        assert!(!is_mlx_numeric(F64));

        // The opt-in predicate is the only float set that carries fp64.
        assert!(is_mlx_cpu_float(F64));
        assert!(is_mlx_cpu_float(F32));
        assert!(is_float64(F64));
        assert!(!is_float64(F32));

        // Data movement is dtype-agnostic (CopyOut sizes from the array's own itemsize).
        assert!(is_movable(F64));
    }

    #[test]
    fn suffix_broadcast_still_rejects_known_mismatches() {
        // Fully known extents that cannot broadcast are still refused, including
        // when an unknown dim sits alongside them.
        assert!(!suffix_broadcast(&[2, 3], &[2, 4]));
        assert!(!suffix_broadcast(&[-1, 3], &[-1, 4]));
        // The ordinary numpy suffix rules keep working.
        assert!(suffix_broadcast(&[2, 3], &[3]));
        assert!(suffix_broadcast(&[2, 3], &[1, 3]));
        assert!(!suffix_broadcast(&[2, 3], &[]));
        assert!(scalar_or_suffix_broadcast(&[2, 3], &[]));
    }

    fn compile_shape_rule(
        domain: &str,
        op_type: &str,
        since_version: i32,
    ) -> &'static CompileShapeRule {
        let index = registry()
            .find_entry_index(domain, op_type, since_version)
            .unwrap_or_else(|| panic!("missing registration for {domain}::{op_type}"));
        &registry().table[index].compile_shape_rule
    }

    #[test]
    fn every_shape_keyed_registration_carries_a_reason() {
        for registered in &registry().table {
            if let CompileShapeRule::Always(CompileShapeSafety::ShapeKeyedOnly { reason }) =
                registered.compile_shape_rule
            {
                assert!(
                    !reason.is_empty(),
                    "{}::{} omitted its shape-specialization reason",
                    registered.entry.domain,
                    registered.entry.op_type
                );
                assert!(
                    reason.contains("output_shapes") || reason.contains("conservatively"),
                    "{}::{} has an unactionable shape-specialization reason: {reason}",
                    registered.entry.domain,
                    registered.entry.op_type
                );
            }
        }
    }

    #[test]
    fn known_missing_output_shape_emitters_cannot_be_shapeless() {
        let always_shape_keyed = [
            ("com.microsoft", "PagedAttention", 1),
            ("com.microsoft", "Attention", 1),
            ("com.microsoft", "PackedMultiHeadAttention", 1),
            ("", "RotaryEmbedding", 23),
            ("com.microsoft", "RotaryEmbedding", 1),
            ("com.microsoft", "MRotaryEmbedding", 1),
            ("", "Scan", 24),
            ("", "DeformConv", 22),
            ("", "AveragePool", 24),
            ("", "MaxPool", 24),
            ("", "LpPool", 24),
            ("", "BitCast", 26),
            ("", "Scatter", 9),
            ("", "LRN", 24),
            ("ai.onnx.ml", "SVMClassifier", 1),
            ("ai.onnx.ml", "FeatureVectorizer", 1),
            ("com.microsoft", "QMoE", 1),
            ("", "RandomNormal", 1),
            ("", "RandomNormalLike", 1),
            ("", "RandomUniform", 1),
            ("", "RandomUniformLike", 1),
            ("", "Bernoulli", 15),
            ("", "Multinomial", 7),
            ("", "RNN", 24),
            ("", "GRU", 24),
            ("", "LSTM", 24),
            ("", "CumSum", 14),
            ("", "CumProd", 26),
            ("", "ScatterElements", 24),
            ("", "CenterCropPad", 24),
            ("", "ScatterND", 24),
            ("", "Slice", 24),
            ("", "Split", 24),
            ("", "Pad", 24),
            ("", "DFT", 17),
            ("", "STFT", 17),
            ("com.microsoft", "CausalConvWithState", 1),
            ("", "CausalConvWithState", 27),
            ("com.microsoft", "LinearAttention", 1),
            ("", "LinearAttention", 27),
            ("", "GridSample", 24),
            ("", "Col2Im", 24),
            ("", "RoiAlign", 24),
            ("", "MaxRoiPool", 24),
            ("", "MaxUnpool", 24),
        ];
        for (domain, op_type, since_version) in always_shape_keyed {
            assert!(
                matches!(
                    compile_shape_rule(domain, op_type, since_version),
                    CompileShapeRule::Always(CompileShapeSafety::ShapeKeyedOnly { .. })
                ),
                "{domain}::{op_type} can emit an MLX primitive without output_shapes"
            );
        }

        for op_type in [
            "MatMulBlockQuantizedFp4Weight",
            "MatMulBlockQuantizedFp8Weight",
            "MatMulNBits",
            "GatherBlockQuantized",
        ] {
            assert!(
                matches!(
                    compile_shape_rule("com.microsoft", op_type, 1),
                    CompileShapeRule::Classified(_)
                ),
                "com.microsoft::{op_type} needs a per-node quant geometry classifier"
            );
        }

        assert!(matches!(
            compile_shape_rule("com.microsoft", "GroupQueryAttention", 1),
            CompileShapeRule::Classified(_)
        ));
    }
}
