//! Engine core: the plan description (`NodeDesc`/`TensorRef`/`OutRef`) and the `TranslationContext`
//! a handler uses to Resolve inputs, Bind outputs, and emit MLX ops. It supports eager translation,
//! compiled decode/prefill/general closures, and recursively inlined control-flow subgraphs.

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

/// A constant initializer payload surfaced at compile time (session-owned storage). For a
/// control-flow BODY initializer the bytes are COPIED into `owned` (the subgraph handle is released
/// after the compile-time walk), and `data` points into that owned buffer; for an ordinary
/// session-owned initializer `owned` is `None` and `data` points at ORT's storage.
#[derive(Clone)]
pub struct InitData {
    pub data: *const c_void,
    pub shape: Vec<i64>,
    pub dtype: ort::ONNXTensorElementDataType,
    /// ORT-reported element count, used to bound typed initializer reads safely.
    pub count: usize,
    /// Owned copy of the bytes (control-flow body initializers), keeping `data` valid.
    #[allow(dead_code)]
    // RAII: backs the `data` pointer above; must outlive it even if not read.
    pub owned: Option<std::sync::Arc<Vec<u8>>>,
}

/// A single node input reference.
#[derive(Clone)]
pub struct TensorRef {
    pub name: String,
    pub source: Src,
    pub ctx_index: usize,
    /// True when a CtxInput is a hoisted constant initializer (wrapped/cached once).
    pub constant: bool,
    /// True when this tensor's VALUE derives purely from input SHAPES + constants (a `Shape`/`Size`
    /// output, or a deterministic chain over such) — independent of runtime DATA. Such a value is a
    /// genuine constant inside a shape-keyed `mlx_compile` trace (no tracer dependency), so it can be
    /// eval'd mid-trace to feed a static reshape/expand/slice/range target.
    pub shape_const: bool,
    pub init: Option<InitData>,
}

impl TensorRef {
    pub fn absent() -> Self {
        TensorRef {
            name: String::new(),
            source: Src::Absent,
            ctx_index: 0,
            constant: false,
            shape_const: false,
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

/// A partial copy-out descriptor for a shared-buffer KV `present` output. In shared-buffer mode
/// `present` is bound to the SAME ORT buffer as `past`, so the rows outside `[offset, offset+count)`
/// on `axis` are already correct (they are last step's `past`); only the `count` newly written rows
/// need to be copied back. This turns the per-token copy-out from O(capacity) into O(new-tokens).
///
/// `alias_ptr` is the ORT `past` input buffer address this `present` is expected to alias. The
/// partial write is applied ONLY when the actual `present` output buffer equals `alias_ptr` — i.e.
/// the runtime really bound `present` onto `past` (share-buffer IoBinding). When they differ (e.g.
/// the op-test harness, or any non-aliasing runtime) the full copy is used, so correctness never
/// depends on an assumption about ORT's binding.
#[derive(Clone, Copy)]
pub struct DeltaWrite {
    pub axis: usize,
    pub offset: i64,
    pub count: i64,
    pub alias_ptr: usize,
}

/// A TENSOR-valued attribute payload (Constant `value`, ConstantOfShape `value`) surfaced at compile
/// time. Raw host bytes are copied into `data` (owned, kept alive for the plan's lifetime) so the
/// handler can materialize an MLX array from it. Mirrors the C++ `CopyScalarAttrs` TENSOR path.
#[derive(Clone)]
#[allow(dead_code)] // Populated for TENSOR-valued attributes; consumed by upcoming handler waves.
pub struct ConstTensor {
    pub data: Vec<u8>,
    pub shape: Vec<i64>,
    pub dtype: ort::ONNXTensorElementDataType,
    pub count: usize,
}

impl ConstTensor {
    /// Raw bytes of a single-element tensor, for one-element attributes such as
    /// `ConstantOfShape`'s `value`. Returns `None` unless the tensor holds exactly one element.
    pub fn single_element_bytes(&self) -> Option<&[u8]> {
        if self.count != 1 || self.data.is_empty() {
            return None;
        }
        Some(self.data.as_slice())
    }
}

/// A control-flow node's body subgraph (If then/else branch, Scan/Loop body), captured recursively so
/// the translator can realize the control flow by translating the body inline. `input_names` /
/// `output_names` are the body graph's FORMAL inputs/outputs (positional). `nodes` are the body's
/// nodes in topological order, whose input TensorRefs already resolve against the body scope with a
/// fall-through to the enclosing scope (implicit inputs). Faithful port of the C++ `SubgraphDesc`.
#[derive(Clone)]
pub struct SubgraphDesc {
    pub attr_name: String,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub nodes: Vec<NodeDesc>,
}

/// One ONNX node with just the metadata the MLX translator needs.
#[derive(Clone)]
pub struct NodeDesc {
    pub node_id: usize,
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub since_version: i32,
    pub compile_shape_safety: crate::registry::CompileShapeSafety,
    pub ints: HashMap<String, i64>,
    pub floats: HashMap<String, f32>,
    pub int_arrays: HashMap<String, Vec<i64>>,
    pub float_arrays: HashMap<String, Vec<f32>>,
    pub strings: HashMap<String, String>,
    /// TENSOR-valued attributes (Constant/ConstantOfShape `value`), keyed by attribute name.
    #[allow(dead_code)] // Consumed by upcoming Constant/ConstantOfShape handler waves.
    pub tensors: HashMap<String, ConstTensor>,
    pub inputs: Vec<TensorRef>,
    pub outputs: Vec<OutRef>,
    /// Body subgraphs for a control-flow node (If/Scan/Loop). Empty for every ordinary op.
    pub subgraphs: Vec<SubgraphDesc>,
}

impl NodeDesc {
    pub fn new(
        op_type: String,
        domain: String,
        since_version: i32,
        compile_shape_safety: crate::registry::CompileShapeSafety,
    ) -> Self {
        NodeDesc {
            node_id: 0,
            name: String::new(),
            op_type,
            domain,
            since_version,
            compile_shape_safety,
            ints: HashMap::new(),
            floats: HashMap::new(),
            int_arrays: HashMap::new(),
            float_arrays: HashMap::new(),
            strings: HashMap::new(),
            tensors: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            subgraphs: Vec::new(),
        }
    }
}

/// A per-token dynamic subgraph input (non-constant `CtxInput`) fed as a compiled-closure argument.
#[derive(Clone)]
pub struct DynInput {
    pub name: String,
    pub ctx_index: usize,
}

/// A distinct RoPE cos/sin cache whose per-step row is pre-sliced OUTSIDE the compiled graph and fed
/// as a synthetic closure input (so the compiled graph never slices a cache at a runtime position —
/// which shapeless `mlx_compile` cannot shape-infer). `key` is the placeholder env key.
#[derive(Clone)]
pub struct SynthRope {
    pub key: String,
    pub cache_name: String,
}

/// Placeholder env key for the pre-sliced cos/sin row of the cache named `cache_name`.
pub fn rope_row_key(cache_name: &str) -> String {
    format!("__rope_row__{cache_name}")
}

/// Placeholder env key for the shared-buffer `valid_past` scalar. In the compiled shared-buffer
/// contract the number of valid past keys (`total_sequence_length - S`) is DATA that advances every
/// step, so it is fed as a live `[1]` int32 closure input (appended after the synth RoPE rows)
/// instead of being resolved from the in-graph `total_sequence_length` — which shapeless compile
/// would freeze at trace time. The GQA op reads it to drive the in-place write offset + causal mask.
pub const GQA_VALID_PAST_KEY: &str = "__gqa_valid_past";

/// Opaque payload handed to the `mlx_closure` trace thunk. Holds the raw pointers the thunk needs to
/// (re)build a `TranslationContext` over the plan for the ONE-TIME shapeless trace: the plan itself
/// (nodes + constant cache), the live ORT api/kernel-context (for constant host reads during the
/// trace), and the mlx stream. `kctx` is refreshed before every apply (only read during the trace).
///
/// It lives in a stable `Box` on the plan (`CompiledSubgraph::payload`) so its address — captured by
/// the base closure at build time — stays valid for the plan's lifetime.
pub struct TracePayload {
    pub plan: *mut Plan,
    pub ort_api: *const ort::OrtApi,
    pub kctx: *mut ort::OrtKernelContext,
    pub stream: mlxsys::mlx_stream,
    /// Which `CompiledSubgraph` slot on the plan this closure is tracing (so the unified core writes
    /// its discovered schema / transient handles back to the right slot).
    pub slot: Slot,
}

/// Selects one of the plan's `CompiledSubgraph` slots. Each slot carries a distinct [`CompiledConfig`]
/// and its own compiled closure + schema, but shares the ONE core in [`crate::compiled`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Slot {
    /// Decode (`plan.compiled`): shapeless, all decode features.
    Decode,
    /// Prefill (`plan.prefill`): shape-keyed, all decode features (Phase 2).
    Prefill,
    /// General static-shape subgraph (`plan.general`): shape-keyed, no attention.
    General,
}

impl Slot {
    #[inline]
    pub fn get(self, plan: &Plan) -> &CompiledSubgraph {
        match self {
            Slot::Decode => &plan.compiled,
            Slot::Prefill => &plan.prefill,
            Slot::General => &plan.general,
        }
    }
    #[inline]
    pub fn get_mut(self, plan: &mut Plan) -> &mut CompiledSubgraph {
        match self {
            Slot::Decode => &mut plan.compiled,
            Slot::Prefill => &mut plan.prefill,
            Slot::General => &mut plan.general,
        }
    }
}

/// How `mlx_compile` keys the fused closure — the single knob that used to distinguish the two
/// historical fast-path modules.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ShapeMode {
    /// Compile SHAPELESS: a growing/changing input shape never retraces (decode's growing KV length,
    /// which would otherwise recompile every token). One closure serves every step.
    Shapeless,
    /// Compile SHAPE-KEYED: `mlx_compile` keys on the input shapes/dtypes and transparently retraces
    /// (re-invokes the thunk) when they change, so a different shape recompiles rather than
    /// miscomputes (general static-shape subgraphs; prefill's varying query length).
    ShapeKeyed,
}

/// Static configuration selecting how a [`CompiledSubgraph`] traces, compiles, and applies. The two
/// historical fast paths — and the new prefill path — are now expressed as configurations of this
/// ONE core (the point of the unification: one place to trace → compile → cache → apply, with the
/// decode specialisations as composable opt-in features rather than a separate module):
///   * decode  = `{ Shapeless,  kv_alias, rope_as_data, delta_copyout }`
///   * general = `{ ShapeKeyed, contiguous_outputs }`
///   * prefill = `{ ShapeKeyed, kv_alias, rope_as_data, delta_copyout }` (Phase 2)
#[derive(Clone, Copy)]
pub struct CompiledConfig {
    /// Shapeless vs shape-keyed `mlx_compile` (see [`ShapeMode`]).
    pub shape_mode: ShapeMode,
    /// Support a fixed-capacity SHARED KV buffer (`present` aliased onto `past` at a runtime-owned max
    /// length): GQA writes the new K/V in place at a data-driven valid-past offset and masks the
    /// buffer tail, instead of the growing past/present concat. Detected per session at build time.
    pub kv_alias: bool,
    /// Feed the RoPE cos/sin rows (pre-sliced OUTSIDE the graph) and the `valid_past` offset as DATA
    /// closure inputs, and trace RoPE in its slice-free `[hd,hd]` matmul form (`rope_dynamic`), so the
    /// per-step position never bakes into the graph. Requires the decoder GQA shape (RoPE == head_dim).
    pub rope_as_data: bool,
    /// Copy back only the `S` new KV rows of a shared-buffer `present` (O(new-tokens) delta copy-out)
    /// rather than the whole `[B,kv,cap,hd]` buffer.
    pub delta_copyout: bool,
    /// Materialise each boundary output to row-major contiguous in the trace (so a general
    /// static-shape copy-out is a straight typed memcpy). The RoPE/KV paths leave it unset (they mirror
    /// the decode trace exactly).
    pub contiguous_outputs: bool,
}

impl CompiledConfig {
    /// Decode: shapeless (growing KV never retraces) with all decode specialisations on.
    pub const fn decode() -> Self {
        CompiledConfig {
            shape_mode: ShapeMode::Shapeless,
            kv_alias: true,
            rope_as_data: true,
            delta_copyout: true,
            contiguous_outputs: false,
        }
    }
    /// General static-shape subgraph: shape-keyed, no attention/KV specialisation, contiguous outputs.
    pub const fn general() -> Self {
        CompiledConfig {
            shape_mode: ShapeMode::ShapeKeyed,
            kv_alias: false,
            rope_as_data: false,
            delta_copyout: false,
            contiguous_outputs: true,
        }
    }
    /// Prefill (Phase 2): the SAME decoder subgraph as decode with EVERY decode specialisation
    /// (`kv_alias` / `rope_as_data` / `delta_copyout`), but a variable query length S>1. The causal
    /// mask's query-position `arange(0, S)` extent and the KV write width are read as Rust ints during
    /// the trace, so `S` bakes into the graph as a constant — a single shapeless closure traced at one
    /// S would miscompute at another. Shape-keyed compilation retraces per distinct S (re-baking it
    /// correctly) while replaying the fused closure for repeated prompt lengths.
    pub const fn prefill() -> Self {
        CompiledConfig {
            shape_mode: ShapeMode::ShapeKeyed,
            kv_alias: true,
            rope_as_data: true,
            delta_copyout: true,
            contiguous_outputs: false,
        }
    }
}

/// Unified compiled fast-path state: ONE parameterised core (`config`) plus the trace/compile/cache
/// machinery that every mode shares. What used to be the decode-only `CompiledDecode` and the
/// static-shape `GeneralCompiled` structs are now just two [`CompiledConfig`]s of this type (plus the
/// Phase 2 prefill config). The RoPE/KV fields below are only populated when the matching feature
/// flag is set; they stay empty/default for a plain general subgraph. Any doubt at build or apply
/// time flips `valid` off and the caller falls back to the always-correct eager translator.
pub struct CompiledSubgraph {
    /// Static mode + opt-in features (see [`CompiledConfig`]).
    pub config: CompiledConfig,
    /// Kill-switch off and no route-specific compile barrier.
    pub enabled: bool,
    /// Have we tried to build the compiled closure yet? (one-shot; failure => eager forever).
    pub attempted: bool,
    /// Is `closure` usable?
    pub valid: bool,
    /// Host control-flow decisions baked into the current general trace. `If` branches and static
    /// `Loop` trip counts are specialized explicitly because MLX's shape-keyed compile cache does
    /// not retrace when scalar input data changes.
    pub control_flow_key: Option<String>,
    /// Whether this general plan contains static control flow and therefore needs a host-decision
    /// specialization key. False for ordinary graphs, avoiding a full node walk on every call.
    pub control_flow_specialization_enabled: bool,
    /// Stable synthetic shape token per control-flow specialization. The token is appended as an
    /// unused closure input so MLX's native shape-keyed cache retains every visited branch/trip
    /// combination and can replay it when execution returns to that decision.
    pub control_flow_specializations: HashMap<String, i32>,
    /// The compiled closure.
    pub closure: Option<crate::mlx::Closure>,
    /// Ordered, de-duplicated dynamic ctx inputs = closure inputs `[0..n)`.
    pub dyn_inputs: Vec<DynInput>,
    /// External boundary outputs, in closure append order.
    pub ext_outputs: Vec<OutRef>,
    /// Transient MLX handles created during the trace; the compiled graph references them after the
    /// thunk returns, so they must outlive it (freed once with the plan).
    pub trace_transient: Vec<Array>,
    /// Stable payload for the trace thunk (self-referential to the enclosing plan).
    pub payload: Option<Box<TracePayload>>,
    // ---- rope_as_data / kv_alias state (empty/default when those features are off) ----
    /// Pre-sliced RoPE cos/sin row inputs, appended AFTER `dyn_inputs` (`rope_as_data`).
    pub synth_ropes: Vec<SynthRope>,
    /// Ctx input to read the RoPE start (past KV length) from, and its sequence axis (`rope_as_data`).
    pub rope_past_ctx_index: i32,
    pub rope_past_axis: i32,
    /// This session drives a fixed-capacity SHARED KV buffer (detected once at build time; `kv_alias`).
    pub shared_kv: bool,
    /// Ctx index of `attention_mask` — its width gives the valid-keys count used to derive the RoPE
    /// start (valid_past) at apply time in shared-buffer mode. `-1` when not shared / not found.
    pub mask_ctx_index: i32,
    /// Ctx index of GQA `total_sequence_length`; preferred over the mask width when available.
    pub total_seq_ctx_index: i32,
    /// GQA `present` KV outputs (shared-buffer mode) as `(present_output_name, past_input_ctx_index)`;
    /// their copy-out is a delta write of only the `S` new rows (`delta_copyout`).
    pub kv_present_names: Vec<(String, usize)>,
    /// Cross-attention K/V boundary inputs copied once per generation and reused for every token.
    /// Keyed by ORT KernelContext input index; cleared when a new self-attention sequence starts.
    pub stable_cross_inputs: HashMap<usize, Array>,
    /// Address of the first cross-attention K tensor for the active generation.
    pub stable_generation_key: Option<usize>,
}

impl CompiledSubgraph {
    pub fn new(config: CompiledConfig) -> Self {
        CompiledSubgraph {
            config,
            enabled: false,
            attempted: false,
            valid: false,
            control_flow_key: None,
            control_flow_specialization_enabled: false,
            control_flow_specializations: HashMap::new(),
            closure: None,
            dyn_inputs: Vec::new(),
            ext_outputs: Vec::new(),
            trace_transient: Vec::new(),
            payload: None,
            synth_ropes: Vec::new(),
            rope_past_ctx_index: -1,
            rope_past_axis: 2,
            shared_kv: false,
            mask_ctx_index: -1,
            total_seq_ctx_index: -1,
            kv_present_names: Vec::new(),
            stable_cross_inputs: HashMap::new(),
            stable_generation_key: None,
        }
    }
}

/// Persistent per-subgraph MLX state: the topo-ordered nodes plus the persistent cache of
/// wrapped/repacked constant arrays (keyed by name, reused across runs — freed with the plan).
pub struct Plan {
    pub nodes: Vec<NodeDesc>,
    pub cache: HashMap<String, Array>,
    pub native_attention_decode: bool,
    pub dedicated_decode_stream: bool,
    /// This subgraph carries float64 tensors and must run on an MLX **CPU** stream: MLX rejects
    /// float64 on Metal outright. GetCapability guarantees an fp64 cluster contains only fp64 nodes,
    /// so flipping the whole plan to the CPU stream never drags fp32 work off the GPU.
    pub requires_cpu_stream: bool,
    pub stable_cross_cache_enabled: bool,
    /// Boundary K/V inputs used directly by cross-attention MHA nodes.
    pub stable_cross_ctx_indices: Vec<usize>,
    /// Compiled-decode fast-path state (shapeless + decode features; see [`CompiledConfig::decode`]).
    pub compiled: CompiledSubgraph,
    /// Compiled-prefill fast-path state (shape-keyed + decode features; see [`CompiledConfig::prefill`]).
    pub prefill: CompiledSubgraph,
    /// General static-shape compiled fast-path state (see [`CompiledConfig::general`]).
    pub general: CompiledSubgraph,
}

pub(crate) fn is_separate_qkv_attention_op(domain: &str, op_type: &str) -> bool {
    op_type == "MultiHeadAttention" || (domain.is_empty() && op_type == "Attention")
}

pub(crate) fn is_separate_qkv_attention(node: &NodeDesc) -> bool {
    is_separate_qkv_attention_op(&node.domain, &node.op_type)
}

pub(crate) fn attention_past_key(node: &NodeDesc) -> Option<&TensorRef> {
    let index = if node.op_type == "MultiHeadAttention" {
        6
    } else if node.domain.is_empty() && node.op_type == "Attention" {
        4
    } else {
        return None;
    };
    node.inputs.get(index)
}

pub(crate) fn attention_cross_kv(node: &NodeDesc) -> Option<(&TensorRef, &TensorRef)> {
    if !is_separate_qkv_attention(node) {
        return None;
    }
    let [key, value] = node.inputs.get(1..3)? else {
        return None;
    };
    (key.source == Src::CtxInput && value.source == Src::CtxInput).then_some((key, value))
}

impl Plan {
    pub fn new(nodes: Vec<NodeDesc>) -> Self {
        let dynamic_ranges: Vec<&NodeDesc> = nodes
            .iter()
            .filter(|node| {
                node.op_type == "Range" && node.inputs.iter().any(|input| !input.constant)
            })
            .collect();
        let dynamic_tiles: Vec<&NodeDesc> = nodes
            .iter()
            .filter(|node| {
                node.op_type == "Tile" && node.inputs.get(1).is_some_and(|input| !input.constant)
            })
            .collect();
        let position_graph = match (dynamic_ranges.as_slice(), dynamic_tiles.as_slice()) {
            ([range], [tile]) => {
                let producer = |name: &str| {
                    nodes
                        .iter()
                        .find(|node| node.outputs.iter().any(|output| output.name == name))
                };
                let limit_is_start_plus_len = producer(&range.inputs[1].name).is_some_and(|add| {
                    add.op_type == "Add"
                        && add
                            .inputs
                            .iter()
                            .any(|input| input.name == range.inputs[0].name)
                });
                let tile_uses_range = producer(&tile.inputs[0].name).is_some_and(|expand| {
                    expand.op_type == "Expand"
                        && expand.inputs.first().is_some_and(|input| {
                            producer(&input.name).is_some_and(|unsqueeze| {
                                unsqueeze.op_type == "Unsqueeze"
                                    && unsqueeze
                                        .inputs
                                        .first()
                                        .is_some_and(|input| input.name == range.outputs[0].name)
                            })
                        })
                });
                let repeats_are_built =
                    producer(&tile.inputs[1].name).is_some_and(|concat| concat.op_type == "Concat");
                range.inputs[2].constant
                    && limit_is_start_plus_len
                    && tile_uses_range
                    && repeats_are_built
            }
            _ => false,
        };
        let optimized_position_graph = match (dynamic_ranges.as_slice(), dynamic_tiles.as_slice()) {
            ([range], [tile]) => {
                let producer = |name: &str| {
                    nodes
                        .iter()
                        .find(|node| node.outputs.iter().any(|output| output.name == name))
                };
                let limit_is_start_plus_len = producer(&range.inputs[1].name).is_some_and(|add| {
                    add.op_type == "Add"
                        && add
                            .inputs
                            .iter()
                            .any(|input| input.name == range.inputs[0].name)
                });
                let tile_uses_range = producer(&tile.inputs[0].name).is_some_and(|unsqueeze| {
                    unsqueeze.op_type == "Unsqueeze"
                        && unsqueeze
                            .inputs
                            .first()
                            .is_some_and(|input| input.name == range.outputs[0].name)
                });
                let repeats_are_built =
                    producer(&tile.inputs[1].name).is_some_and(|concat| concat.op_type == "Concat");
                limit_is_start_plus_len && tile_uses_range && repeats_are_built
            }
            _ => false,
        };
        let has_separate_qkv_attention = nodes.iter().any(is_separate_qkv_attention);
        let native_attention_decode = position_graph
            && nodes.len() > 32
            && has_separate_qkv_attention
            && nodes.iter().any(|node| {
                node.op_type == "MatMulNBits" && node.ints.get("bits").copied().unwrap_or(4) == 8
            });
        let dedicated_decode_stream = (native_attention_decode || optimized_position_graph)
            && nodes.len() > 32
            && has_separate_qkv_attention
            && nodes.iter().any(|node| {
                node.op_type == "MatMulNBits" && node.ints.get("bits").copied().unwrap_or(4) == 8
            });
        let mut stable_cross_ctx_indices = Vec::new();
        for (key, value) in nodes.iter().filter_map(attention_cross_kv) {
            for input in [key, value] {
                if !stable_cross_ctx_indices.contains(&input.ctx_index) {
                    stable_cross_ctx_indices.push(input.ctx_index);
                }
            }
        }
        let has_self_past_reset = nodes.iter().any(|node| {
            attention_past_key(node).is_some_and(|input| input.source == Src::CtxInput)
        });
        if !has_self_past_reset {
            stable_cross_ctx_indices.clear();
        }
        let stable_cross_cache_enabled =
            !std::env::var_os("ONNXRUNTIME_EP_MLX_NO_STABLE_CROSS_CACHE")
                .map(|value| value != "0" && !value.is_empty())
                .unwrap_or(false);
        // Best-effort fp64 detection from the descriptors alone (declared output dtypes + hoisted
        // initializers). `build_plan` ORs in the authoritative claim-time answer, which also sees
        // input dtypes; this covers the plans built directly in tests and control-flow bodies.
        let requires_cpu_stream = nodes.iter().any(|node| {
            node.outputs
                .iter()
                .any(|o| crate::registry::is_float64(o.otype))
                || node.inputs.iter().any(|i| {
                    i.init
                        .as_ref()
                        .is_some_and(|init| crate::registry::is_float64(init.dtype))
                })
        });
        Plan {
            nodes,
            cache: HashMap::new(),
            native_attention_decode,
            dedicated_decode_stream: dedicated_decode_stream && !requires_cpu_stream,
            requires_cpu_stream,
            stable_cross_cache_enabled,
            stable_cross_ctx_indices,
            compiled: CompiledSubgraph::new(CompiledConfig::decode()),
            prefill: CompiledSubgraph::new(CompiledConfig::prefill()),
            general: CompiledSubgraph::new(CompiledConfig::general()),
        }
    }
}

/// ONNX tensor element type -> MLX dtype (faithful port of `MlxDtypeFromOnnx`). Unknown types fall
/// back to fp32 so a stray dtype never crashes the wrap.
pub fn mlx_dtype_from_onnx(t: ort::ONNXTensorElementDataType) -> mlxsys::mlx_dtype {
    use ort::*;
    #[allow(non_upper_case_globals)]
    match t {
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
            mlxsys::mlx_dtype__MLX_FLOAT32
        }
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
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => {
            mlxsys::mlx_dtype__MLX_INT16
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
            mlxsys::mlx_dtype__MLX_INT32
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
            mlxsys::mlx_dtype__MLX_INT64
        }
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => {
            mlxsys::mlx_dtype__MLX_UINT8
        }
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

/// Narrow an ONNX i64 dimension to MLX's 32-bit `int` shape type, erroring on
/// overflow instead of silently wrapping (`as i32`). ONNX shape/dim values are
/// i64; MLX's C ABI uses `const int*` for shapes, so a dimension that does not
/// fit i32 cannot be represented — real tensors never approach this, so this is
/// defense-in-depth against corrupt/absurd dims. Sentinels `-1` (reshape infer)
/// and `0` (copy/allowzero) fit i32 and pass through unchanged.
pub(crate) fn dim_i32(d: i64) -> Result<i32, MlxError> {
    i32::try_from(d).map_err(|_| format!("MLX: dimension {d} does not fit MLX's i32 shape range"))
}

/// Human-readable MLX dtype name for trace Args (e.g. `"float32"`, `"bfloat16"`).
pub fn dtype_name(dt: mlxsys::mlx_dtype) -> &'static str {
    #[allow(non_upper_case_globals)]
    match dt {
        mlxsys::mlx_dtype__MLX_BOOL => "bool",
        mlxsys::mlx_dtype__MLX_UINT8 => "uint8",
        mlxsys::mlx_dtype__MLX_UINT16 => "uint16",
        mlxsys::mlx_dtype__MLX_UINT32 => "uint32",
        mlxsys::mlx_dtype__MLX_UINT64 => "uint64",
        mlxsys::mlx_dtype__MLX_INT8 => "int8",
        mlxsys::mlx_dtype__MLX_INT16 => "int16",
        mlxsys::mlx_dtype__MLX_INT32 => "int32",
        mlxsys::mlx_dtype__MLX_INT64 => "int64",
        mlxsys::mlx_dtype__MLX_FLOAT16 => "float16",
        mlxsys::mlx_dtype__MLX_FLOAT32 => "float32",
        mlxsys::mlx_dtype__MLX_FLOAT64 => "float64",
        mlxsys::mlx_dtype__MLX_BFLOAT16 => "bfloat16",
        mlxsys::mlx_dtype__MLX_COMPLEX64 => "complex64",
        _ => "unknown",
    }
}

/// Raw host bytes for a constant (initializer / constant-ctx-input) tensor, surfaced at translate
/// time so shape/axes/indices operands (Reshape shape, Slice starts/ends/axes/steps, Pad pads, …)
/// can be read as plain host integers. Faithful port of the C++ `HostBytes` / `RawHost`.
pub struct HostBytes {
    pub data: *const c_void,
    #[allow(dead_code)] // Retained for symmetry/debugging; reads use `count`+`dtype`.
    pub shape: Vec<i64>,
    pub count: usize,
    pub dtype: ort::ONNXTensorElementDataType,
}

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
    /// Per-translation cast CSE. This is intentionally scoped to one graph build: runtime inputs
    /// must never be retained across Compute calls, while repeated Q/K/V and gate/up projections
    /// should share one explicit activation cast inside the compiled closure.
    astype_cache: HashMap<(usize, u32), mlxsys::mlx_array>,
    /// Path a handler declared for the CURRENT node (fast vs composed); the dispatcher
    /// resets this before each handler and reads it after (see `registry::translate`).
    path_mark: Option<crate::trace::PathMark>,
    /// Cached tracing-enabled gate so `mark_fast`/`mark_composed` are a cheap no-op (and
    /// never allocate a reason string) when tracing is off.
    trace_enabled: bool,
    /// True while translating INSIDE the compiled-decode closure trace. In this mode RoPE uses the
    /// pre-sliced cos/sin ROW placeholders (fed as extra closure inputs) + a matmul rotate-half, so
    /// the graph carries no dynamic Slice (which shapeless `mlx_compile` cannot shape-infer).
    rope_dynamic: bool,
    /// True when the compiled-decode trace is for a fixed-capacity SHARED KV buffer session (GQA
    /// writes the new K/V in place at the valid-past offset and masks the buffer tail). Mirrors
    /// `plan.compiled.shared_kv`; only meaningful while `rope_dynamic` is set.
    shared_kv: bool,
    /// True when the compiled trace is SHAPE-KEYED (prefill) rather than shapeless (decode). In this
    /// mode the query length `S` and `valid_past` are fixed for the shape key, so the shared-buffer
    /// attention can be STATICALLY narrowed to the valid prefix `[0, valid_past+S)` (a static slice +
    /// causal SDPA) instead of masking over the full KV capacity. Only meaningful with `rope_dynamic`.
    compiled_shape_keyed: bool,
    /// The static `valid_past` (number of already-valid KV rows) for a shape-keyed compiled trace —
    /// known per shape key because the `attention_mask` width (= `valid_past + S`) is part of the key.
    /// Only meaningful when `compiled_shape_keyed` is set.
    compiled_valid_past: i32,
    /// Shared-buffer KV `present` outputs (name -> delta) for the EAGER path: only the `count` new
    /// rows at `offset` need copying back (the rest already alias correct `past` rows). Empty on the
    /// growing path so its copy-out stays a full, bit-for-bit-unchanged memcpy.
    kv_deltas: HashMap<String, DeltaWrite>,
    /// Evaluated MLX arrays that must be copied back into mutable ORT input buffers for
    /// side-effecting ops such as PagedAttention's in-place KV-cache update.
    input_writebacks: Vec<(*mut c_void, mlxsys::mlx_array, usize)>,
    /// True while translating inside the GENERAL compiled-subgraph closure trace. In this mode the
    /// dynamic inputs are shapeless tracer placeholders with no host data, so any mid-graph host eval
    /// (`contiguous_eval` — the host-computed Det/NonZero/Unique ops) is illegal and must fail the
    /// trace cleanly so the caller falls back to the eager translator instead of eval-ing a tracer.
    in_general_trace: bool,
    /// Shared-buffer KV `present` outputs discovered during a COMPILED trace (`rope_dynamic`), as
    /// `(present_output_name, past_input_ctx_index)`. Collected here (not written straight to a plan
    /// slot) so the unified core can hand them to whichever `CompiledSubgraph` slot it is tracing.
    compiled_kv_present: Vec<(String, usize)>,
    /// True while translating inside ANY SHAPE-KEYED compiled subgraph trace (`CompiledConfig::general`
    /// or `CompiledConfig::prefill` — both `ShapeMode::ShapeKeyed`), as opposed to a SHAPELESS decode
    /// trace or plain eager (non-compiled) translation. Set once, unconditionally, from `cfg.shape_mode`
    /// right after `TranslationContext::new` — independent of `compiled_shape_keyed` (which additionally
    /// requires the RoPE-as-data shared-buffer "narrow" detection to have succeeded). Multi-token
    /// matrix-matrix compiled prefill work (M>1) always shows up here; decode (M==1, Shapeless) never
    /// does. Used to gate opt-in fast kernels that must never run during decode.
    shape_keyed_compile: bool,
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
            astype_cache: HashMap::new(),
            path_mark: None,
            trace_enabled: crate::trace::tracer().is_enabled(),
            rope_dynamic: false,
            shared_kv: false,
            compiled_shape_keyed: false,
            compiled_valid_past: 0,
            kv_deltas: HashMap::new(),
            input_writebacks: Vec::new(),
            in_general_trace: false,
            compiled_kv_present: Vec::new(),
            shape_keyed_compile: false,
        }
    }

    /// Declare that the current node took its **fused MLX kernel** fast path (green/normal).
    /// `kernel` is the MLX kernel name (e.g. `"mlx_fast_sdpa"`). Cheap no-op when tracing is off.
    #[inline]
    pub fn mark_fast(&mut self, kernel: &'static str) {
        if self.trace_enabled {
            self.path_mark = Some(crate::trace::PathMark::Fast(kernel));
        }
    }

    /// Declare that the current node fell back to a slower **composed/generic** path even though
    /// a fused MLX kernel exists — this is flagged prominently in the trace. `reason` explains why
    /// (e.g. `"block_size 16 → dequant+dense matmul"`). Cheap no-op (no string alloc) when tracing
    /// is off.
    #[inline]
    pub fn mark_composed(&mut self, reason: impl Into<String>) {
        if self.trace_enabled {
            self.path_mark = Some(crate::trace::PathMark::Composed(reason.into()));
        }
    }

    /// Clear the per-node path slot before dispatching a handler.
    #[inline]
    pub fn reset_path_mark(&mut self) {
        self.path_mark = None;
    }

    /// Take the path a handler declared for the node just dispatched (see `mark_fast`/`mark_composed`).
    #[inline]
    pub fn take_path_mark(&mut self) -> Option<crate::trace::PathMark> {
        self.path_mark.take()
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

    /// Look up a persistent (plan-cached) array by key — the borrowed raw handle if present. Used by
    /// weight-repack handlers (MatMulNBits) to reuse a once-built constant across runs.
    pub fn cache_get(&self, key: &str) -> Option<mlxsys::mlx_array> {
        self.plan.cache.get(key).map(|a| a.as_raw())
    }

    /// Insert an owning array into the persistent plan cache under `key` (freed with the plan) and
    /// return its borrowed raw handle. Use only for genuinely constant (initializer) data.
    pub fn cache_put(&mut self, key: String, a: Array) -> mlxsys::mlx_array {
        let raw = a.as_raw();
        self.plan.cache.insert(key, a);
        raw
    }

    /// Bind a node output name to a produced MLX array (visible to downstream nodes and CopyOut).
    pub fn bind(&mut self, o: &OutRef, a: mlxsys::mlx_array) {
        self.env.insert(o.name.clone(), a);
    }

    pub fn write_back_input(
        &mut self,
        input: &TensorRef,
        a: mlxsys::mlx_array,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let a = self.contiguous(a)?;
        self.env.insert(input.name.clone(), a);
        match input.source {
            Src::CtxInput => {
                let (data, shape, _) = self.read_ctx_input(input.ctx_index)?;
                let arr = std::mem::ManuallyDrop::new(Array::from_raw(a));
                let count = shape.iter().try_fold(1usize, |count, &dim| {
                    usize::try_from(dim)
                        .ok()
                        .and_then(|dim| count.checked_mul(dim))
                        .ok_or_else(|| "MLX: input writeback shape overflow".to_string())
                })?;
                self.input_writebacks
                    .push((data as *mut c_void, a, count * arr.itemsize()));
            }
            Src::Intermediate => {}
            _ => return Err("MLX: cannot write back an immutable input".to_string()),
        }
        Ok(a)
    }

    /// Emit the per-op trace detail for the node just translated (only when tracing is on).
    ///
    /// Reads each output's shape/dtype/size from the (lazy) bound arrays — shape and dtype
    /// are graph metadata available WITHOUT eval, so the normal (fine-off) path stays fully
    /// fused. When [`fine_enabled`](crate::trace::MlxTracer::fine_enabled) is set, additionally
    /// forces an `mlx_array_eval` of this node's outputs to time it individually — a
    /// GPU-inclusive per-op bar that BREAKS fusion (debug-only, slower).
    pub fn trace_node(
        &mut self,
        _op_type: &str,
        node: &NodeDesc,
        start: Option<std::time::Instant>,
    ) {
        if !self.trace_enabled {
            return;
        }
        let Some(start) = start else {
            return;
        };
        let tr = crate::trace::tracer();

        // Output metadata (shapes / dtype / elements / bytes) — from whatever is materialized.
        let mut out_shapes: Vec<String> = Vec::new();
        let mut dtype = "";
        let mut elements: u64 = 0;
        let mut bytes: u64 = 0;
        for o in &node.outputs {
            if o.name.is_empty() {
                continue;
            }
            if let Some(&raw) = self.env.get(&o.name) {
                // Borrow the handle for metadata; do NOT free it (owned by arena/cache).
                let arr = std::mem::ManuallyDrop::new(Array::from_raw(raw));
                let sh = arr.shape();
                let cnt: u64 = sh.iter().map(|&d| d.max(0) as u64).product();
                out_shapes.push(format!("{sh:?}"));
                dtype = dtype_name(arr.dtype());
                elements += cnt;
                bytes += cnt * arr.itemsize() as u64;
            }
        }
        // Input shapes, best-effort from whatever is already materialized in the env.
        let mut in_shapes: Vec<String> = Vec::new();
        for inp in &node.inputs {
            if inp.name.is_empty() {
                continue;
            }
            if let Some(&raw) = self.env.get(&inp.name) {
                let arr = std::mem::ManuallyDrop::new(Array::from_raw(raw));
                in_shapes.push(format!("{:?}", arr.shape()));
            }
        }
        let out_s = out_shapes.join(";");
        let in_s = in_shapes.join(";");

        // A build-time span with resource metadata. The fused subgraph runs as a single
        // `mlx.eval` (see finish_boundary); per-kernel GPU detail is the Xcode GPU capture.
        tr.record_op_meta(
            node,
            start,
            start.elapsed(),
            &out_s,
            &in_s,
            dtype,
            elements,
            bytes,
        );
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
                let ishape: Vec<i32> = shape
                    .iter()
                    .map(|&d| dim_i32(d))
                    .collect::<Result<_, _>>()?;
                if r.constant {
                    // Runtime ctx-input buffers are valid only for this Compute call. Keep the
                    // defensive copy for compatibility with plans built by older ORT behavior.
                    let arr = Array::from_data(data, &ishape, mlx_dtype_from_onnx(dtype));
                    let raw = arr.as_raw();
                    self.plan.cache.insert(r.name.clone(), arr);
                    Ok(raw)
                } else {
                    // Zero-copy wrap of the live ORT input buffer (ORT owns it for the whole Compute
                    // call; MLX borrows via a no-op deallocator and the wrapper drops before we
                    // return, after the synchronous boundary eval). Dominant win in shared-buffer
                    // mode where the per-token past K/V is a full [B,kv,cap,hd] buffer that
                    // `from_data` used to memcpy every token.
                    let arr = Array::from_data_managed(data, &ishape, mlx_dtype_from_onnx(dtype));
                    let raw = self.keep(arr);
                    self.env.insert(r.name.clone(), raw);
                    Ok(raw)
                }
            }
            Src::Initializer => {
                let init = r
                    .init
                    .as_ref()
                    .ok_or_else(|| format!("MLX: initializer {} has no data", r.name))?;
                // Body-local initializers may legally shadow an outer/other-branch name. Qualify
                // copied body constants by their stable owned allocation so branch specialization
                // cannot reuse a same-named constant from another scope.
                let cache_key = if init.owned.is_some() {
                    format!("{}\0{:p}", r.name, init.data)
                } else {
                    r.name.clone()
                };
                if let Some(a) = self.plan.cache.get(&cache_key) {
                    return Ok(a.as_raw());
                }
                let ishape: Vec<i32> = init
                    .shape
                    .iter()
                    .map(|&d| dim_i32(d))
                    .collect::<Result<_, _>>()?;
                // Compile copied initializer bytes into Plan-owned Arc storage. Borrow that stable
                // allocation directly for the plan lifetime.
                let arr =
                    Array::from_data_managed(init.data, &ishape, mlx_dtype_from_onnx(init.dtype));
                let raw = arr.as_raw();
                self.plan.cache.insert(cache_key, arr);
                Ok(raw)
            }
            Src::Absent => Err("MLX: absent input".to_string()),
        }
    }

    /// Read a constant/parameter input's HOST bytes at translate time (shape/axes/indices operands).
    /// Handles both a compile-time `Initializer` and a constant/dynamic `CtxInput` (read live from
    /// the kernel context each run). Faithful port of `TranslationContext::RawHost`.
    pub fn raw_host(&self, r: &TensorRef) -> Result<HostBytes, MlxError> {
        match r.source {
            Src::Initializer => {
                let init = r
                    .init
                    .as_ref()
                    .ok_or_else(|| format!("MLX: initializer {} has no data", r.name))?;
                Ok(HostBytes {
                    data: init.data,
                    shape: init.shape.clone(),
                    count: init.count,
                    dtype: init.dtype,
                })
            }

            Src::CtxInput => {
                let (data, shape, dtype) = self.read_ctx_input(r.ctx_index)?;
                let count = shape.iter().map(|&d| d as usize).product::<usize>();
                Ok(HostBytes {
                    data,
                    shape,
                    count,
                    dtype,
                })
            }
            _ => Err(format!("MLX: RawHost on non-constant input {}", r.name)),
        }
    }

    /// Read a constant int64 parameter input (shape/axes/starts/ends/steps/pads/repeats/split) as a
    /// host `Vec<i64>` at translate time. The claim predicate verified it is a tensor(int64) input.
    pub fn read_ints(&self, r: &TensorRef) -> Result<Vec<i64>, MlxError> {
        let h = self.raw_host(r)?;
        if h.data.is_null() {
            return Ok(Vec::new());
        }
        let p = h.data as *const i64;
        Ok(unsafe { std::slice::from_raw_parts(p, h.count) }.to_vec())
    }

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
        op: unsafe extern "C" fn(
            *mut mlxsys::mlx_array,
            mlxsys::mlx_array,
            mlxsys::mlx_stream,
        ) -> i32,
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

    /// Cast once per source handle and target dtype during this graph translation.
    pub fn astype_once(
        &mut self,
        a: mlxsys::mlx_array,
        t: mlxsys::mlx_dtype,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let key = (a.ctx as usize, t as u32);
        if let Some(&casted) = self.astype_cache.get(&key) {
            return Ok(casted);
        }
        let casted = self.astype(a, t)?;
        self.astype_cache.insert(key, casted);
        Ok(casted)
    }

    /// Materialize and retain a cast of a genuinely constant array in the plan cache.
    pub fn cache_astype(
        &mut self,
        key: String,
        a: mlxsys::mlx_array,
        t: mlxsys::mlx_dtype,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        if let Some(casted) = self.cache_get(&key) {
            return Ok(casted);
        }
        let mut raw = unsafe { mlxsys::mlx_array_new() };
        let rc = unsafe { mlxsys::mlx_astype(&mut raw, a, t, self.stream) };
        if rc != 0 {
            unsafe { mlxsys::mlx_array_free(raw) };
            return Err("mlx_astype failed".to_string());
        }
        let casted = Array::from_raw(raw);
        casted.eval();
        Ok(self.cache_put(key, casted))
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

    /// Generic result-producing MLX op runner: builds a fresh result array, invokes the closure with
    /// `(&mut result, stream)`, re-wraps whatever handle the op produced (RAII), errors on non-zero
    /// return, and keeps + returns the raw result. This replaces the per-signature helper boilerplate
    /// so each op handler is a one-liner regardless of the underlying `mlx_*` arity.
    pub fn emit<F>(&mut self, f: F) -> Result<mlxsys::mlx_array, MlxError>
    where
        F: FnOnce(*mut mlxsys::mlx_array, mlxsys::mlx_stream) -> i32,
    {
        let mut res = Array::new();
        let mut raw = res.as_raw();
        let rc = f(&mut raw, self.stream);
        res = Array::from_raw(raw);
        if rc != 0 {
            return Err("mlx op failed".to_string());
        }
        Ok(self.keep(res))
    }

    /// Shape-preserving cumulative sum.
    ///
    /// MLX 0.32's backing `Scan` primitive lacks `Primitive::output_shapes`, so every registered
    /// handler that can reach this helper must declare `CompileShapeSafety::ShapeKeyedOnly`.
    pub fn cumsum(
        &mut self,
        a: mlxsys::mlx_array,
        axis: i32,
        reverse: bool,
        inclusive: bool,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.emit(|res, stream| unsafe {
            mlxsys::mlx_cumsum(res, a, axis, reverse, inclusive, stream)
        })
    }

    // ---- array introspection (borrowed raw handles; ownership stays with the arena/cache) ---------

    pub fn shape_of(&self, a: mlxsys::mlx_array) -> Vec<i32> {
        let nd = unsafe { mlxsys::mlx_array_ndim(a) };
        let sh = unsafe { mlxsys::mlx_array_shape(a) };
        (0..nd).map(|i| unsafe { *sh.add(i) }).collect()
    }

    pub fn ndim(&self, a: mlxsys::mlx_array) -> usize {
        unsafe { mlxsys::mlx_array_ndim(a) }
    }

    pub fn dim(&self, a: mlxsys::mlx_array, i: i32) -> i32 {
        unsafe { mlxsys::mlx_array_dim(a, i) }
    }

    pub fn size_of(&self, a: mlxsys::mlx_array) -> usize {
        unsafe { mlxsys::mlx_array_size(a) }
    }

    pub fn dtype_of(&self, a: mlxsys::mlx_array) -> mlxsys::mlx_dtype {
        unsafe { mlxsys::mlx_array_dtype(a) }
    }

    // ---- constant materialization helpers ---------------------------------------------------------

    /// A kept 0-d float32 scalar array.
    pub fn scalar_f32(&mut self, v: f32) -> mlxsys::mlx_array {
        self.keep(Array::from_raw(unsafe { mlxsys::mlx_array_new_float32(v) }))
    }

    /// A kept 0-d float64 scalar array. Used by the fp64 CPU-stream path so a constant like Selu's
    /// alpha is not first rounded through fp32.
    pub fn scalar_f64(&mut self, v: f64) -> mlxsys::mlx_array {
        let sh: [i32; 0] = [];
        self.keep(Array::from_data(
            &v as *const f64 as *const c_void,
            &sh,
            mlxsys::mlx_dtype__MLX_FLOAT64,
        ))
    }

    /// A kept 0-d int32 scalar array.
    pub fn scalar_i32(&mut self, v: i32) -> mlxsys::mlx_array {
        self.keep(Array::from_raw(unsafe { mlxsys::mlx_array_new_int(v) }))
    }

    /// A kept 0-d int64 scalar array.
    pub fn scalar_i64(&mut self, v: i64) -> mlxsys::mlx_array {
        let sh: [i32; 0] = [];
        self.keep(Array::from_data(
            &v as *const i64 as *const c_void,
            &sh,
            mlxsys::mlx_dtype__MLX_INT64,
        ))
    }

    /// A kept 0-d scalar built byte-exactly from host data of the given MLX dtype.
    ///
    /// Deliberately does not widen through FLOAT64 as a transport: the EP runs on a GPU stream and
    /// MLX rejects float64 there, so a widening round-trip would abort the process.
    pub fn scalar_from_bytes(
        &mut self,
        bytes: &[u8],
        dtype: mlxsys::mlx_dtype,
    ) -> mlxsys::mlx_array {
        let sh: [i32; 0] = [];
        self.keep(Array::from_data(
            bytes.as_ptr() as *const c_void,
            &sh,
            dtype,
        ))
    }

    /// A kept 1-D (or 0-D) int64 array wrapping host values (Shape/Size outputs).
    // builder-style: constructs an mlx_array from host data into the ctx arena.
    #[allow(clippy::wrong_self_convention)]
    pub fn from_host_i64(&mut self, data: &[i64], shape: &[i32]) -> mlxsys::mlx_array {
        self.keep(Array::from_data(
            data.as_ptr() as *const c_void,
            shape,
            mlxsys::mlx_dtype__MLX_INT64,
        ))
    }

    // ---- common data-movement helpers -------------------------------------------------------------

    pub fn reshape(
        &mut self,
        a: mlxsys::mlx_array,
        shape: &[i32],
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.emit(|res, s| unsafe { mlxsys::mlx_reshape(res, a, shape.as_ptr(), shape.len(), s) })
    }

    pub fn transpose(
        &mut self,
        a: mlxsys::mlx_array,
        axes: &[i32],
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.emit(|res, s| unsafe {
            mlxsys::mlx_transpose_axes(res, a, axes.as_ptr(), axes.len(), s)
        })
    }

    /// Force a (possibly strided/broadcast) view to row-major contiguous — required before a boundary
    /// output produced by a view op (transpose/slice/expand/split) is memcpy'd across the ORT boundary.
    pub fn contiguous(&mut self, a: mlxsys::mlx_array) -> Result<mlxsys::mlx_array, MlxError> {
        self.emit(|res, s| unsafe { mlxsys::mlx_contiguous(res, a, false, s) })
    }

    pub fn zeros(
        &mut self,
        shape: &[i32],
        dtype: mlxsys::mlx_dtype,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.emit(|res, s| unsafe { mlxsys::mlx_zeros(res, shape.as_ptr(), shape.len(), dtype, s) })
    }

    /// Softmax over the last axis (precise), used by the Softmax handler.
    /// Softmax over a single (already-normalized, or negative) axis via `mlx_softmax_axis`.
    ///
    /// The `precise` flag asks MLX to accumulate in float32. That is what we want for fp16/bf16, but
    /// for a float64 input it is a *down*cast: an input near float32's max (3.4e38 is a perfectly
    /// ordinary float64) overflows to inf and the result comes back NaN. Ask for the precise path
    /// only when the input is actually narrower than float32.
    pub fn softmax_axis(
        &mut self,
        a: mlxsys::mlx_array,
        axis: i32,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let precise = self.dtype_of(a) != mlxsys::mlx_dtype__MLX_FLOAT64;
        let mut res = Array::new();
        let mut raw = res.as_raw();
        let rc = unsafe { mlxsys::mlx_softmax_axis(&mut raw, a, axis, precise, self.stream) };
        res = Array::from_raw(raw);
        if rc != 0 {
            return Err("mlx_softmax_axis failed".to_string());
        }
        Ok(self.keep(res))
    }

    // ---- extra op helpers shared by the signal/random/recurrent/ssm/misc/controlflow ops ---------

    /// `a * b` (elementwise, with MLX broadcast).
    pub fn mul(
        &mut self,
        a: mlxsys::mlx_array,
        b: mlxsys::mlx_array,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.binary(mlxsys::mlx_multiply, a, b)
    }

    /// `a + b`.
    pub fn add(
        &mut self,
        a: mlxsys::mlx_array,
        b: mlxsys::mlx_array,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.binary(mlxsys::mlx_add, a, b)
    }

    /// `a - b`.
    pub fn sub(
        &mut self,
        a: mlxsys::mlx_array,
        b: mlxsys::mlx_array,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.binary(mlxsys::mlx_subtract, a, b)
    }

    /// `a @ b` (matmul).
    pub fn matmul(
        &mut self,
        a: mlxsys::mlx_array,
        b: mlxsys::mlx_array,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.emit(|res, s| unsafe { mlxsys::mlx_matmul(res, a, b, s) })
    }

    /// Concatenate two arrays along `axis`.
    pub fn concat2(
        &mut self,
        a: mlxsys::mlx_array,
        b: mlxsys::mlx_array,
        axis: i32,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let mut vec = VectorArray::new();
        vec.append(a);
        vec.append(b);
        self.emit(|res, s| unsafe { mlxsys::mlx_concatenate_axis(res, vec.as_raw(), axis, s) })
    }

    /// Contiguous strided slice with unit stride (`start`/`stop` per axis).
    pub fn slice(
        &mut self,
        a: mlxsys::mlx_array,
        start: &[i32],
        stop: &[i32],
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let stride = vec![1i32; start.len()];
        self.emit(|res, s| unsafe {
            mlxsys::mlx_slice(
                res,
                a,
                start.as_ptr(),
                start.len(),
                stop.as_ptr(),
                stop.len(),
                stride.as_ptr(),
                stride.len(),
                s,
            )
        })
    }

    /// `expand_dims(a, axis)`.
    pub fn expand_dims(
        &mut self,
        a: mlxsys::mlx_array,
        axis: i32,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.emit(|res, s| unsafe { mlxsys::mlx_expand_dims(res, a, axis, s) })
    }

    /// `arange(start, stop, step)` of the given dtype.
    pub fn arange(
        &mut self,
        start: f64,
        stop: f64,
        step: f64,
        dtype: mlxsys::mlx_dtype,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.emit(|res, s| unsafe { mlxsys::mlx_arange(res, start, stop, step, dtype, s) })
    }

    /// Elementwise `a <= b` (broadcasting), producing a bool array.
    pub fn less_equal(
        &mut self,
        a: mlxsys::mlx_array,
        b: mlxsys::mlx_array,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.binary(mlxsys::mlx_less_equal, a, b)
    }

    /// `where(cond, x, y)` (elementwise select with broadcasting).
    pub fn where_(
        &mut self,
        cond: mlxsys::mlx_array,
        x: mlxsys::mlx_array,
        y: mlxsys::mlx_array,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.emit(|res, s| unsafe { mlxsys::mlx_where(res, cond, x, y, s) })
    }

    /// Functional `slice_update` with a DATA start offset (an int array giving the start index for
    /// each of `axes`; unspecified axes start at 0). Returns the full-shape updated `src`.
    pub fn slice_update_dynamic(
        &mut self,
        src: mlxsys::mlx_array,
        update: mlxsys::mlx_array,
        start: mlxsys::mlx_array,
        axes: &[i32],
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.emit(|res, s| unsafe {
            mlxsys::mlx_slice_update_dynamic(res, src, update, start, axes.as_ptr(), axes.len(), s)
        })
    }

    /// `squeeze(a, axis)`.
    pub fn squeeze(
        &mut self,
        a: mlxsys::mlx_array,
        axis: i32,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        self.emit(|res, s| unsafe { mlxsys::mlx_squeeze_axis(res, a, axis, s) })
    }

    /// Stack a list of same-shaped arrays along a new `axis`.
    pub fn stack(
        &mut self,
        parts: &[mlxsys::mlx_array],
        axis: i32,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let mut vec = VectorArray::new();
        for &p in parts {
            vec.append(p);
        }
        self.emit(|res, s| unsafe { mlxsys::mlx_stack_axis(res, vec.as_raw(), axis, s) })
    }

    /// A kept 0-d complex64 scalar array (real, imag).
    pub fn scalar_complex(&mut self, re: f32, im: f32) -> mlxsys::mlx_array {
        self.keep(Array::from_raw(unsafe {
            mlxsys::mlx_array_new_complex(re, im)
        }))
    }

    /// A kept 0-d bool scalar array.
    pub fn scalar_bool(&mut self, v: bool) -> mlxsys::mlx_array {
        self.keep(Array::from_raw(unsafe { mlxsys::mlx_array_new_bool(v) }))
    }

    /// A kept array wrapping host bytes of the given `dtype` and `shape`.
    // builder-style: constructs an mlx_array from host data into the ctx arena.
    #[allow(clippy::wrong_self_convention)]
    pub fn from_host(
        &mut self,
        data: *const c_void,
        shape: &[i32],
        dtype: mlxsys::mlx_dtype,
    ) -> mlxsys::mlx_array {
        self.keep(Array::from_data(data, shape, dtype))
    }

    /// Force `a` contiguous, evaluate it, and return `(kept handle, shape, dtype)` so a handler can
    /// read its host bytes mid-graph (the host-computed ops: Det / NonZero / Unique). The kept handle
    /// stays alive for the rest of the run.
    pub fn contiguous_eval(&mut self, a: mlxsys::mlx_array) -> Result<mlxsys::mlx_array, MlxError> {
        if self.in_general_trace {
            // A host-computed op (Det/NonZero/Unique) needs the input's DATA, but in a general
            // compiled trace the inputs are shapeless tracer placeholders with no data. Fail the
            // trace so `try_compiled_general` falls back to the eager translator (which evals real
            // data). Never eval a tracer here — that would abort inside MLX.
            return Err("MLX: host eval not permitted in general compiled trace".to_string());
        }
        let r = self.contiguous(a)?;
        let mut v = VectorArray::new();
        v.append(r);
        mlx::eval(&v)?;
        Ok(r)
    }

    /// Raw host byte pointer of an (evaluated) array — valid until the array is freed.
    pub fn host_ptr(&self, a: mlxsys::mlx_array) -> *const u8 {
        unsafe { mlxsys::mlx_array_data_uint8(a) }
    }

    /// Like [`Self::contiguous_eval`] but permitted inside a general trace. ONLY call when the array
    /// is provably shape-const (a `Shape`/`Size`-derived value or a chain over constants): it has no
    /// tracer dependency, so eval computes a genuine constant rather than aborting on a tracer.
    fn contiguous_eval_const(
        &mut self,
        a: mlxsys::mlx_array,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let r = self.contiguous(a)?;
        let mut v = VectorArray::new();
        v.append(r);
        mlx::eval(&v)?;
        Ok(r)
    }

    /// Read an int64 vector honoring the ref's SOURCE. Constant initializers / subgraph boundary
    /// inputs are read from host bytes (cheap). A dynamic intermediate is materialized by a mid-graph
    /// eval — permitted inside a general trace ONLY when the ref is `shape_const` (no tracer
    /// dependency); a data-dependent runtime value in a general trace fails so the plan falls back to
    /// eager (where real data exists).
    pub fn read_ints_eval(&mut self, r: &TensorRef) -> Result<Vec<i64>, MlxError> {
        if !matches!(r.source, Src::Intermediate) {
            return self.read_ints(r);
        }
        if self.in_general_trace && !r.shape_const {
            return Err(
                "MLX: data-dependent runtime shape cannot be read in a general trace".to_string(),
            );
        }
        let a = self.resolve(r)?;
        let a = self.astype(a, mlxsys::mlx_dtype__MLX_INT64)?;
        let a = self.contiguous_eval_const(a)?;
        let n = self.size_of(a);
        if n == 0 {
            return Ok(Vec::new());
        }
        let p = unsafe { mlxsys::mlx_array_data_int64(a) };
        if p.is_null() {
            return Ok(Vec::new());
        }
        Ok(unsafe { std::slice::from_raw_parts(p, n) }.to_vec())
    }

    /// Evaluate a 0-d/1-element integer array and read its scalar value host-side (int32 or int64).
    /// Used by the eager GQA path to recover `total_sequence_length` (an in-graph scalar) so the
    /// valid-past length becomes a known integer for static slice bounds. This forces a small
    /// mid-graph eval, mirroring the host-computed ops (Det/NonZero/Unique via `contiguous_eval`).
    pub fn read_scalar_i64(&mut self, a: mlxsys::mlx_array) -> Result<i64, MlxError> {
        let dt = self.dtype_of(a);
        let a = if dt == mlxsys::mlx_dtype__MLX_INT64 {
            a
        } else {
            self.astype(a, mlxsys::mlx_dtype__MLX_INT64)?
        };
        let a = self.contiguous_eval(a)?;
        let mut v: i64 = 0;
        let rc = unsafe { mlxsys::mlx_array_item_int64(&mut v, a) };
        if rc != 0 {
            return Err("mlx_array_item_int64 failed".to_string());
        }
        Ok(v)
    }

    /// In-place KV write: functional `slice_update` of `update` into `src` over `[start, stop)`
    /// (unit stride). Returns the full-shape updated buffer (the untouched region carries `src`'s
    /// data through), matching the runtime's fixed-capacity shared KV buffer contract.
    pub fn slice_update(
        &mut self,
        src: mlxsys::mlx_array,
        update: mlxsys::mlx_array,
        start: &[i32],
        stop: &[i32],
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let strides = vec![1i32; start.len()];
        self.emit(|res, s| unsafe {
            mlxsys::mlx_slice_update(
                res,
                src,
                update,
                start.as_ptr(),
                start.len(),
                stop.as_ptr(),
                stop.len(),
                strides.as_ptr(),
                strides.len(),
                s,
            )
        })
    }

    /// Translate a control-flow body subgraph inline: bind its formal inputs to `inputs` (positional),
    /// dispatch every body node through the registry (implicit inputs fall through to the enclosing
    /// env), collect + return the arrays bound to the body's formal outputs, then restore the env.
    /// Faithful port of the C++ `TranslationContext::RunSubgraph`.
    pub fn run_subgraph(
        &mut self,
        sg: &SubgraphDesc,
        inputs: &[mlxsys::mlx_array],
    ) -> Result<Vec<mlxsys::mlx_array>, MlxError> {
        if inputs.len() != sg.input_names.len() {
            return Err(format!(
                "MLX RunSubgraph: input arity mismatch for body '{}'",
                sg.attr_name
            ));
        }
        // Names this body binds (formal inputs + every produced output); snapshot shadowed outer ones.
        let mut body_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        for nm in &sg.input_names {
            if !nm.is_empty() {
                body_names.insert(nm.clone());
            }
        }
        for node in &sg.nodes {
            for o in &node.outputs {
                if !o.name.is_empty() {
                    body_names.insert(o.name.clone());
                }
            }
        }
        let mut saved: HashMap<String, mlxsys::mlx_array> = HashMap::new();
        for nm in &body_names {
            if let Some(&a) = self.env.get(nm) {
                saved.insert(nm.clone(), a);
            }
        }
        // Bind formal inputs, then translate the body in topological order.
        for (i, nm) in sg.input_names.iter().enumerate() {
            if !nm.is_empty() {
                self.env.insert(nm.clone(), inputs[i]);
            }
        }
        for node in &sg.nodes {
            crate::registry::translate(self, node)?;
        }
        // Collect the body's formal outputs (before restoring the env).
        let mut outs = Vec::with_capacity(sg.output_names.len());
        for on in &sg.output_names {
            match self.env.get(on) {
                Some(&a) => outs.push(a),
                None => {
                    return Err(format!(
                        "MLX RunSubgraph: body '{}' did not produce output {on}",
                        sg.attr_name
                    ));
                }
            }
        }
        // Restore the enclosing scope.
        for nm in &body_names {
            self.env.remove(nm);
        }
        for (k, v) in saved {
            self.env.insert(k, v);
        }
        Ok(outs)
    }

    // ---- compiled decode support ---------------------------------------------------------------

    /// True while translating inside the compiled-decode closure trace (see [`Self::rope_dynamic`]).
    #[inline]
    pub fn rope_dynamic(&self) -> bool {
        self.rope_dynamic
    }

    /// True when the compiled-decode trace is for a fixed-capacity shared KV buffer session.
    #[inline]
    pub fn shared_kv(&self) -> bool {
        self.shared_kv
    }

    /// Register a shared-buffer KV `present` output for delta copy-out. `offset`/`count` are the new
    /// K/V rows written this step (axis 2 of `[B,kv,cap,hd]`) and `past_ctx_index` is the ctx index
    /// of the matching `past` input (whose buffer `present` should alias). In the EAGER path this
    /// records the live per-run delta (including the `past` pointer) consumed by `copy_out`. In the
    /// COMPILED trace (`rope_dynamic`) `offset` is still the capacity placeholder, so we only remember
    /// `(name, past_ctx_index)` on the plan; the real per-token offset/count/pointer are supplied at
    /// apply time by `try_compiled_decode`.
    pub fn record_kv_present(
        &mut self,
        out_name: &str,
        offset: i64,
        count: i64,
        past_ctx_index: usize,
    ) {
        if self.rope_dynamic {
            if !self.compiled_kv_present.iter().any(|(n, _)| n == out_name) {
                self.compiled_kv_present
                    .push((out_name.to_string(), past_ctx_index));
            }
        } else {
            let alias_ptr = self
                .read_ctx_input(past_ctx_index)
                .map(|(p, _, _)| p as usize)
                .unwrap_or(0);
            self.kv_deltas.insert(
                out_name.to_string(),
                DeltaWrite {
                    axis: 2,
                    offset,
                    count,
                    alias_ptr,
                },
            );
        }
    }

    /// The live `[1]` int32 `valid_past` array seeded into the compiled shared-buffer trace (see
    /// [`GQA_VALID_PAST_KEY`]). `None` outside the compiled shared-buffer path.
    #[inline]
    pub fn shared_valid_past(&self) -> Option<mlxsys::mlx_array> {
        self.env.get(GQA_VALID_PAST_KEY).copied()
    }

    /// Put this context into the compiled-decode/prefill TRACE mode (`rope_dynamic = true`), recording
    /// whether the session drives a shared KV buffer. Used by the unified core while building the
    /// closure body (replaces the old decode-only `new_trace`). `shape_keyed` marks a prefill trace
    /// (static shapes for the key); `valid_past` is the static valid-past for that key (only used when
    /// `shape_keyed` is set, to statically narrow the shared-buffer attention to the valid prefix).
    pub(crate) fn set_compiled_trace(
        &mut self,
        shared_kv: bool,
        shape_keyed: bool,
        valid_past: i32,
    ) {
        self.rope_dynamic = true;
        self.shared_kv = shared_kv;
        self.compiled_shape_keyed = shape_keyed;
        self.compiled_valid_past = valid_past;
    }

    /// True while translating inside a SHAPE-KEYED compiled (prefill) trace — the shared-buffer
    /// attention can be statically narrowed to the valid prefix. Only meaningful with `rope_dynamic`.
    #[inline]
    pub fn compiled_shape_keyed(&self) -> bool {
        self.compiled_shape_keyed
    }

    /// The static `valid_past` for a shape-keyed compiled (prefill) trace (see `compiled_shape_keyed`).
    #[inline]
    pub fn compiled_valid_past(&self) -> i32 {
        self.compiled_valid_past
    }

    /// Mark this context as a GENERAL compiled-subgraph trace (see [`Self::in_general_trace`]).
    pub(crate) fn set_general_trace(&mut self) {
        self.in_general_trace = true;
    }

    /// Record whether this translation is happening inside a `ShapeMode::ShapeKeyed` compiled
    /// subgraph (`general` or `prefill`) as opposed to a `Shapeless` (decode) one or plain eager
    /// translation. Call once, right after `TranslationContext::new`, from the closure's
    /// `CompiledConfig::shape_mode` (see [`Self::shape_keyed_compile`]).
    pub(crate) fn set_shape_keyed_compile(&mut self, shape_keyed: bool) {
        self.shape_keyed_compile = shape_keyed;
    }

    /// True while translating inside ANY shape-keyed compiled subgraph (general or prefill) — i.e. a
    /// multi-token, matrix-matrix compiled pass, never decode. See the field doc for the distinction
    /// from [`Self::compiled_shape_keyed`] (which is additionally scoped to the RoPE-as-data
    /// shared-buffer narrowing feature).
    #[inline]
    pub fn shape_keyed_compile(&self) -> bool {
        self.shape_keyed_compile
    }

    pub(crate) fn native_attention_decode(&self) -> bool {
        self.plan.native_attention_decode
    }

    /// Take the shared-buffer KV `present` outputs discovered during the compiled trace, handing them
    /// to the core so it can store them on the traced `CompiledSubgraph` slot. Leaves the list empty.
    pub(crate) fn take_compiled_kv_present(&mut self) -> Vec<(String, usize)> {
        std::mem::take(&mut self.compiled_kv_present)
    }

    /// Seed an env binding (the compiled-closure placeholders: dynamic ctx inputs + pre-sliced RoPE
    /// rows). `raw` is a borrowed handle owned by the trace arena.
    pub(crate) fn seed(&mut self, name: String, raw: mlxsys::mlx_array) {
        self.env.insert(name, raw);
    }

    /// The raw handle bound to `name` in the env, if any (compiled-closure output collection).
    pub(crate) fn env_get(&self, name: &str) -> Option<mlxsys::mlx_array> {
        self.env.get(name).copied()
    }

    /// Take ownership of this run's transient arena (handing it to the plan so the compiled graph's
    /// handles outlive the trace). Leaves the context arena empty.
    pub(crate) fn take_arena(&mut self) -> Vec<Array> {
        std::mem::take(&mut self.arena)
    }

    /// Translate every node into MLX ops (no eval) and return the cast external boundary outputs as
    /// a fresh vector — the body of a compiled closure trace (decode, prefill, or general). Unlike
    /// [`Self::execute`] this does NOT eval or copy out; the compiled closure defers evaluation to the
    /// single per-step eval. When `contiguous` is set each boundary output is additionally
    /// materialised to row-major contiguous (mirroring [`Self::finish_boundary`]) so a general
    /// static-shape copy-out is a straight typed memcpy; the RoPE/KV paths leave it unset to mirror
    /// the decode trace exactly.
    pub(crate) fn run_trace(
        &mut self,
        ext_outputs: &[OutRef],
        contiguous: bool,
    ) -> Result<VectorArray, MlxError> {
        let nodes = std::mem::take(&mut self.plan.nodes);
        let mut result = Ok(());
        for node in &nodes {
            if let Err(e) = crate::registry::translate(self, node) {
                result = Err(e);
                break;
            }
        }
        self.plan.nodes = nodes;
        result?;
        // Cast each boundary output to its ORT output dtype so the per-step copy-out is a straight
        // typed memcpy (mirrors `finish_boundary`), and append in the fixed closure output order.
        let mut res = VectorArray::new();
        for o in ext_outputs {
            let a = self
                .env_get(&o.name)
                .ok_or_else(|| format!("MLX: compiled trace missing output {}", o.name))?;
            let casted = self.astype(a, mlx_dtype_from_onnx(o.otype))?;
            let casted = if contiguous {
                self.contiguous(casted)?
            } else {
                casted
            };
            res.append(casted);
        }
        Ok(res)
    }

    /// The pre-sliced full-width cos/sin RoPE row placeholder for `cache_name`, reshaped to
    /// `[1,1,S,2*half]` for broadcast over `[B,H,S,2*half]`. Compiled-decode trace only.
    pub fn rope_row_full(
        &mut self,
        cache_name: &str,
        seq: i32,
        half: i32,
    ) -> Result<mlxsys::mlx_array, MlxError> {
        let key = rope_row_key(cache_name);
        let row = self
            .env
            .get(&key)
            .copied()
            .ok_or_else(|| format!("MLX: missing RoPE row placeholder for {cache_name}"))?;
        self.reshape(row, &[1, 1, seq, 2 * half])
    }

    /// Constant `[hd,hd]` matrix `M` such that `x @ M == rotate_half(x)` for non-interleaved RoPE,
    /// i.e. `rotate_half([x1,x2]) = [-x2, x1]` with `half = hd/2`. Built once (fp32) and cached on the
    /// plan. Lets the compiled decode graph do rotate-half with a matmul instead of a Slice (which
    /// shapeless `mlx_compile` cannot shape-infer). The caller casts it to the operand dtype.
    pub fn rotate_half_matrix(&mut self, hd: i32, half: i32) -> mlxsys::mlx_array {
        let key = format!("__rope_rotate_half_{hd}");
        if let Some(a) = self.plan.cache.get(&key) {
            return a.as_raw();
        }
        let n = (hd as usize) * (hd as usize);
        let mut m = vec![0.0f32; n];
        for i in 0..(half as usize) {
            let hd_u = hd as usize;
            let half_u = half as usize;
            m[(i + half_u) * hd_u + i] = -1.0; // col i (<half) picks -x[i+half]
            m[i * hd_u + (i + half_u)] = 1.0; // col i+half picks  x[i]
        }
        let shp = [hd, hd];
        let arr = Array::from_data(
            m.as_ptr() as *const c_void,
            &shp,
            mlxsys::mlx_dtype__MLX_FLOAT32,
        );
        self.cache_put(key, arr)
    }

    /// Translate every node, cast+collect boundary outputs, one `mlx_eval`, copy each output back
    /// across the ORT boundary. Faithful port of `ExecuteEager`.
    pub fn execute(&mut self) -> Result<(), MlxError> {
        let nodes = std::mem::take(&mut self.plan.nodes);
        let mut result = Ok(());
        let tr = crate::trace::tracer();
        {
            // Timing attribution: the eager per-node translation (graph build) phase.
            let _phase = tr.phase("translate");
            for node in &nodes {
                if let Err(e) = crate::registry::translate(self, node) {
                    result = Err(e);
                    break;
                }
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
                if o.external
                    && let Some(&a) = self.env.get(&o.name)
                {
                    let casted = self.astype(a, mlx_dtype_from_onnx(o.otype))?;
                    // Materialise to row-major contiguous HERE (once, at the boundary) so the
                    // flat copy_out memcpy is valid. Intermediate view ops (transpose/slice/
                    // expand/split) therefore stay zero-copy strided views that MLX folds into
                    // consuming kernels; only actual subgraph outputs pay a contiguous copy.
                    let casted = self.contiguous(casted)?;
                    outs.append(casted);
                    ext.push((o.clone(), casted));
                }
                for &(_, a, _) in &self.input_writebacks {
                    outs.append(a);
                }
            }
        }
        // The single synchronous `mlx_eval` boundary: with tracing on this is wrapped
        // in the `mlx.eval` (cat "gpu") span, whose CPU wall time is the GPU-inclusive
        // time of the whole fused subgraph (MLX blocks here until the GPU work lands).
        // Sample GPU-memory counters just before and after so the curve shows the eval.
        let tr = crate::trace::tracer();
        tr.sample_gpu_counters();
        let eval_t0 = if tr.active() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        {
            // Wrap the FIRST eval in a Metal GPU capture when requested (one-shot; the
            // guard stops the capture on drop). `None`/near-zero cost otherwise.
            let _cap = tr.begin_gpu_capture();
            let _eval = tr.eval_region();
            mlx::eval(&outs)?;
        }
        if let Some(t0) = eval_t0 {
            // Timing attribution: the synchronous GPU-inclusive eval phase (summary).
            tr.record_phase("eval", t0.elapsed());
        }
        tr.sample_gpu_counters();
        for (o, a) in &ext {
            self.copy_out(o, *a)?;
        }
        for &(dst, a, bytes) in &self.input_writebacks {
            let arr = std::mem::ManuallyDrop::new(Array::from_raw(a));
            unsafe {
                std::ptr::copy_nonoverlapping(arr.data_bytes(), dst as *mut u8, bytes);
            }
        }
        Ok(())
    }

    /// Create the ORT output tensor with the MLX result shape and memcpy on unified memory. For a
    /// shared-buffer KV `present` output only the newly written rows are copied (delta write).
    fn copy_out(&self, o: &OutRef, a: mlxsys::mlx_array) -> Result<(), MlxError> {
        let delta = self.kv_deltas.get(&o.name).copied();
        copy_out_raw_delta(self.ort_api, self.kctx, o, a, delta)
    }
}

/// Read a ctx input's `(data ptr, shape, dtype)` directly from the kernel context (no
/// `TranslationContext` needed — used by the compiled-decode path).
pub(crate) fn read_ctx_input_raw(
    ort_api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
    index: usize,
) -> Result<(*const c_void, Vec<i64>, ort::ONNXTensorElementDataType), MlxError> {
    unsafe {
        let api = &*ort_api;
        let mut val: *const ort::OrtValue = std::ptr::null();
        let st = (api.KernelContext_GetInput.unwrap())(kctx, index, &mut val);
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

/// Create the ORT output tensor with the MLX result shape and memcpy on unified memory; when
/// `delta` is `Some` only the `count` rows at `offset` along `axis` are memcpy'd back (shared-buffer
/// `axis` are memcpy'd back (shared-buffer KV `present`: the rest of the buffer already holds the
/// correct `past` rows because `present` aliases `past` in ORT memory). The ORT output tensor is
/// still created at the MLX array's FULL shape so ORT sees the whole `[B,kv,cap,hd]` present.
pub(crate) fn copy_out_raw_delta(
    ort_api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
    o: &OutRef,
    a: mlxsys::mlx_array,
    delta: Option<DeltaWrite>,
) -> Result<(), MlxError> {
    let arr = std::mem::ManuallyDrop::new(Array::from_raw(a)); // borrow, do not free
    let shape = arr.shape();
    let count: usize = shape.iter().map(|&d| d as usize).product();
    let itemsize = arr.itemsize();
    unsafe {
        let api = &*ort_api;
        let mut out: *mut ort::OrtValue = std::ptr::null_mut();
        let st = (api.KernelContext_GetOutput.unwrap())(
            kctx,
            o.ctx_index,
            shape.as_ptr(),
            shape.len(),
            &mut out,
        );
        if !st.is_null() || out.is_null() {
            return Err(format!(
                "MLX: KernelContext_GetOutput({}) failed",
                o.ctx_index
            ));
        }
        let mut dst: *mut c_void = std::ptr::null_mut();
        (api.GetTensorMutableData.unwrap())(out, &mut dst);
        let src = arr.data_bytes();
        if src.is_null() || dst.is_null() {
            return Ok(());
        }
        // Memory + timing view: record whether this copy-out took the delta (new-rows-only) or full
        // path, the bytes moved, and the wall time. Gated so a traced-off run pays one atomic load.
        let tr = crate::trace::tracer();
        let t0 = if tr.active() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut was_delta = false;
        let mut moved_bytes: u64 = (count * itemsize) as u64;
        match delta {
            // Partial copy: only the newly written rows along `axis`, and ONLY when `present` really
            // aliases `past` in ORT memory (dst pointer matches the recorded `past` address). `past`
            // supplies every other row, so they are already correct. For a row-major [outer.., axis,
            // inner..] layout each of the `outer` slabs contributes one contiguous run of
            // `count*inner` elements at `(outer_idx*axis_len + offset)*inner`.
            Some(d)
                if d.count > 0
                    && d.axis < shape.len()
                    && d.alias_ptr != 0
                    && d.alias_ptr == dst as usize =>
            {
                let axis_len = shape[d.axis].max(0) as usize;
                let offset = d.offset.max(0) as usize;
                let n_rows = (d.count as usize).min(axis_len.saturating_sub(offset));
                if n_rows == 0 {
                    return Ok(());
                }
                let outer: usize = shape[..d.axis]
                    .iter()
                    .map(|&x| x as usize)
                    .product::<usize>()
                    .max(1);
                let inner: usize = shape[d.axis + 1..]
                    .iter()
                    .map(|&x| x as usize)
                    .product::<usize>()
                    .max(1);
                let run = n_rows * inner * itemsize; // bytes per outer slab
                let stride = axis_len * inner * itemsize; // bytes between slabs
                let start = offset * inner * itemsize; // byte offset of first row within a slab
                for o_idx in 0..outer {
                    let byte_off = o_idx * stride + start;
                    std::ptr::copy_nonoverlapping(
                        src.add(byte_off),
                        (dst as *mut u8).add(byte_off),
                        run,
                    );
                }
                was_delta = true;
                moved_bytes = (outer * run) as u64;
            }
            // No delta, or `present` did NOT alias `past`: full contiguous copy (growing path is
            // bit-for-bit unchanged; also the safe fallback whenever aliasing can't be confirmed).
            _ => {
                std::ptr::copy_nonoverlapping(src, dst as *mut u8, count * itemsize);
            }
        }
        if let Some(t0) = t0 {
            tr.record_copyout(was_delta, moved_bytes, t0.elapsed());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(name: &str) -> TensorRef {
        TensorRef {
            name: name.to_string(),
            source: Src::Intermediate,
            ctx_index: 0,
            constant: false,
            shape_const: false,
            init: None,
        }
    }

    fn ctx_input(name: &str, ctx_index: usize) -> TensorRef {
        TensorRef {
            name: name.to_string(),
            source: Src::CtxInput,
            ctx_index,
            constant: false,
            shape_const: false,
            init: None,
        }
    }

    fn output(name: &str) -> OutRef {
        OutRef {
            name: name.to_string(),
            external: false,
            ctx_index: 0,
            otype: ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        }
    }

    fn node(op_type: &str, inputs: &[&str], output_name: &str) -> NodeDesc {
        let mut node = NodeDesc::new(
            op_type.to_string(),
            String::new(),
            17,
            crate::registry::CompileShapeSafety::Shapeless,
        );
        node.inputs = inputs.iter().map(|name| input(name)).collect();
        node.outputs = vec![output(output_name)];
        node
    }

    fn optimized_whisper_nodes(hidden_size: i64) -> Vec<NodeDesc> {
        let mut nodes = vec![
            node("Add", &["start", "length"], "limit"),
            node("Range", &["start", "limit", "step"], "range"),
            node("Unsqueeze", &["range", "axes"], "positions"),
            node("Concat", &["one", "batch"], "repeats"),
            node("Tile", &["positions", "repeats"], "tiled"),
            node("MultiHeadAttention", &["q", "k", "v"], "attention"),
        ];
        let mut vocab = node("MatMulNBits", &["hidden", "w", "s", "z"], "logits");
        vocab.ints.insert("bits".to_string(), 8);
        vocab.ints.insert("K".to_string(), hidden_size);
        vocab.ints.insert("N".to_string(), 51_865);
        nodes.push(vocab);
        while nodes.len() <= 32 {
            let index = nodes.len();
            nodes.push(node("Add", &["a", "b"], &format!("filler_{index}")));
        }
        nodes
    }

    #[test]
    fn optimized_decoder_position_graph_gets_dedicated_stream() {
        let tiny = Plan::new(optimized_whisper_nodes(384));
        assert!(!tiny.native_attention_decode);
        assert!(tiny.dedicated_decode_stream);

        let base = Plan::new(optimized_whisper_nodes(512));
        assert!(!base.native_attention_decode);
        assert!(base.dedicated_decode_stream);

        let mut ambiguous = optimized_whisper_nodes(384);
        ambiguous.push(node("Range", &["x", "y", "z"], "another_range"));
        assert!(!Plan::new(ambiguous).dedicated_decode_stream);

        let no_attention = optimized_whisper_nodes(384)
            .into_iter()
            .filter(|node| node.op_type != "MultiHeadAttention")
            .collect();
        assert!(!Plan::new(no_attention).dedicated_decode_stream);

        let mut native_attention = optimized_whisper_nodes(384);
        native_attention
            .iter_mut()
            .find(|node| node.op_type == "MultiHeadAttention")
            .unwrap()
            .op_type = "Attention".to_string();
        assert!(Plan::new(native_attention).dedicated_decode_stream);
    }

    #[test]
    fn stable_cross_inputs_require_a_boundary_kv_pair() {
        let mut cross = node("MultiHeadAttention", &["q", "k", "v"], "attention");
        cross.inputs[1] = ctx_input("k", 4);
        cross.inputs[2] = ctx_input("v", 5);
        let mut self_attention = node(
            "MultiHeadAttention",
            &["q", "k", "v", "bias", "mask", "rel", "past_k", "past_v"],
            "self_attention",
        );
        self_attention.inputs[6] = ctx_input("past_k", 6);
        self_attention.inputs[7] = ctx_input("past_v", 7);
        assert_eq!(
            Plan::new(vec![self_attention, cross]).stable_cross_ctx_indices,
            vec![4, 5]
        );

        let mut partial = node("MultiHeadAttention", &["q", "k", "v"], "attention");
        partial.inputs[1] = ctx_input("k", 4);
        assert!(Plan::new(vec![partial]).stable_cross_ctx_indices.is_empty());

        let mut native = node("Attention", &["q", "k", "v"], "attention");
        native.inputs[1] = ctx_input("k", 8);
        native.inputs[2] = ctx_input("v", 9);
        let mut native_self = node(
            "Attention",
            &["q", "k", "v", "mask", "past_k", "past_v"],
            "self_attention",
        );
        native_self.inputs[4] = ctx_input("past_k", 10);
        native_self.inputs[5] = ctx_input("past_v", 11);
        assert_eq!(
            Plan::new(vec![native_self, native]).stable_cross_ctx_indices,
            vec![8, 9]
        );

        let mut contrib = node("Attention", &["q", "w", "bias"], "attention");
        contrib.domain = "com.microsoft".to_string();
        contrib.inputs[1] = ctx_input("w", 8);
        contrib.inputs[2] = ctx_input("bias", 9);
        assert!(Plan::new(vec![contrib]).stable_cross_ctx_indices.is_empty());
    }
}
