//! `MlxEp` — our `OrtEp` C-ABI vtable, generalized from the single-Add spike into a real engine:
//!
//!   * GetCapability claims nodes via the registry claim predicates and groups them into maximal
//!     convex connected clusters (`build_convex_clusters`, a faithful port of ep.cc's union-find +
//!     reachability-bitset algorithm — non-convex fusion creates a cycle ORT rejects).
//!   * Compile extracts each node's `NodeDesc` (op_type/domain/since_version + attributes + I/O
//!     tensor refs) and builds one `Plan` per fused subgraph, owned by its `OrtNodeComputeInfo`.
//!   * Compute (RunPlan) resolves subgraph inputs from the KernelContext, runs each node's handler
//!     in topo order, one `mlx_eval`, and writes each subgraph output.
//!
//! Raw `unsafe`/FFI is confined to this boundary layer + `sys`; the ops use the safe `Array` wrappers.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;

use crate::engine::{Plan, Slot, TranslationContext, is_separate_qkv_attention_op};
use crate::factory::ORT_API_VERSION;
use crate::mlx::Stream;
use crate::ort_graph::{NodeMetadata, OrtGraphSnapshot};
use crate::partition::{
    build_contiguous_clusters, build_convex_clusters, infer_layer_boundary_values,
    split_annotated_layer_clusters,
};
use crate::plan_builder;
use crate::registry::{CompilePartitionClass, NodeView, claimable};
use crate::sys::{mlx, ort};

fn use_dedicated_decode_stream(enabled: bool, seq_len: Option<i32>) -> bool {
    enabled && seq_len == Some(1)
}

fn reset_stable_cross_caches(plan: &mut Plan, generation_key: Option<usize>) {
    for slot in [Slot::Decode, Slot::Prefill] {
        let compiled = slot.get_mut(plan);
        compiled.stable_cross_inputs.clear();
        compiled.stable_generation_key = generation_key;
    }
}

#[repr(C)]
pub struct MlxEp {
    base: ort::OrtEp,
    ort_api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    name: CString,
    stream: Stream,
    /// The MLX CPU stream, used only by subgraphs that carry float64 (MLX has no Metal fp64 path).
    /// Created up front and shared: `mlx_default_cpu_stream_new` returns MLX's default CPU stream,
    /// so this is a handle, not a second execution context.
    cpu_stream: Stream,
}

impl MlxEp {
    pub fn new(
        ort_api: *const ort::OrtApi,
        ep_api: *const ort::OrtEpApi,
        name: &CStr,
        _logger: *const ort::OrtLogger,
    ) -> Box<MlxEp> {
        let mut base: ort::OrtEp = unsafe { std::mem::zeroed() };
        base.ort_version_supported = ORT_API_VERSION;
        base.GetName = Some(get_name);
        base.GetCapability = Some(get_capability);
        base.Compile = Some(compile);
        base.ReleaseNodeComputeInfos = Some(release_node_compute_infos);
        base.GetDefaultMemoryDevice = Some(get_default_memory_device);
        Box::new(MlxEp {
            base,
            ort_api,
            ep_api,
            name: name.to_owned(),
            stream: Stream::new_default_gpu(),
            cpu_stream: Stream::new_default_cpu(),
        })
    }

    pub fn as_ptr(self: Box<Self>) -> *mut ort::OrtEp {
        Box::into_raw(self) as *mut ort::OrtEp
    }
}

// The per-EP mlx stream is now owned by the `Stream` RAII wrapper, freed exactly once when ORT
// calls ReleaseEp (which drops our Box<MlxEp>). No manual free / no explicit Drop needed.

// On EP teardown, flush the (env-gated) trace. The tracer's collector accumulates
// across all sessions in the process, so each teardown rewrites the full cumulative
// trace; the last one leaves the complete file on disk (no-op when tracing is off).
impl Drop for MlxEp {
    fn drop(&mut self) {
        let tr = crate::trace::tracer();
        // Compact agent-friendly "slowest ops" summary (stderr + trace metadata) before
        // the JSON is written, so the ranking is embedded in the exported trace too.
        tr.log_slowest_ops();
        // The at-a-glance session digest (claim rate, per-path Compute breakdown, memory movement,
        // time attribution). Printed to stderr when tracing OR the verbose flag is on; also embedded
        // in the JSON trace. No-op / no stderr when neither is set.
        tr.log_summary();
        tr.export();
    }
}

#[inline]
unsafe fn this(p: *const ort::OrtEp) -> *const MlxEp {
    p as *const MlxEp
}

unsafe extern "C" fn get_name(p: *const ort::OrtEp) -> *const c_char {
    unsafe { (*this(p)).name.as_ptr() }
}

unsafe extern "C" fn get_default_memory_device(
    _p: *const ort::OrtEp,
    device: *mut *const ort::OrtMemoryDevice,
) -> *mut ort::OrtStatus {
    unsafe {
        // I/O stays on the CPU allocator (unified memory); no device memory advertised.
        *device = ptr::null();
        ptr::null_mut()
    }
}

// ---------------------------------------------------------------------------
// GetCapability: claim via registry + convex clustering.
// ---------------------------------------------------------------------------

unsafe extern "C" fn get_capability(
    p: *mut ort::OrtEp,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> *mut ort::OrtStatus {
    let api = unsafe { (*this(p)).ort_api };
    unsafe {
        crate::guard_ffi_status(api, "get_capability", || {
            get_capability_impl(p, graph, support)
        })
    }
}

unsafe fn get_capability_impl(
    p: *mut ort::OrtEp,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> *mut ort::OrtStatus {
    unsafe {
        let ep = &*this(p);
        let api = &*ep.ort_api;
        let ep_api = &*ep.ep_api;
        let snapshot = match OrtGraphSnapshot::from_ort(api, graph) {
            Ok(snapshot) => snapshot,
            Err(error) => return ep_fail_status(api, error),
        };
        let node_order = match snapshot.ir.topological_order() {
            Ok(order) => order,
            Err(error) => {
                return ep_fail_status(
                    api,
                    format!("optimized graph IR has no topological order: {error}"),
                );
            }
        };
        if node_order.is_empty() {
            return ptr::null_mut();
        }
        // Registry predicates still inspect attributes and constant initializer values through ORT,
        // but every topology decision below uses the owned snapshot keyed by stable ORT node IDs.
        let mut raw_nodes: Vec<*const ort::OrtNode> = vec![ptr::null(); node_order.len()];
        let st = (api.Graph_GetNodes.unwrap())(graph, raw_nodes.as_mut_ptr(), raw_nodes.len());
        if !st.is_null() {
            return st;
        }
        let mut raw_nodes_by_id = HashMap::with_capacity(raw_nodes.len());
        for node in raw_nodes {
            let mut ort_node_id = 0usize;
            let st = (api.Node_GetId.unwrap())(node, &mut ort_node_id);
            if !st.is_null() {
                return st;
            }
            let Some(&node_id) = snapshot.node_by_ort_id.get(&ort_node_id) else {
                return ep_fail_status(
                    api,
                    format!("optimized graph snapshot is missing ORT node ID {ort_node_id}"),
                );
            };
            raw_nodes_by_id.insert(node_id, node);
        }

        // Control-flow support: ORT partitions bottom-up, presenting a CF node's body subgraph to
        // GetCapability BEFORE the parent graph that owns the node. If this graph is the body of a
        // control-flow node we can translate WHOLE, decline ALL body nodes here (so ORT leaves the body
        // intact); we then claim the CF node itself at the parent level and translate its body in Compile
        // via Node_GetSubgraphs.
        //
        // If the parent CF op is one we CANNOT translate wholesale, we must ALSO decline every body node:
        // ORT would otherwise fuse the claimed body nodes into a node in the EP's private domain and
        // splice it back into the subgraph, but nested subgraphs carry no opset import for that domain,
        // yielding an INVALID_GRAPH ("No opset import for domain 'MLXExecutionProvider'") at session
        // creation (e.g. the Loop that function-inlined SequenceMap expands to). Such body ops simply
        // run on ORT's CPU control flow instead.
        let mut in_cf_body = false;
        if let Some(get_parent) = api.Graph_GetParentNode {
            let mut parent: *const ort::OrtNode = ptr::null();
            let st = get_parent(graph, &mut parent);
            if st.is_null() && !parent.is_null() {
                in_cf_body = true;
            } else if !st.is_null() {
                release_status(api, st);
            }
        }

        // Foundry's Q8 Whisper decoder has runtime Range/Tile nodes for position IDs. Keep the
        // complete decoder on MLX; the translator handles those two dynamic shape nodes inside the
        // EP so they do not create ORT CPU islands.
        let native_q8_attention_graph = node_order.len() > 32
            && node_order.iter().any(|&node| {
                let metadata = &snapshot.nodes[&node];
                is_separate_qkv_attention_op(&metadata.domain, &metadata.op_type)
            })
            && node_order.iter().any(|&node| {
                let raw_node = raw_nodes_by_id[&node];
                let view = NodeView::new(ep.ort_api, raw_node);
                // Bits is an operator attribute, intentionally left to the registry's ORT view.
                // The graph identity and topology remain owned by the snapshot.
                view.op_type() == "MatMulNBits" && view.int_attr("bits", 4) == 8
            })
            && node_order
                .iter()
                .filter(|&&node| {
                    let view = NodeView::new(ep.ort_api, raw_nodes_by_id[&node]);
                    view.op_type() == "Range"
                        && view.read_const_scalar_f64(2) == Some(1.0)
                        && (0..3).any(|index| view.read_const_scalar_f64(index).is_none())
                })
                .count()
                == 1
            && node_order
                .iter()
                .filter(|&&node| {
                    let view = NodeView::new(ep.ort_api, raw_nodes_by_id[&node]);
                    view.op_type() == "Tile" && !view.is_const_int64(1)
                })
                .count()
                == 1;

        let mixed_vision_quant_graph = node_order.iter().any(|&node| {
            let metadata = &snapshot.nodes[&node];
            metadata.domain == "com.microsoft" && metadata.op_type == "PackedMultiHeadAttention"
        }) && node_order.iter().any(|&node| {
            let metadata = &snapshot.nodes[&node];
            metadata.domain == "com.microsoft" && metadata.op_type == "MatMulNBits"
        });

        // Which nodes can MLX translate exactly (registry claim predicate).
        let supported: Vec<bool> = node_order
            .iter()
            .map(|&node_id| {
                if in_cf_body {
                    return false;
                }
                let view = NodeView::new(ep.ort_api, raw_nodes_by_id[&node_id]);
                if native_q8_attention_graph && matches!(view.op_type().as_str(), "Range" | "Tile")
                {
                    return true;
                }
                claimable(&view)
            })
            .collect();

        // fp64 colour: which claimed nodes carry a float64 tensor. Used to keep fp64 work in its own
        // cluster (and thus on its own MLX CPU stream) — see `build_convex_clusters`.
        let float64: Vec<bool> = node_order
            .iter()
            .map(|&node| snapshot_node_uses_float64(&snapshot, node))
            .collect();
        // In a decoder graph, colour each node by the registry's strongest allowed route:
        // shapeless, shape-keyed-only, or eager-only. A stricter node cannot poison compilation for
        // a more capable neighbour. Non-decoder graphs use only the general route, so splitting
        // them would add boundaries for no benefit.
        let attention_anchors = node_order
            .iter()
            .copied()
            .filter(|node| is_decoder_attention_anchor(&snapshot.nodes[node]))
            .collect::<Vec<_>>();
        let decoder_graph = !attention_anchors.is_empty();
        let compile_class: Vec<CompilePartitionClass> = node_order
            .iter()
            .map(|&node| {
                let view = NodeView::new(ep.ort_api, raw_nodes_by_id[&node]);
                if decoder_graph {
                    crate::registry::compile_shape_safety(&view).partition_class()
                } else {
                    CompilePartitionClass::Shapeless
                }
            })
            .collect();

        // Claiming view: build the per-op fallback reasons for the declined nodes (only when
        // observability is active, so this extra FFI never touches the traced-off fast path). The
        // legacy `ONNXRUNTIME_EP_MLX_CLAIM_DEBUG` env still forces the raw stderr dump.
        let tr = crate::trace::tracer();
        let mut rejected: Vec<(String, usize, String, Vec<String>)> = Vec::new();
        if tr.active() || std::env::var_os("ONNXRUNTIME_EP_MLX_CLAIM_DEBUG").is_some() {
            use std::collections::BTreeMap;
            // Per op-type: (count, first-reason, up to a few node names for locating them).
            let mut acc: BTreeMap<String, (usize, String, Vec<String>)> = BTreeMap::new();
            for (&node, &ok) in node_order.iter().zip(supported.iter()) {
                if !ok {
                    let view = NodeView::new(ep.ort_api, raw_nodes_by_id[&node]);
                    // Keyed by the domain-qualified name: `Attention` exists in
                    // both the default domain and `com.microsoft`, and merging
                    // them would report one count for two different ops.
                    let e = acc
                        .entry(crate::registry::qualified_op_name(&view))
                        .or_insert((0, String::new(), Vec::new()));
                    e.0 += 1;
                    if e.1.is_empty() {
                        e.1 = if in_cf_body {
                            "inside a control-flow subgraph body — claimed as part of the parent \
                             If/Loop/Scan, not individually"
                                .to_string()
                        } else {
                            crate::registry::claim_decision(&view)
                                .err()
                                .map(|c| c.into_owned())
                                .unwrap_or_else(|| "declined (no reason reported)".to_string())
                        };
                    }
                    if e.2.len() < 16 {
                        let nm = view.name();
                        if !nm.is_empty() {
                            e.2.push(nm);
                        }
                    }
                }
            }
            rejected = acc
                .into_iter()
                .map(|(op, (n, why, names))| (op, n, why, names))
                .collect();
            rejected.sort_by_key(|a| std::cmp::Reverse(a.1));
            if std::env::var_os("ONNXRUNTIME_EP_MLX_CLAIM_DEBUG").is_some() {
                for (op, n, why, names) in &rejected {
                    log::debug!("unclaimed {op} x{n} ({why}): {names:?}");
                }
            }
        }

        // This vision export contains many dynamic control-flow/shape branches. ORT's fusion API
        // accounts for implicit control dependencies that are not exposed by the tensor-name graph
        // used by `build_convex_clusters`, so its maximal clusters can be rejected as non-convex.
        // Maximal supported intervals in ORT's topological node order are conservatively convex:
        // every node that could lie between two members is included, while still avoiding the
        // thousands of singleton partitions previously used for this graph.
        let clusters = if mixed_vision_quant_graph {
            build_contiguous_clusters(&node_order, &supported, &float64, &compile_class)
        } else {
            build_convex_clusters(
                &snapshot.ir,
                &node_order,
                &supported,
                &float64,
                &compile_class,
            )
        };
        let layer_boundary_outputs = if in_cf_body {
            HashSet::new()
        } else {
            match graph_metadata_value(api, graph, c"onnxruntime_ep_mlx.layer_boundary_outputs") {
                Ok(Some(value)) => match serde_json::from_str::<Vec<String>>(&value) {
                    Ok(outputs) => outputs
                        .into_iter()
                        .filter_map(|name| snapshot.value_by_name.get(&name).copied())
                        .filter(|&value| snapshot.ir.value(value).producer.is_some())
                        .collect(),
                    Err(error) => {
                        log::warn!(
                            "ignoring invalid onnxruntime_ep_mlx.layer_boundary_outputs metadata: \
                             {error}"
                        );
                        HashSet::new()
                    }
                },
                Ok(None) => {
                    infer_layer_boundary_values(&snapshot.ir, &node_order, &attention_anchors)
                }
                Err(st) => return st,
            }
        };
        let layer_count = layer_boundary_outputs.len();
        let layer_partition_span = select_layer_partition_span(
            std::env::var("ONNXRUNTIME_EP_MLX_LAYER_PARTITIONS")
                .ok()
                .as_deref(),
            layer_count,
        );
        let clusters = match layer_partition_span {
            Some(span) => split_annotated_layer_clusters(
                &snapshot.ir,
                clusters,
                &layer_boundary_outputs,
                span,
            ),
            None => clusters,
        };

        let add_fuse = ep_api.EpGraphSupportInfo_AddNodesToFuse.unwrap();
        let mut claimed = 0usize;
        for cluster in &clusters {
            let group: Vec<*const ort::OrtNode> =
                cluster.iter().map(|node| raw_nodes_by_id[node]).collect();
            let mut opts: ort::OrtNodeFusionOptions = std::mem::zeroed();
            opts.ort_version_supported = ORT_API_VERSION;
            // Initializers are copied into Plan-owned storage during Compile, so they do not need
            // to remain runtime fused-node inputs.
            opts.drop_constant_initializers = true;
            let st = add_fuse(support, group.as_ptr(), group.len(), &opts);
            if !st.is_null() {
                return st;
            }
            claimed += cluster.len();
        }
        // Claiming view: claimed/total nodes, fused-subgraph count (fragmentation signal), and the
        // per-op fallback reasons — structured spans/counters + the session summary (near-zero cost
        // and no stderr spam when tracing is off). Replaces the old unconditional eprintlns.
        tr.record_claim(claimed, node_order.len(), clusters.len(), &rejected);
        ptr::null_mut()
    }
}

/// Select a layer span from an explicit override or the graph size. Auto mode keeps roughly seven
/// partitions while capping each at eight layers; smaller decoders remain whole to avoid needless
/// token-at-a-time boundary overhead.
fn select_layer_partition_span(value: Option<&str>, layer_count: usize) -> Option<usize> {
    match value.map(str::trim) {
        Some("0" | "off" | "false") => None,
        Some("" | "auto") | None => {
            (layer_count >= 23).then(|| layer_count.div_ceil(7).clamp(4, 8))
        }
        Some(value) => value.parse::<usize>().ok().filter(|&span| span > 0),
    }
}

fn is_decoder_attention_anchor(node: &NodeMetadata) -> bool {
    if node.domain == "com.microsoft" && node.op_type == "PagedAttention" {
        return true;
    }
    let has_present_cache = node.output_slots.get(1).is_some_and(Option::is_some)
        && node.output_slots.get(2).is_some_and(Option::is_some);
    has_present_cache
        && ((node.domain == "com.microsoft"
            && matches!(
                node.op_type.as_str(),
                "GroupQueryAttention" | "MultiHeadAttention"
            ))
            || (node.domain.is_empty() && node.op_type == "Attention"))
}

fn snapshot_node_uses_float64(snapshot: &OrtGraphSnapshot, node: onnx_runtime_ir::NodeId) -> bool {
    snapshot.nodes[&node]
        .input_slots
        .iter()
        .chain(snapshot.nodes[&node].output_slots.iter())
        .flatten()
        .any(|value| {
            snapshot.values[value]
                .tensor
                .as_ref()
                .and_then(|tensor| tensor.dtype)
                == Some(onnx_runtime_ir::DataType::Float64)
        })
}

unsafe fn ep_fail_status(api: &ort::OrtApi, message: impl AsRef<str>) -> *mut ort::OrtStatus {
    let message = CString::new(message.as_ref())
        .unwrap_or_else(|_| CString::new("MLX capability inspection failed").unwrap());
    unsafe { (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, message.as_ptr()) }
}

#[cfg(test)]
mod layer_partition_tests {
    use super::select_layer_partition_span;

    #[test]
    fn auto_layer_partition_span_scales_with_decoder_size() {
        assert_eq!(select_layer_partition_span(None, 16), None);
        assert_eq!(select_layer_partition_span(None, 23), Some(4));
        assert_eq!(select_layer_partition_span(None, 24), Some(4));
        assert_eq!(select_layer_partition_span(None, 52), Some(8));
        assert_eq!(select_layer_partition_span(Some("auto"), 80), Some(8));
    }

    #[test]
    fn layer_partition_span_accepts_overrides() {
        assert_eq!(select_layer_partition_span(Some("0"), 52), None);
        assert_eq!(select_layer_partition_span(Some("off"), 52), None);
        assert_eq!(select_layer_partition_span(Some("5"), 52), Some(5));
        assert_eq!(select_layer_partition_span(Some("invalid"), 52), None);
    }
}

/// Release a non-null `OrtStatus` returned on an error / not-found path (the OrtApi allocates a
/// status object the caller owns even for benign "not found" / "buffer too small" results).
#[inline]
unsafe fn release_status(api: &ort::OrtApi, st: *mut ort::OrtStatus) {
    unsafe {
        if !st.is_null() {
            (api.ReleaseStatus.unwrap())(st);
        }
    }
}

unsafe fn graph_metadata_value(
    api: &ort::OrtApi,
    graph: *const ort::OrtGraph,
    key: &CStr,
) -> Result<Option<String>, *mut ort::OrtStatus> {
    unsafe {
        let mut metadata: *mut ort::OrtModelMetadata = ptr::null_mut();
        let st = (api.Graph_GetModelMetadata.unwrap())(graph, &mut metadata);
        if !st.is_null() {
            return Err(st);
        }

        let mut allocator: *mut ort::OrtAllocator = ptr::null_mut();
        let st = (api.GetAllocatorWithDefaultOptions.unwrap())(&mut allocator);
        if !st.is_null() {
            (api.ReleaseModelMetadata.unwrap())(metadata);
            return Err(st);
        }

        let mut value: *mut c_char = ptr::null_mut();
        let st = (api.ModelMetadataLookupCustomMetadataMap.unwrap())(
            metadata,
            allocator,
            key.as_ptr(),
            &mut value,
        );
        if !st.is_null() {
            (api.ReleaseModelMetadata.unwrap())(metadata);
            return Err(st);
        }

        let result = if value.is_null() {
            None
        } else {
            let owned = CStr::from_ptr(value).to_string_lossy().into_owned();
            ((*allocator).Free.unwrap())(allocator, value.cast());
            Some(owned)
        };
        (api.ReleaseModelMetadata.unwrap())(metadata);
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Compile: build one Plan (topo-ordered NodeDescs) per fused subgraph.
// ---------------------------------------------------------------------------

unsafe extern "C" fn compile(
    p: *mut ort::OrtEp,
    graphs: *mut *const ort::OrtGraph,
    fused_nodes: *mut *const ort::OrtNode,
    count: usize,
    node_compute_infos: *mut *mut ort::OrtNodeComputeInfo,
    ep_context_nodes: *mut *mut ort::OrtNode,
) -> *mut ort::OrtStatus {
    let api = unsafe { (*this(p)).ort_api };
    unsafe {
        crate::guard_ffi_status(api, "compile", || {
            compile_impl(
                p,
                graphs,
                fused_nodes,
                count,
                node_compute_infos,
                ep_context_nodes,
            )
        })
    }
}

unsafe fn compile_impl(
    p: *mut ort::OrtEp,
    graphs: *mut *const ort::OrtGraph,
    fused_nodes: *mut *const ort::OrtNode,
    count: usize,
    node_compute_infos: *mut *mut ort::OrtNodeComputeInfo,
    _ep_context_nodes: *mut *mut ort::OrtNode,
) -> *mut ort::OrtStatus {
    unsafe {
        let ep = &*this(p);
        let api = &*ep.ort_api;

        for i in 0..count {
            let graph = *graphs.add(i);
            let fused_node = *fused_nodes.add(i);
            match plan_builder::build_plan(api, graph, fused_node) {
                Ok(plan) => {
                    // A compiled subgraph carries the stream it runs on: fp64 clusters get the MLX
                    // CPU stream, everything else keeps the GPU stream.
                    let stream = if plan.requires_cpu_stream {
                        ep.cpu_stream.as_raw()
                    } else {
                        ep.stream.as_raw()
                    };
                    let info = SubgraphComputeInfo::new(ep.ort_api, stream, plan);
                    *node_compute_infos.add(i) =
                        Box::into_raw(info) as *mut ort::OrtNodeComputeInfo;
                }
                Err(msg) => {
                    let c = CString::new(msg)
                        .unwrap_or_else(|_| CString::new("MLX compile error").unwrap());
                    return (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr());
                }
            }
        }
        ptr::null_mut()
    }
}

// ---------------------------------------------------------------------------
// Per-fused-subgraph compute info: owns the Plan, runs it through MLX.
// ---------------------------------------------------------------------------

#[repr(C)]
struct SubgraphComputeInfo {
    base: ort::OrtNodeComputeInfo,
    ort_api: *const ort::OrtApi,
    stream: mlx::mlx_stream,
    decode_stream: std::sync::Mutex<Option<Stream>>,
    // ORT permits concurrent Run() on one InferenceSession, and CreateState hands every Run the
    // SAME SubgraphComputeInfo. `plan` is mutated by Compute (the compiled-closure cache is filled
    // on cache-MISS; the eager translator writes intermediates), so it must be serialized to avoid
    // mutable aliasing / a data race. MLX drives a single default stream, so this node is serial
    // regardless — one session per thread remains the path to real cross-request parallelism.
    plan: std::sync::Mutex<Plan>,
    // First thread to run Compute "owns" this session's MLX stream. MLX 0.6.0 eval is thread-affine:
    // a foreign-thread eval aborts the HOST process. We detect a cross-thread Run here and return a
    // clean OrtStatus instead of letting MLX take the process down. Concurrency = session-per-thread.
    owner_thread: std::sync::Mutex<Option<std::thread::ThreadId>>,
}

impl SubgraphComputeInfo {
    fn new(
        ort_api: *const ort::OrtApi,
        stream: mlx::mlx_stream,
        plan: Plan,
    ) -> Box<SubgraphComputeInfo> {
        let mut base: ort::OrtNodeComputeInfo = unsafe { std::mem::zeroed() };
        base.ort_version_supported = ORT_API_VERSION;
        base.CreateState = Some(create_state);
        base.Compute = Some(compute);
        base.ReleaseState = Some(release_state);
        Box::new(SubgraphComputeInfo {
            base,
            ort_api,
            stream,
            decode_stream: std::sync::Mutex::new(None),
            plan: std::sync::Mutex::new(plan),
            owner_thread: std::sync::Mutex::new(None),
        })
    }
}

unsafe extern "C" fn create_state(
    this_ptr: *mut ort::OrtNodeComputeInfo,
    _compute_context: *mut ort::OrtNodeComputeContext,
    compute_state: *mut *mut c_void,
) -> *mut ort::OrtStatus {
    unsafe {
        *compute_state = this_ptr as *mut c_void;
        ptr::null_mut()
    }
}

unsafe extern "C" fn release_state(_this: *mut ort::OrtNodeComputeInfo, _state: *mut c_void) {}

unsafe extern "C" fn compute(
    this: *mut ort::OrtNodeComputeInfo,
    state: *mut c_void,
    kctx: *mut ort::OrtKernelContext,
) -> *mut ort::OrtStatus {
    let api = unsafe { (*(state as *const SubgraphComputeInfo)).ort_api };
    unsafe { crate::guard_ffi_status(api, "compute", || compute_impl(this, state, kctx)) }
}

unsafe fn compute_impl(
    _this: *mut ort::OrtNodeComputeInfo,
    state: *mut c_void,
    kctx: *mut ort::OrtKernelContext,
) -> *mut ort::OrtStatus {
    unsafe {
        let info = &*(state as *const SubgraphComputeInfo);
        let api = &*info.ort_api;

        // Thread-affinity guard: MLX 0.6.0 eval is bound to the thread that first drove this
        // session's stream. A Run from any other thread would abort the host process inside MLX, so
        // bind the owner on first Compute and reject a cross-thread Run with a clean OrtStatus.
        let cur_thread = std::thread::current().id();
        {
            let mut owner = info.owner_thread.lock().unwrap_or_else(|e| e.into_inner());
            match *owner {
                None => *owner = Some(cur_thread),
                Some(t) if t == cur_thread => {}
                Some(t) => {
                    let msg = format!(
                        "onnxruntime-mlx: this InferenceSession first ran on thread {t:?} but Run() \
                         was called from {cur_thread:?}. MLX eval is thread-affine — use one \
                         InferenceSession per thread for concurrent inference."
                    );
                    let c = CString::new(msg).unwrap_or_else(|_| {
                        CString::new("onnxruntime-mlx: cross-thread Run() is not supported")
                            .unwrap()
                    });
                    return (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr());
                }
            }
        }

        // Serialize per-subgraph state: ORT allows concurrent Run() on one session, but the
        // compiled-closure cache (plan.compiled/prefill/general) is mutated on cache-MISS and the
        // eager translator writes intermediates into `plan` — concurrent Compute on the same node
        // must not alias it. `plan_ptr` stays valid for the whole call while the guard is held.
        let mut plan_guard = info.plan.lock().unwrap_or_else(|e| e.into_inner());
        let plan_ptr: *mut Plan = &mut *plan_guard;
        let seq_len = crate::compiled::detect_seq_len(info.ort_api, kctx, &*plan_ptr);
        let generation_key =
            crate::compiled::detect_attention_generation_key(info.ort_api, kctx, &*plan_ptr);
        let new_generation =
            crate::compiled::detect_attention_self_past_len(info.ort_api, kctx, &*plan_ptr)
                == Some(0)
                || generation_key != (*plan_ptr).compiled.stable_generation_key;
        if new_generation {
            reset_stable_cross_caches(&mut *plan_ptr, generation_key);
        }
        let mut decode_stream_guard = info.decode_stream.lock().unwrap_or_else(|e| e.into_inner());
        let stream = if use_dedicated_decode_stream((*plan_ptr).dedicated_decode_stream, seq_len) {
            decode_stream_guard
                .get_or_insert_with(Stream::new_gpu)
                .as_raw()
        } else {
            info.stream
        };

        let node_count = (*plan_ptr).nodes.len();
        let tr = crate::trace::tracer();
        tr.note_thread("mlx.ep.compute");
        let _region = tr.subgraph_region(node_count);
        tr.sample_gpu_counters();

        // Compiled-decode fast path: handle single-token (S==1) decode via the once-compiled
        // shapeless closure; prefill (S>1) is handled by the shape-keyed prefill path below, and
        // ineligible plans fall through to the eager translator.
        let native_batch_supported = !(*plan_ptr).native_attention_decode
            || crate::compiled::detect_batch_size(info.ort_api, kctx, &*plan_ptr) == Some(1);
        if (*plan_ptr).compiled.enabled && seq_len == Some(1) && native_batch_supported {
            // Cache state: replay (HIT) if the shapeless closure is already compiled, else first
            // trace+compile (MISS). Decode is shapeless, so it never retraces (empty shape key).
            let pre_valid = (*plan_ptr).compiled.valid;
            match crate::compiled::try_compiled(plan_ptr, Slot::Decode, info.ort_api, kctx, stream)
            {
                Ok(true) => {
                    let cache = if pre_valid {
                        crate::trace::CacheState::Hit
                    } else {
                        crate::trace::CacheState::Miss
                    };
                    tr.record_compute_path(
                        crate::trace::ComputePath::Decode,
                        cache,
                        "",
                        &(*plan_ptr).nodes,
                    );
                    return ptr::null_mut();
                }
                Ok(false) => { /* not eligible — fall back to eager below */ }
                Err(msg) => {
                    let c = CString::new(format!("MLX compiled decode failed: {msg}"))
                        .unwrap_or_else(|_| CString::new("MLX compiled decode failed").unwrap());
                    return (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr());
                }
            }
        }

        // Compiled-prefill fast path (Phase 2): the SAME decoder subgraph as decode but at query
        // length S>1. `S` bakes into the trace (causal-mask extent, KV write width), so this uses the
        // unified core in SHAPE-KEYED mode — it retraces per distinct prompt length and replays the
        // fused closure for repeats. Declines (=> eager) for any non-decoder / partial-rotary shape.
        if (*plan_ptr).prefill.enabled && matches!(seq_len, Some(s) if s > 1) {
            let pre_valid = (*plan_ptr).prefill.valid;
            match crate::compiled::try_compiled(plan_ptr, Slot::Prefill, info.ort_api, kctx, stream)
            {
                Ok(true) => {
                    // Shape-keyed on the query length S: a changed key means MLX retraced under us.
                    let cache = if pre_valid {
                        crate::trace::CacheState::Hit
                    } else {
                        crate::trace::CacheState::Miss
                    };
                    let key = seq_len.map(|s| format!("S{s}")).unwrap_or_default();
                    tr.record_compute_path(
                        crate::trace::ComputePath::Prefill,
                        cache,
                        &key,
                        &(*plan_ptr).nodes,
                    );
                    return ptr::null_mut();
                }
                Ok(false) => { /* not eligible — fall back to eager below */ }
                Err(msg) => {
                    let c = CString::new(format!("MLX compiled prefill failed: {msg}"))
                        .unwrap_or_else(|_| CString::new("MLX compiled prefill failed").unwrap());
                    return (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr());
                }
            }
        }

        // General compiled fast path: trace + fuse any claimed static-shape subgraph (CNN / audio /
        // supported control flow) into a shape-keyed closure and replay it. Static Scan unrolls by
        // shape; If/Loop add a host-decision specialization key. Declines on any trace/apply doubt.
        if (*plan_ptr).general.enabled {
            let pre_valid = (*plan_ptr).general.valid;
            match crate::compiled::try_compiled(plan_ptr, Slot::General, info.ort_api, kctx, stream)
            {
                Ok(true) => {
                    let cache = if pre_valid {
                        crate::trace::CacheState::Hit
                    } else {
                        crate::trace::CacheState::Miss
                    };
                    // Shape key from the primary dynamic input so a changed audio/frame size shows as
                    // a RETRACE (only read when observability is active).
                    let key = if tr.active() {
                        let mut key =
                            compute_shape_key(info.ort_api, kctx, &(*plan_ptr).general.dyn_inputs);
                        if let Some(control) = &(*plan_ptr).general.control_flow_key
                            && !control.is_empty()
                        {
                            key.push('|');
                            key.push_str(control);
                        }
                        key
                    } else {
                        String::new()
                    };
                    tr.record_compute_path(
                        crate::trace::ComputePath::General,
                        cache,
                        &key,
                        &(*plan_ptr).nodes,
                    );
                    return ptr::null_mut();
                }
                Ok(false) => { /* not eligible — fall back to eager below */ }
                Err(msg) => {
                    let c = CString::new(format!("MLX compiled general failed: {msg}"))
                        .unwrap_or_else(|_| CString::new("MLX compiled general failed").unwrap());
                    return (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr());
                }
            }
        }

        let mut tctx = TranslationContext::new(&mut *plan_ptr, info.ort_api, kctx, stream);
        match tctx.execute() {
            Ok(()) => {
                tr.record_compute_path(
                    crate::trace::ComputePath::Eager,
                    crate::trace::CacheState::Na,
                    "",
                    &(*plan_ptr).nodes,
                );
                ptr::null_mut()
            }
            Err(msg) => {
                let c = CString::new(format!("MLX subgraph failed: {msg}"))
                    .unwrap_or_else(|_| CString::new("MLX subgraph failed").unwrap());
                (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr())
            }
        }
    }
}

/// Shape/dtype key for every dynamic closure input, used to classify a shape-keyed general Compute
/// as HIT vs RETRACE. Best-effort: inputs that cannot be read are omitted. Only called when
/// observability is active.
unsafe fn compute_shape_key(
    ort_api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
    inputs: &[crate::engine::DynInput],
) -> String {
    let mut key = String::new();
    for input in inputs {
        if let Ok((_data, shape, dtype)) =
            crate::engine::read_ctx_input_raw(ort_api, kctx, input.ctx_index)
        {
            if !key.is_empty() {
                key.push(';');
            }
            key.push_str(&format!("{}:{dtype:?}:{shape:?}", input.ctx_index));
        }
    }
    key
}

unsafe extern "C" fn release_node_compute_infos(
    _p: *mut ort::OrtEp,
    infos: *mut *mut ort::OrtNodeComputeInfo,
    num: usize,
) {
    unsafe {
        for i in 0..num {
            let ptr = *infos.add(i);
            if !ptr.is_null() {
                drop(Box::from_raw(ptr as *mut SubgraphComputeInfo));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Plan, reset_stable_cross_caches, use_dedicated_decode_stream};

    #[test]
    fn dedicated_stream_is_decode_only() {
        assert!(use_dedicated_decode_stream(true, Some(1)));
        assert!(!use_dedicated_decode_stream(true, Some(2)));
        assert!(!use_dedicated_decode_stream(true, None));
        assert!(!use_dedicated_decode_stream(false, Some(1)));
    }

    #[test]
    fn generation_reset_covers_decode_and_prefill() {
        let mut plan = Plan::new(Vec::new());
        reset_stable_cross_caches(&mut plan, Some(42));
        assert_eq!(plan.compiled.stable_generation_key, Some(42));
        assert_eq!(plan.prefill.stable_generation_key, Some(42));
    }
}

#[cfg(test)]
mod float64_plan_tests {
    use crate::engine::{NodeDesc, OutRef, Plan};
    use crate::sys::ort;

    fn node_with_output(otype: ort::ONNXTensorElementDataType) -> NodeDesc {
        let mut node = NodeDesc::new(
            "Elu".to_string(),
            String::new(),
            6,
            crate::registry::CompileShapeSafety::Shapeless,
        );
        node.outputs.push(OutRef {
            name: "y".to_string(),
            external: true,
            ctx_index: 0,
            otype,
        });
        node
    }

    #[test]
    fn float64_plan_requests_the_cpu_stream() {
        let plan = Plan::new(vec![node_with_output(
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        )]);
        assert!(plan.requires_cpu_stream);
        assert!(!plan.dedicated_decode_stream);
    }

    #[test]
    fn float32_plan_keeps_the_gpu_stream() {
        let plan = Plan::new(vec![node_with_output(
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        )]);
        assert!(!plan.requires_cpu_stream);
    }
}
