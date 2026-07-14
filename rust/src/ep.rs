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
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

use crate::engine::{InitData, NodeDesc, OutRef, Plan, Src, TensorRef, TranslationContext};
use crate::factory::ORT_API_VERSION;
use crate::mlx::Stream;
use crate::registry::{claimable, NodeView};
use crate::sys::{mlx, ort};

#[repr(C)]
pub struct MlxEp {
    base: ort::OrtEp,
    ort_api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    name: CString,
    stream: Stream,
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
        })
    }

    pub fn as_ptr(self: Box<Self>) -> *mut ort::OrtEp {
        Box::into_raw(self) as *mut ort::OrtEp
    }
}

// The per-EP mlx stream is now owned by the `Stream` RAII wrapper, freed exactly once when ORT
// calls ReleaseEp (which drops our Box<MlxEp>). No manual free / no explicit Drop needed.

#[inline]
unsafe fn this(p: *const ort::OrtEp) -> *const MlxEp {
    p as *const MlxEp
}

unsafe extern "C" fn get_name(p: *const ort::OrtEp) -> *const c_char {
    (*this(p)).name.as_ptr()
}

unsafe extern "C" fn get_default_memory_device(
    _p: *const ort::OrtEp,
    device: *mut *const ort::OrtMemoryDevice,
) -> *mut ort::OrtStatus {
    // I/O stays on the CPU allocator (unified memory); no device memory advertised.
    *device = ptr::null();
    ptr::null_mut()
}

// ---------------------------------------------------------------------------
// GetCapability: claim via registry + convex clustering.
// ---------------------------------------------------------------------------

unsafe extern "C" fn get_capability(
    p: *mut ort::OrtEp,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> *mut ort::OrtStatus {
    let ep = &*this(p);
    let api = &*ep.ort_api;
    let ep_api = &*ep.ep_api;

    let mut num: usize = 0;
    let st = (api.Graph_GetNumNodes.unwrap())(graph, &mut num);
    if !st.is_null() {
        return st;
    }
    if num == 0 {
        return ptr::null_mut();
    }
    let mut nodes: Vec<*const ort::OrtNode> = vec![ptr::null(); num];
    let st = (api.Graph_GetNodes.unwrap())(graph, nodes.as_mut_ptr(), num);
    if !st.is_null() {
        return st;
    }

    // Which nodes can MLX translate exactly (registry claim predicate).
    let supported: Vec<bool> = nodes
        .iter()
        .map(|&node| {
            let view = NodeView::new(ep.ort_api, node);
            claimable(&view)
        })
        .collect();

    let clusters = build_convex_clusters(api, &nodes, &supported);

    let add_fuse = ep_api.EpGraphSupportInfo_AddNodesToFuse.unwrap();
    let mut claimed = 0usize;
    for cluster in &clusters {
        let group: Vec<*const ort::OrtNode> = cluster.iter().map(|&i| nodes[i]).collect();
        let mut opts: ort::OrtNodeFusionOptions = std::mem::zeroed();
        opts.ort_version_supported = ORT_API_VERSION;
        // ORT supplies constant initializers as runtime fused-node inputs (we read them at Run).
        opts.drop_constant_initializers = false;
        let st = add_fuse(support, group.as_ptr(), group.len(), &opts);
        if !st.is_null() {
            return st;
        }
        claimed += cluster.len();
    }
    eprintln!(
        "[rust-mlx-ep] GetCapability: claimed {claimed} of {num} node(s) across {} fused subgraph(s)",
        clusters.len()
    );
    ptr::null_mut()
}

/// Value-info tensor name, or "" for an omitted optional slot.
unsafe fn value_info_name(api: &ort::OrtApi, vi: *const ort::OrtValueInfo) -> String {
    if vi.is_null() {
        return String::new();
    }
    let mut p: *const c_char = ptr::null();
    let st = (api.GetValueInfoName.unwrap())(vi, &mut p);
    if !st.is_null() || p.is_null() {
        return String::new();
    }
    CStr::from_ptr(p).to_string_lossy().into_owned()
}

unsafe fn node_input_names(api: &ort::OrtApi, node: *const ort::OrtNode) -> Vec<String> {
    let mut n: usize = 0;
    (api.Node_GetNumInputs.unwrap())(node, &mut n);
    let mut v: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); n];
    if n > 0 {
        (api.Node_GetInputs.unwrap())(node, v.as_mut_ptr(), n);
    }
    v.iter().map(|&vi| value_info_name(api, vi)).collect()
}

unsafe fn node_output_names(api: &ort::OrtApi, node: *const ort::OrtNode) -> Vec<String> {
    let mut n: usize = 0;
    (api.Node_GetNumOutputs.unwrap())(node, &mut n);
    let mut v: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); n];
    if n > 0 {
        (api.Node_GetOutputs.unwrap())(node, v.as_mut_ptr(), n);
    }
    v.iter().map(|&vi| value_info_name(api, vi)).collect()
}

/// Groups supported nodes into maximal, convex, connected clusters. A set S is convex (a valid single
/// fused node) iff no node x outside S lies on a path between two members of S. Faithful port of
/// `BuildConvexClusters` (union-find + reachability bitsets).
fn build_convex_clusters(
    api: &ort::OrtApi,
    nodes: &[*const ort::OrtNode],
    supported: &[bool],
) -> Vec<Vec<usize>> {
    let n = nodes.len();
    let words = (n + 63) / 64;

    // tensor name -> producing node index.
    let mut producer: HashMap<String, usize> = HashMap::new();
    for (i, &node) in nodes.iter().enumerate() {
        for name in unsafe { node_output_names(api, node) } {
            if !name.is_empty() {
                producer.entry(name).or_insert(i);
            }
        }
    }

    // Direct successors / predecessors within the graph.
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..n {
        let mut seen: HashSet<usize> = HashSet::new();
        for name in unsafe { node_input_names(api, nodes[j]) } {
            if name.is_empty() {
                continue;
            }
            if let Some(&i) = producer.get(&name) {
                if i != j && seen.insert(i) {
                    succ[i].push(j);
                    pred[j].push(i);
                }
            }
        }
    }

    // Kahn topological order for reachability accumulation.
    let mut indeg: Vec<usize> = pred.iter().map(|p| p.len()).collect();
    let mut stack: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(u) = stack.pop() {
        order.push(u);
        for &v in &succ[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                stack.push(v);
            }
        }
    }
    if order.len() != n {
        order = (0..n).collect();
    }

    // reach[i] = set of nodes reachable from i (transitive successors, excluding i).
    let mut reach: Vec<Vec<u64>> = vec![vec![0u64; words]; n];
    for &u in order.iter().rev() {
        for &v in &succ[u] {
            bit_set(&mut reach[u], v);
            let src = reach[v].clone();
            bit_or_into(&mut reach[u], &src);
        }
    }

    // Cluster state keyed by union-find root.
    let mut parent: Vec<usize> = (0..n).collect();
    let mut cluster_bits: Vec<Vec<u64>> = vec![vec![0u64; words]; n];
    let mut reach_bits: Vec<Vec<u64>> = vec![vec![0u64; words]; n];
    for i in 0..n {
        if supported[i] {
            bit_set(&mut cluster_bits[i], i);
            reach_bits[i] = reach[i].clone();
        }
    }

    // Candidate merge edges: direct data edges between two supported nodes.
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for u in 0..n {
        if !supported[u] {
            continue;
        }
        for &v in &succ[u] {
            if supported[v] {
                edges.push((u, v));
            }
        }
    }

    let is_convex = |s_bits: &[u64], reach_s: &[u64], reach: &[Vec<u64>]| -> bool {
        for x in 0..n {
            if bit_test(s_bits, x) {
                continue;
            }
            if !bit_test(reach_s, x) {
                continue; // S cannot reach x
            }
            if bit_intersects(&reach[x], s_bits) {
                return false; // x can reach back into S
            }
        }
        true
    };

    let mut changed = true;
    while changed {
        changed = false;
        for &(a, b) in &edges {
            let ra = uf_find(&mut parent, a);
            let rb = uf_find(&mut parent, b);
            if ra == rb {
                continue;
            }
            let mut merged = cluster_bits[ra].clone();
            bit_or_into(&mut merged, &cluster_bits[rb]);
            let mut merged_reach = reach_bits[ra].clone();
            bit_or_into(&mut merged_reach, &reach_bits[rb]);
            if !is_convex(&merged, &merged_reach, &reach) {
                continue;
            }
            parent[rb] = ra;
            cluster_bits[ra] = merged;
            reach_bits[ra] = merged_reach;
            changed = true;
        }
    }

    let mut grouped: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        if supported[i] {
            let root = uf_find(&mut parent, i);
            grouped.entry(root).or_default().push(i);
        }
    }
    let mut clusters: Vec<Vec<usize>> = grouped
        .into_values()
        .map(|mut c| {
            c.sort_unstable();
            c
        })
        .collect();
    clusters.sort_by_key(|c| c[0]);
    clusters
}

fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

#[inline]
fn bit_set(b: &mut [u64], i: usize) {
    b[i >> 6] |= 1u64 << (i & 63);
}
#[inline]
fn bit_test(b: &[u64], i: usize) -> bool {
    (b[i >> 6] >> (i & 63)) & 1 != 0
}
#[inline]
fn bit_or_into(dst: &mut [u64], src: &[u64]) {
    for i in 0..dst.len() {
        dst[i] |= src[i];
    }
}
#[inline]
fn bit_intersects(a: &[u64], b: &[u64]) -> bool {
    a.iter().zip(b.iter()).any(|(x, y)| x & y != 0)
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
    _ep_context_nodes: *mut *mut ort::OrtNode,
) -> *mut ort::OrtStatus {
    let ep = &*this(p);
    let api = &*ep.ort_api;

    for i in 0..count {
        let graph = *graphs.add(i);
        let fused_node = *fused_nodes.add(i);
        match build_plan(api, graph, fused_node) {
            Ok(plan) => {
                let info = SubgraphComputeInfo::new(ep.ort_api, ep.stream.as_raw(), plan);
                *node_compute_infos.add(i) = Box::into_raw(info) as *mut ort::OrtNodeComputeInfo;
            }
            Err(msg) => {
                let c =
                    CString::new(msg).unwrap_or_else(|_| CString::new("MLX compile error").unwrap());
                return (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr());
            }
        }
    }
    ptr::null_mut()
}

unsafe fn build_plan(
    api: &ort::OrtApi,
    graph: *const ort::OrtGraph,
    fused_node: *const ort::OrtNode,
) -> Result<Plan, String> {
    // Fused-node input/output name -> OrtKernelContext index (the runtime I/O boundary).
    let ctx_input_index: HashMap<String, usize> = node_input_names(api, fused_node)
        .into_iter()
        .enumerate()
        .filter(|(_, n)| !n.is_empty())
        .map(|(k, n)| (n, k))
        .collect();
    let ctx_output_index: HashMap<String, usize> = node_output_names(api, fused_node)
        .into_iter()
        .enumerate()
        .filter(|(_, n)| !n.is_empty())
        .map(|(k, n)| (n, k))
        .collect();

    // Constant initializers referenced by the subgraph (session-owned storage).
    let initializers = collect_initializers(api, graph)?;

    // Subgraph nodes.
    let mut num_nodes: usize = 0;
    (api.Graph_GetNumNodes.unwrap())(graph, &mut num_nodes);
    let mut snodes: Vec<*const ort::OrtNode> = vec![ptr::null(); num_nodes];
    if num_nodes > 0 {
        (api.Graph_GetNodes.unwrap())(graph, snodes.as_mut_ptr(), num_nodes);
    }

    // Producer of each intra-subgraph tensor.
    let mut producer: HashMap<String, usize> = HashMap::new();
    for (k, &node) in snodes.iter().enumerate() {
        for name in node_output_names(api, node) {
            if !name.is_empty() {
                producer.entry(name).or_insert(k);
            }
        }
    }

    // Topological order over the subgraph.
    let order = topo_order(api, &snodes, &producer);

    let mut nodes: Vec<NodeDesc> = Vec::with_capacity(snodes.len());
    for &idx in &order {
        let node = snodes[idx];
        let op_type = node_op_type(api, node);
        let domain = node_domain(api, node);
        let since_version = node_since_version(api, node);
        let mut nd = NodeDesc::new(op_type, domain, since_version);

        collect_attributes(api, node, &mut nd);

        // Inputs.
        for name in node_input_names(api, node) {
            let tr = if name.is_empty() {
                TensorRef::absent()
            } else if producer.contains_key(&name) {
                TensorRef {
                    name,
                    source: Src::Intermediate,
                    ctx_index: 0,
                    constant: false,
                    init: None,
                }
            } else if let Some(&ci) = ctx_input_index.get(&name) {
                // A constant ctx input's compile-time init pointer goes stale after Compile; the
                // `constant` flag lets Resolve wrap/cache it once from live ctx data on first Run.
                let constant = initializers.contains_key(&name);
                TensorRef {
                    name,
                    source: Src::CtxInput,
                    ctx_index: ci,
                    constant,
                    init: None,
                }
            } else if let Some(init) = initializers.get(&name) {
                TensorRef {
                    name,
                    source: Src::Initializer,
                    ctx_index: 0,
                    constant: false,
                    init: Some(init.clone()),
                }
            } else {
                return Err(format!("MLX could not resolve subgraph input {name}"));
            };
            nd.inputs.push(tr);
        }

        // Outputs.
        for name in node_output_names(api, node) {
            let otype = output_element_type(api, node, &name);
            let (external, ctx_index) = match ctx_output_index.get(&name) {
                Some(&ci) if !name.is_empty() => (true, ci),
                _ => (false, 0),
            };
            nd.outputs.push(OutRef {
                name,
                external,
                ctx_index,
                otype,
            });
        }

        nodes.push(nd);
    }

    Ok(Plan::new(nodes))
}

unsafe fn collect_initializers(
    api: &ort::OrtApi,
    graph: *const ort::OrtGraph,
) -> Result<HashMap<String, InitData>, String> {
    let mut map = HashMap::new();
    let mut num: usize = 0;
    (api.Graph_GetNumInitializers.unwrap())(graph, &mut num);
    if num == 0 {
        return Ok(map);
    }
    let mut vis: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); num];
    (api.Graph_GetInitializers.unwrap())(graph, vis.as_mut_ptr(), num);
    for &vi in &vis {
        let name = value_info_name(api, vi);
        if name.is_empty() {
            continue;
        }
        let mut value: *const ort::OrtValue = ptr::null();
        let st = (api.ValueInfo_GetInitializerValue.unwrap())(vi, &mut value);
        if !st.is_null() || value.is_null() {
            continue;
        }
        let mut info: *mut ort::OrtTensorTypeAndShapeInfo = ptr::null_mut();
        (api.GetTensorTypeAndShape.unwrap())(value, &mut info);
        let mut nd: usize = 0;
        (api.GetDimensionsCount.unwrap())(info, &mut nd);
        let mut dims = vec![0i64; nd];
        if nd > 0 {
            (api.GetDimensions.unwrap())(info, dims.as_mut_ptr(), nd);
        }
        let mut etype: ort::ONNXTensorElementDataType = 0;
        (api.GetTensorElementType.unwrap())(info, &mut etype);
        let mut count: usize = 0;
        (api.GetTensorShapeElementCount.unwrap())(info, &mut count);
        (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info);
        let mut data: *const c_void = ptr::null();
        (api.GetTensorData.unwrap())(value, &mut data);
        map.insert(
            name,
            InitData {
                data,
                shape: dims,
                dtype: etype,
                count,
            },
        );
    }
    Ok(map)
}

unsafe fn topo_order(
    api: &ort::OrtApi,
    snodes: &[*const ort::OrtNode],
    producer: &HashMap<String, usize>,
) -> Vec<usize> {
    let n = snodes.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indeg: Vec<usize> = vec![0; n];
    for j in 0..n {
        let mut seen: HashSet<usize> = HashSet::new();
        for name in node_input_names(api, snodes[j]) {
            if name.is_empty() {
                continue;
            }
            if let Some(&i) = producer.get(&name) {
                if i != j && seen.insert(i) {
                    succ[i].push(j);
                    indeg[j] += 1;
                }
            }
        }
    }
    let mut stack: Vec<usize> = (0..n).filter(|&k| indeg[k] == 0).collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(u) = stack.pop() {
        order.push(u);
        for &v in &succ[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                stack.push(v);
            }
        }
    }
    if order.len() != n {
        order = (0..n).collect();
    }
    order
}

unsafe fn node_op_type(api: &ort::OrtApi, node: *const ort::OrtNode) -> String {
    let mut p: *const c_char = ptr::null();
    (api.Node_GetOperatorType.unwrap())(node, &mut p);
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

unsafe fn node_domain(api: &ort::OrtApi, node: *const ort::OrtNode) -> String {
    let mut p: *const c_char = ptr::null();
    (api.Node_GetDomain.unwrap())(node, &mut p);
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

unsafe fn node_since_version(api: &ort::OrtApi, node: *const ort::OrtNode) -> i32 {
    let mut v: i32 = 0;
    (api.Node_GetSinceVersion.unwrap())(node, &mut v);
    v
}

/// Element type of node output named `name` (UNDEFINED if not a tensor).
unsafe fn output_element_type(
    api: &ort::OrtApi,
    node: *const ort::OrtNode,
    name: &str,
) -> ort::ONNXTensorElementDataType {
    let mut n: usize = 0;
    (api.Node_GetNumOutputs.unwrap())(node, &mut n);
    let mut v: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); n];
    if n > 0 {
        (api.Node_GetOutputs.unwrap())(node, v.as_mut_ptr(), n);
    }
    for &vi in &v {
        if vi.is_null() || value_info_name(api, vi) != name {
            continue;
        }
        let mut ti: *const ort::OrtTypeInfo = ptr::null();
        let st = (api.GetValueInfoTypeInfo.unwrap())(vi, &mut ti);
        if !st.is_null() || ti.is_null() {
            return 0;
        }
        let mut onnx_type: ort::ONNXType = 0;
        (api.GetOnnxTypeFromTypeInfo.unwrap())(ti, &mut onnx_type);
        if onnx_type != ort::ONNXType_ONNX_TYPE_TENSOR {
            return 0;
        }
        let mut tsi: *const ort::OrtTensorTypeAndShapeInfo = ptr::null();
        (api.CastTypeInfoToTensorInfo.unwrap())(ti, &mut tsi);
        if tsi.is_null() {
            return 0;
        }
        let mut dtype: ort::ONNXTensorElementDataType = 0;
        (api.GetTensorElementType.unwrap())(tsi, &mut dtype);
        return dtype;
    }
    0
}

/// Generic attribute copy: every INT/FLOAT/INTS/FLOATS/STRING attr into the NodeDesc maps.
unsafe fn collect_attributes(api: &ort::OrtApi, node: *const ort::OrtNode, nd: &mut NodeDesc) {
    let mut num: usize = 0;
    (api.Node_GetNumAttributes.unwrap())(node, &mut num);
    if num == 0 {
        return;
    }
    let mut attrs: Vec<*const ort::OrtOpAttr> = vec![ptr::null(); num];
    (api.Node_GetAttributes.unwrap())(node, attrs.as_mut_ptr(), num);
    let read = api.ReadOpAttr.unwrap();
    for &attr in &attrs {
        if attr.is_null() {
            continue;
        }
        let mut name_p: *const c_char = ptr::null();
        (api.OpAttr_GetName.unwrap())(attr, &mut name_p);
        if name_p.is_null() {
            continue;
        }
        let name = CStr::from_ptr(name_p).to_string_lossy().into_owned();
        let mut atype: ort::OrtOpAttrType = 0;
        (api.OpAttr_GetType.unwrap())(attr, &mut atype);
        match atype {
            t if t == ort::OrtOpAttrType_ORT_OP_ATTR_INT => {
                let mut v: i64 = 0;
                let mut out: usize = 0;
                let st = read(
                    attr,
                    atype,
                    &mut v as *mut i64 as *mut c_void,
                    std::mem::size_of::<i64>(),
                    &mut out,
                );
                if st.is_null() {
                    nd.ints.insert(name, v);
                }
            }
            t if t == ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT => {
                let mut v: f32 = 0.0;
                let mut out: usize = 0;
                let st = read(
                    attr,
                    atype,
                    &mut v as *mut f32 as *mut c_void,
                    std::mem::size_of::<f32>(),
                    &mut out,
                );
                if st.is_null() {
                    nd.floats.insert(name, v);
                }
            }
            t if t == ort::OrtOpAttrType_ORT_OP_ATTR_INTS => {
                if let Some(v) = read_array::<i64>(read, attr, atype) {
                    nd.int_arrays.insert(name, v);
                }
            }
            t if t == ort::OrtOpAttrType_ORT_OP_ATTR_FLOATS => {
                if let Some(v) = read_array::<f32>(read, attr, atype) {
                    nd.float_arrays.insert(name, v);
                }
            }
            t if t == ort::OrtOpAttrType_ORT_OP_ATTR_STRING => {
                let mut needed: usize = 0;
                let _ = read(attr, atype, ptr::null_mut(), 0, &mut needed);
                if needed > 0 {
                    let mut buf: Vec<u8> = vec![0u8; needed];
                    let mut out: usize = 0;
                    let st = read(attr, atype, buf.as_mut_ptr() as *mut c_void, needed, &mut out);
                    if st.is_null() {
                        buf.truncate(out.min(needed));
                        if let Ok(s) = String::from_utf8(buf) {
                            nd.strings.insert(name, s);
                        }
                    }
                }
            }
            _ => {} // STRINGS / GRAPH / TENSOR not carried by wave-1 ops.
        }
    }
}

/// Read an array-valued attribute (INTS/FLOATS): size, allocate, read.
unsafe fn read_array<T: Copy + Default>(
    read: unsafe extern "C" fn(
        *const ort::OrtOpAttr,
        ort::OrtOpAttrType,
        *mut c_void,
        usize,
        *mut usize,
    ) -> *mut ort::OrtStatus,
    attr: *const ort::OrtOpAttr,
    atype: ort::OrtOpAttrType,
) -> Option<Vec<T>> {
    let mut needed_bytes: usize = 0;
    let _ = read(attr, atype, ptr::null_mut(), 0, &mut needed_bytes);
    if needed_bytes == 0 {
        return Some(Vec::new());
    }
    let elem = std::mem::size_of::<T>();
    let count = needed_bytes / elem;
    let mut buf: Vec<T> = vec![T::default(); count];
    let mut out: usize = 0;
    let st = read(
        attr,
        atype,
        buf.as_mut_ptr() as *mut c_void,
        needed_bytes,
        &mut out,
    );
    if st.is_null() {
        Some(buf)
    } else {
        None
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
    plan: Plan,
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
            plan,
        })
    }
}

unsafe extern "C" fn create_state(
    this_ptr: *mut ort::OrtNodeComputeInfo,
    _compute_context: *mut ort::OrtNodeComputeContext,
    compute_state: *mut *mut c_void,
) -> *mut ort::OrtStatus {
    *compute_state = this_ptr as *mut c_void;
    ptr::null_mut()
}

unsafe extern "C" fn release_state(_this: *mut ort::OrtNodeComputeInfo, _state: *mut c_void) {}

unsafe extern "C" fn compute(
    _this: *mut ort::OrtNodeComputeInfo,
    state: *mut c_void,
    kctx: *mut ort::OrtKernelContext,
) -> *mut ort::OrtStatus {
    let info = &mut *(state as *mut SubgraphComputeInfo);
    let api = &*info.ort_api;

    let node_count = info.plan.nodes.len();
    let mut tctx = TranslationContext::new(&mut info.plan, info.ort_api, kctx, info.stream);
    match tctx.execute() {
        Ok(()) => {
            eprintln!("[rust-mlx-ep] Compute: subgraph run via mlx-c ({node_count} node(s))");
            ptr::null_mut()
        }
        Err(msg) => {
            let c = CString::new(format!("MLX subgraph failed: {msg}"))
                .unwrap_or_else(|_| CString::new("MLX subgraph failed").unwrap());
            (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr())
        }
    }
}

unsafe extern "C" fn release_node_compute_infos(
    _p: *mut ort::OrtEp,
    infos: *mut *mut ort::OrtNodeComputeInfo,
    num: usize,
) {
    for i in 0..num {
        let ptr = *infos.add(i);
        if !ptr.is_null() {
            drop(Box::from_raw(ptr as *mut SubgraphComputeInfo));
        }
    }
}
