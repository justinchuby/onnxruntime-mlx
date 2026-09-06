//! Compile-time conversion from an owned [`OrtGraphSnapshot`] to an execution [`Plan`].
//!
//! The snapshot is the authority for graph structure.  The short-lived sidecars below only retain
//! ABI handles while `Compile` is executing, so they can read node attributes, classify concrete
//! forms, and copy initializer bytes.  Nothing borrowed from ORT reaches `Plan`.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, c_char, c_void};
use std::ptr;
use std::sync::Arc;

use onnx_runtime_ir::{NodeId, ValueId};

use crate::engine::{ConstTensor, InitData, NodeDesc, OutRef, Plan, Src, SubgraphDesc, TensorRef};
use crate::ort_graph::{DimensionMetadata, OrtGraphSnapshot};
use crate::registry::NodeView;
use crate::sys::ort;

struct GraphSidecars {
    nodes: HashMap<NodeId, *const ort::OrtNode>,
    initializers: HashMap<ValueId, InitData>,
    subgraphs: HashMap<(NodeId, String), GraphSidecars>,
}

/// Build an execution plan while ORT's compile-time graph views are valid.
///
/// # Safety
/// `graph` and `fused_node` are live ORT objects for the duration of this call.
pub(crate) unsafe fn build_plan(
    api: &ort::OrtApi,
    graph: *const ort::OrtGraph,
    fused_node: *const ort::OrtNode,
) -> Result<Plan, String> {
    let snapshot = unsafe { OrtGraphSnapshot::from_ort(api, graph)? };
    let sidecars = unsafe { GraphSidecars::capture(api, graph, &snapshot)? };
    let ctx_inputs = unsafe { fused_boundary_indices(api, fused_node, true, &snapshot)? };
    let ctx_outputs = unsafe { fused_boundary_indices(api, fused_node, false, &snapshot)? };
    let ctx_input_names = boundary_names(&ctx_inputs, &snapshot);
    let no_enclosing_initializers = HashMap::new();
    let nodes = build_graph(
        api,
        &snapshot,
        &sidecars,
        &ctx_inputs,
        &ctx_input_names,
        &ctx_outputs,
        &HashSet::new(),
        &no_enclosing_initializers,
        true,
    )?;

    let has_control_flow = any_control_flow(&nodes);
    let has_compile_barrier = has_compile_barrier(&nodes);
    let mut plan = Plan::new(nodes);
    if snapshot
        .ir
        .topological_order()
        .map_err(|error| format!("optimized graph IR has no topological order: {error}"))?
        .into_iter()
        .any(|id| crate::registry::node_uses_float64(&NodeView::new(api, sidecars.nodes[&id])))
    {
        plan.requires_cpu_stream = true;
        plan.dedicated_decode_stream = false;
    }
    plan.compiled.enabled =
        crate::compiled::decode_enabled(has_control_flow || has_compile_barrier, &plan.nodes);
    plan.prefill.enabled =
        crate::compiled::prefill_enabled(has_control_flow || has_compile_barrier, &plan.nodes);
    plan.general.enabled = crate::compiled::general_enabled(has_compile_barrier, &plan.nodes);
    plan.general.control_flow_specialization_enabled = has_control_flow;
    if plan.requires_cpu_stream {
        plan.compiled.enabled = false;
        plan.prefill.enabled = false;
        plan.general.enabled = false;
    }
    Ok(plan)
}

impl GraphSidecars {
    unsafe fn capture(
        api: &ort::OrtApi,
        graph: *const ort::OrtGraph,
        snapshot: &OrtGraphSnapshot,
    ) -> Result<Self, String> {
        let mut count = 0;
        check(
            api,
            unsafe { (api.Graph_GetNumNodes.unwrap())(graph, &mut count) },
            "reading nodes",
        )?;
        let mut raw_nodes = vec![ptr::null(); count];
        if count != 0 {
            check(
                api,
                unsafe { (api.Graph_GetNodes.unwrap())(graph, raw_nodes.as_mut_ptr(), count) },
                "reading nodes",
            )?;
        }
        let mut nodes = HashMap::with_capacity(count);
        let mut subgraphs = HashMap::new();
        for raw in raw_nodes {
            let mut ort_id = 0;
            check(
                api,
                unsafe { (api.Node_GetId.unwrap())(raw, &mut ort_id) },
                "reading node ID",
            )?;
            let &id = snapshot
                .node_by_ort_id
                .get(&ort_id)
                .ok_or_else(|| format!("snapshot is missing raw node {ort_id}"))?;
            nodes.insert(id, raw);

            let mut sub_count = 0;
            check(
                api,
                unsafe { (api.Node_GetNumSubgraphs.unwrap())(raw, &mut sub_count) },
                "reading node subgraph count",
            )?;
            if sub_count != 0 {
                let mut graphs = vec![ptr::null(); sub_count];
                let mut names = vec![ptr::null(); sub_count];
                check(
                    api,
                    unsafe {
                        (api.Node_GetSubgraphs.unwrap())(
                            raw,
                            graphs.as_mut_ptr(),
                            sub_count,
                            names.as_mut_ptr(),
                        )
                    },
                    "reading node subgraphs",
                )?;
                for (raw_graph, raw_name) in graphs.into_iter().zip(names) {
                    let name = cstr(raw_name);
                    let child_snapshot = snapshot
                        .subgraphs
                        .get(&(id, name.clone()))
                        .ok_or_else(|| format!("snapshot is missing subgraph {name:?}"))?;
                    subgraphs.insert((id, name), unsafe {
                        Self::capture(api, raw_graph, child_snapshot)?
                    });
                }
            }
        }
        let raw_initializers = unsafe { collect_initializer_bytes(api, graph)? };
        let mut initializers = HashMap::new();
        for &value in snapshot.initializers.keys() {
            let name = &snapshot.values[&value].name;
            if let Some(&data) = raw_initializers.get(name) {
                initializers.insert(value, snapshot_initializer_data(snapshot, value, data)?);
            }
        }
        Ok(Self {
            nodes,
            initializers,
            subgraphs,
        })
    }
}

unsafe fn fused_boundary_indices(
    api: &ort::OrtApi,
    fused_node: *const ort::OrtNode,
    inputs: bool,
    snapshot: &OrtGraphSnapshot,
) -> Result<HashMap<ValueId, usize>, String> {
    let mut count = 0;
    check(
        api,
        unsafe {
            if inputs {
                (api.Node_GetNumInputs.unwrap())(fused_node, &mut count)
            } else {
                (api.Node_GetNumOutputs.unwrap())(fused_node, &mut count)
            }
        },
        "reading fused-node boundary count",
    )?;
    let mut slots = vec![ptr::null(); count];
    if count != 0 {
        check(
            api,
            unsafe {
                if inputs {
                    (api.Node_GetInputs.unwrap())(fused_node, slots.as_mut_ptr(), count)
                } else {
                    (api.Node_GetOutputs.unwrap())(fused_node, slots.as_mut_ptr(), count)
                }
            },
            "reading fused-node boundary slots",
        )?;
    }
    let mut indices = HashMap::new();
    for (index, slot) in slots.into_iter().enumerate() {
        if slot.is_null() {
            continue;
        }
        let name = unsafe { value_name(api, slot)? };
        if name.is_empty() {
            continue;
        }
        let value = snapshot
            .value_by_name
            .get(&name)
            .copied()
            .ok_or_else(|| format!("snapshot is missing fused-node boundary {name:?}"))?;
        indices.insert(value, index);
    }
    Ok(indices)
}

fn boundary_names(
    indices: &HashMap<ValueId, usize>,
    snapshot: &OrtGraphSnapshot,
) -> HashMap<String, usize> {
    indices
        .iter()
        .map(|(&value, &index)| (snapshot.values[&value].name.clone(), index))
        .collect()
}

fn initializer_scope(
    snapshot: &OrtGraphSnapshot,
    sidecars: &GraphSidecars,
    enclosing: &HashMap<String, InitData>,
) -> Result<HashMap<String, InitData>, String> {
    let mut inits = enclosing
        .iter()
        .map(|(name, data)| unsafe { own_init_data(data).map(|data| (name.clone(), data)) })
        .collect::<Result<HashMap<_, _>, _>>()?;
    for (&value, data) in &sidecars.initializers {
        inits.insert(snapshot.values[&value].name.clone(), unsafe {
            own_init_data(data)?
        });
    }
    Ok(inits)
}

fn build_graph(
    api: &ort::OrtApi,
    snapshot: &OrtGraphSnapshot,
    sidecars: &GraphSidecars,
    ctx_inputs: &HashMap<ValueId, usize>,
    ctx_input_names: &HashMap<String, usize>,
    ctx_outputs: &HashMap<ValueId, usize>,
    enclosing_names: &HashSet<String>,
    enclosing_inits: &HashMap<String, InitData>,
    root: bool,
) -> Result<Vec<NodeDesc>, String> {
    let inits = if root {
        // Root initializers are already copied exactly once. Convert their stable ValueId keys to
        // names here, after snapshot capture, rather than consulting the raw graph again.
        sidecars
            .initializers
            .iter()
            .map(|(&value, data)| unsafe {
                own_init_data(data).map(|data| (snapshot.values[&value].name.clone(), data))
            })
            .collect::<Result<HashMap<_, _>, _>>()?
    } else {
        initializer_scope(snapshot, sidecars, enclosing_inits)?
    };
    let formals = snapshot
        .graph_input_slots
        .iter()
        .flatten()
        .map(|value| snapshot.values[value].name.clone())
        .collect::<HashSet<_>>();
    let mut visible = enclosing_names.clone();
    visible.extend(formals.iter().cloned());
    for node in snapshot.nodes.values() {
        for value in node.output_slots.iter().flatten() {
            visible.insert(snapshot.values[value].name.clone());
        }
    }

    let order = snapshot
        .ir
        .topological_order()
        .map_err(|error| format!("optimized graph IR has no topological order: {error}"))?;
    let mut nodes = Vec::with_capacity(order.len());
    for id in order {
        let metadata = &snapshot.nodes[&id];
        let raw = sidecars.nodes[&id];
        let view = NodeView::new(api, raw);
        let mut node = NodeDesc::new(
            metadata.op_type.clone(),
            metadata.domain.clone(),
            metadata.since_version,
            crate::registry::compile_shape_safety(&view),
        );
        let tracer = crate::trace::tracer();
        if tracer.is_enabled() {
            node.node_id = metadata.ort_node_id;
            node.name = metadata.name.clone();
        }
        unsafe { collect_attributes(api, raw, &mut node) };
        let _span = tracer.op_span(
            &node,
            metadata.input_slots.len(),
            metadata.output_slots.len(),
        );

        for (input_index, value) in metadata.input_slots.iter().enumerate() {
            node.inputs.push(match value {
                None => TensorRef::absent(),
                Some(value) => {
                    if !snapshot
                        .ir
                        .uses(*value)
                        .contains(&(id, input_index as u32))
                    {
                        return Err(format!(
                            "optimized graph IR is missing consumer link for {} input {input_index}",
                            metadata.name
                        ));
                    }
                    let name = snapshot.values[&value].name.clone();
                    if snapshot.ir.value(*value).producer.is_some()
                        || (!root && formals.contains(&name))
                    {
                        tensor_ref(name, Src::Intermediate, 0, None)
                    } else if let Some(init) = inits.get(&name) {
                        tensor_ref(name, Src::Initializer, 0, Some(init.clone()))
                    } else if let Some(&index) = if root {
                        ctx_inputs.get(value)
                    } else {
                        ctx_input_names.get(&name)
                    } {
                        tensor_ref(name, Src::CtxInput, index, None)
                    } else if enclosing_names.contains(&name) {
                        tensor_ref(name, Src::Intermediate, 0, None)
                    } else {
                        return Err(format!("MLX could not resolve subgraph input {name}"));
                    }
                }
            });
        }
        for value in &metadata.output_slots {
            node.outputs.push(match value {
                None => OutRef {
                    name: String::new(),
                    external: false,
                    ctx_index: 0,
                    otype: 0,
                },
                Some(value) => {
                    let name = snapshot.values[&value].name.clone();
                    let external = root && ctx_outputs.contains_key(value);
                    OutRef {
                        otype: snapshot.values[&value]
                            .tensor
                            .as_ref()
                            .and_then(|tensor| tensor.dtype)
                            .map_or(0, |dtype| dtype.to_onnx() as u32),
                        ctx_index: if external { ctx_outputs[value] } else { 0 },
                        name,
                        external,
                    }
                }
            });
        }
        let mut subgraphs = snapshot
            .subgraphs
            .iter()
            .filter(|((parent, _), _)| *parent == id)
            .collect::<Vec<_>>();
        subgraphs.sort_unstable_by(|((_, left), _), ((_, right), _)| left.cmp(right));
        for ((_, attr_name), child) in subgraphs {
            let child_sidecars = &sidecars.subgraphs[&(id, attr_name.clone())];
            let child_nodes = build_graph(
                api,
                child,
                child_sidecars,
                ctx_inputs,
                ctx_input_names,
                &HashMap::new(),
                &visible,
                &inits,
                false,
            )?;
            node.subgraphs.push(SubgraphDesc {
                attr_name: attr_name.clone(),
                input_names: child
                    .graph_input_slots
                    .iter()
                    .map(|slot| {
                        slot.map_or_else(String::new, |value| child.values[&value].name.clone())
                    })
                    .collect(),
                output_names: child
                    .graph_output_slots
                    .iter()
                    .map(|slot| {
                        slot.map_or_else(String::new, |value| child.values[&value].name.clone())
                    })
                    .collect(),
                nodes: child_nodes,
            });
        }
        nodes.push(node);
    }
    mark_shape_constants(&mut nodes, &inits.keys().cloned().collect());
    Ok(nodes)
}

fn tensor_ref(name: String, source: Src, ctx_index: usize, init: Option<InitData>) -> TensorRef {
    TensorRef {
        name,
        source,
        ctx_index,
        constant: false,
        shape_const: false,
        init,
    }
}

fn mark_shape_constants(nodes: &mut [NodeDesc], inherited: &HashSet<String>) {
    let mut constants = inherited.clone();
    for node in nodes {
        for input in &mut node.inputs {
            input.shape_const = !input.name.is_empty() && constants.contains(&input.name);
        }
        for subgraph in &mut node.subgraphs {
            let mut inner = constants.clone();
            for name in &subgraph.input_names {
                inner.remove(name);
            }
            for body_node in &subgraph.nodes {
                for output in &body_node.outputs {
                    inner.remove(&output.name);
                }
            }
            mark_shape_constants(&mut subgraph.nodes, &inner);
        }
        let random = matches!(
            node.op_type.as_str(),
            "RandomNormal"
                | "RandomUniform"
                | "RandomNormalLike"
                | "RandomUniformLike"
                | "Bernoulli"
                | "Multinomial"
                | "Dropout"
        );
        if node.subgraphs.is_empty()
            && (matches!(node.op_type.as_str(), "Shape" | "Size")
                || (!random
                    && node.inputs.iter().all(|input| {
                        matches!(input.source, Src::Absent | Src::Initializer)
                            || input.constant
                            || input.shape_const
                    })))
        {
            for output in &node.outputs {
                if !output.name.is_empty() {
                    constants.insert(output.name.clone());
                }
            }
        }
    }
}

fn any_control_flow(nodes: &[NodeDesc]) -> bool {
    nodes.iter().any(|node| {
        matches!(
            node.op_type.as_str(),
            "If" | "Loop" | "Scan" | "SequenceMap"
        ) || node
            .subgraphs
            .iter()
            .any(|subgraph| any_control_flow(&subgraph.nodes))
    })
}

fn has_compile_barrier(nodes: &[NodeDesc]) -> bool {
    nodes.iter().any(|node| {
        node.op_type == "PackedMultiHeadAttention"
            || node
                .subgraphs
                .iter()
                .any(|subgraph| has_compile_barrier(&subgraph.nodes))
    })
}

fn snapshot_initializer_data(
    snapshot: &OrtGraphSnapshot,
    value: ValueId,
    data: *const c_void,
) -> Result<InitData, String> {
    let metadata = &snapshot.initializers[&value];
    let name = &snapshot.values[&value].name;
    let dtype = metadata
        .dtype
        .ok_or_else(|| format!("initializer {name} has no tensor dtype"))?
        .to_onnx() as u32;
    let shape = metadata
        .shape
        .as_ref()
        .ok_or_else(|| format!("initializer {name} has no tensor shape"))?
        .iter()
        .map(|dimension| match dimension {
            DimensionMetadata::Static(value) => i64::try_from(*value)
                .map_err(|_| format!("initializer {name} dimension overflows i64")),
            DimensionMetadata::Symbolic(_) => {
                Err(format!("initializer {name} has a symbolic dimension"))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let count = shape.iter().try_fold(1usize, |count, dimension| {
        usize::try_from(*dimension)
            .ok()
            .and_then(|dimension| count.checked_mul(dimension))
            .ok_or_else(|| format!("initializer {name} element count overflows"))
    })?;
    Ok(InitData {
        data,
        shape,
        dtype,
        count,
        owned: None,
    })
}

unsafe fn collect_initializer_bytes(
    api: &ort::OrtApi,
    graph: *const ort::OrtGraph,
) -> Result<HashMap<String, *const c_void>, String> {
    let mut count = 0;
    check(
        api,
        unsafe { (api.Graph_GetNumInitializers.unwrap())(graph, &mut count) },
        "reading initializers",
    )?;
    let mut infos = vec![ptr::null(); count];
    if count != 0 {
        check(
            api,
            unsafe { (api.Graph_GetInitializers.unwrap())(graph, infos.as_mut_ptr(), count) },
            "reading initializers",
        )?;
    }
    let mut result = HashMap::new();
    for info in infos {
        let name = unsafe { value_name(api, info)? };
        let mut value = ptr::null();
        let status = unsafe { (api.ValueInfo_GetInitializerValue.unwrap())(info, &mut value) };
        if !status.is_null() {
            unsafe { release_status(api, status) };
            continue;
        }
        if value.is_null() {
            continue;
        }
        let mut data = ptr::null();
        check(
            api,
            unsafe { (api.GetTensorData.unwrap())(value, &mut data) },
            "reading initializer bytes",
        )?;
        result.insert(name, data);
    }
    Ok(result)
}

unsafe fn initializer_data(
    api: &ort::OrtApi,
    value: *const ort::OrtValue,
) -> Result<InitData, String> {
    let mut info = ptr::null_mut();
    check(
        api,
        unsafe { (api.GetTensorTypeAndShape.unwrap())(value, &mut info) },
        "reading initializer shape",
    )?;
    let result = (|| unsafe {
        let mut rank = 0;
        check(
            api,
            (api.GetDimensionsCount.unwrap())(info, &mut rank),
            "reading initializer rank",
        )?;
        let mut shape = vec![0; rank];
        if rank != 0 {
            check(
                api,
                (api.GetDimensions.unwrap())(info, shape.as_mut_ptr(), rank),
                "reading initializer dimensions",
            )?;
        }
        let mut dtype = 0;
        check(
            api,
            (api.GetTensorElementType.unwrap())(info, &mut dtype),
            "reading initializer type",
        )?;
        let mut count = 0;
        check(
            api,
            (api.GetTensorShapeElementCount.unwrap())(info, &mut count),
            "reading initializer count",
        )?;
        let mut data = ptr::null();
        check(
            api,
            (api.GetTensorData.unwrap())(value, &mut data),
            "reading initializer data",
        )?;
        Ok(InitData {
            data,
            shape,
            dtype,
            count,
            owned: None,
        })
    })();
    unsafe { (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info) };
    result
}

unsafe fn own_init_data(source: &InitData) -> Result<InitData, String> {
    if source.owned.is_some() || source.data.is_null() {
        return Ok(source.clone());
    }
    let dtype = onnx_runtime_ir::DataType::from_onnx(source.dtype as i32)
        .ok_or_else(|| format!("initializer has unsupported dtype {}", source.dtype))?;
    let bytes = dtype
        .checked_storage_bytes(source.count)
        .ok_or_else(|| "initializer byte count overflows".to_string())?;
    let owned =
        Arc::new(unsafe { std::slice::from_raw_parts(source.data.cast::<u8>(), bytes) }.to_vec());
    Ok(InitData {
        data: owned.as_ptr().cast(),
        shape: source.shape.clone(),
        dtype: source.dtype,
        count: source.count,
        owned: Some(owned),
    })
}

fn element_byte_size(dtype: ort::ONNXTensorElementDataType) -> usize {
    onnx_runtime_ir::DataType::from_onnx(dtype as i32).map_or(0, |dtype| dtype.byte_size())
}

unsafe fn collect_attributes(api: &ort::OrtApi, node: *const ort::OrtNode, desc: &mut NodeDesc) {
    let mut count = 0;
    unsafe { (api.Node_GetNumAttributes.unwrap())(node, &mut count) };
    let mut attrs = vec![ptr::null(); count];
    if count != 0 {
        unsafe { (api.Node_GetAttributes.unwrap())(node, attrs.as_mut_ptr(), count) };
    }
    for attr in attrs {
        if attr.is_null() {
            continue;
        }
        let mut name = ptr::null();
        unsafe { (api.OpAttr_GetName.unwrap())(attr, &mut name) };
        let name = cstr(name);
        let mut ty = 0;
        unsafe { (api.OpAttr_GetType.unwrap())(attr, &mut ty) };
        if ty == ort::OrtOpAttrType_ORT_OP_ATTR_INT {
            if let Some(value) = unsafe { read_scalar::<i64>(api, attr, ty) } {
                desc.ints.insert(name, value);
            }
        } else if ty == ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT {
            if let Some(value) = unsafe { read_scalar::<f32>(api, attr, ty) } {
                desc.floats.insert(name, value);
            }
        } else if ty == ort::OrtOpAttrType_ORT_OP_ATTR_INTS {
            if let Some(value) = unsafe { read_array::<i64>(api, attr, ty) } {
                desc.int_arrays.insert(name, value);
            }
        } else if ty == ort::OrtOpAttrType_ORT_OP_ATTR_FLOATS {
            if let Some(value) = unsafe { read_array::<f32>(api, attr, ty) } {
                desc.float_arrays.insert(name, value);
            }
        } else if ty == ort::OrtOpAttrType_ORT_OP_ATTR_STRING {
            if let Some(value) = unsafe { read_string(api, attr, ty) } {
                desc.strings.insert(name, value);
            }
        } else if ty == ort::OrtOpAttrType_ORT_OP_ATTR_TENSOR {
            if let Some(value) = unsafe { read_tensor_attr(api, attr) } {
                desc.tensors.insert(name, value);
            }
        }
    }
}

unsafe fn read_scalar<T: Copy + Default>(
    api: &ort::OrtApi,
    attr: *const ort::OrtOpAttr,
    ty: ort::OrtOpAttrType,
) -> Option<T> {
    let mut value = T::default();
    let mut written = 0;
    let status = unsafe {
        (api.ReadOpAttr.unwrap())(
            attr,
            ty,
            (&mut value as *mut T).cast(),
            std::mem::size_of::<T>(),
            &mut written,
        )
    };
    if status.is_null() {
        Some(value)
    } else {
        unsafe { release_status(api, status) };
        None
    }
}

unsafe fn read_array<T: Copy + Default>(
    api: &ort::OrtApi,
    attr: *const ort::OrtOpAttr,
    ty: ort::OrtOpAttrType,
) -> Option<Vec<T>> {
    let mut bytes = 0;
    unsafe {
        release_status(
            api,
            (api.ReadOpAttr.unwrap())(attr, ty, ptr::null_mut(), 0, &mut bytes),
        )
    };
    let mut result = vec![T::default(); bytes / std::mem::size_of::<T>()];
    let mut written = 0;
    let status = unsafe {
        (api.ReadOpAttr.unwrap())(attr, ty, result.as_mut_ptr().cast(), bytes, &mut written)
    };
    if status.is_null() {
        Some(result)
    } else {
        unsafe { release_status(api, status) };
        None
    }
}

unsafe fn read_string(
    api: &ort::OrtApi,
    attr: *const ort::OrtOpAttr,
    ty: ort::OrtOpAttrType,
) -> Option<String> {
    let mut bytes = 0;
    unsafe {
        release_status(
            api,
            (api.ReadOpAttr.unwrap())(attr, ty, ptr::null_mut(), 0, &mut bytes),
        )
    };
    let mut result = vec![0; bytes];
    let mut written = 0;
    let status = unsafe {
        (api.ReadOpAttr.unwrap())(attr, ty, result.as_mut_ptr().cast(), bytes, &mut written)
    };
    if status.is_null() {
        result.truncate(written.min(bytes));
        String::from_utf8(result).ok()
    } else {
        unsafe { release_status(api, status) };
        None
    }
}

unsafe fn read_tensor_attr(api: &ort::OrtApi, attr: *const ort::OrtOpAttr) -> Option<ConstTensor> {
    let mut value = ptr::null_mut();
    let status = unsafe { (api.OpAttr_GetTensorAttributeAsOrtValue.unwrap())(attr, &mut value) };
    if !status.is_null() || value.is_null() {
        unsafe { release_status(api, status) };
        return None;
    }
    let result = unsafe { initializer_data(api, value) }
        .ok()
        .and_then(|data| {
            let width = element_byte_size(data.dtype);
            (!data.data.is_null())
                .then(|| data.count.checked_mul(width))
                .flatten()
                .filter(|_| width != 0)
                .map(|bytes| ConstTensor {
                    data: unsafe { std::slice::from_raw_parts(data.data.cast(), bytes) }.to_vec(),
                    shape: data.shape,
                    dtype: data.dtype,
                    count: data.count,
                })
        });
    unsafe { (api.ReleaseValue.unwrap())(value) };
    result
}

fn cstr(value: *const c_char) -> String {
    if value.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(value).to_string_lossy().into_owned() }
    }
}

unsafe fn value_name(api: &ort::OrtApi, value: *const ort::OrtValueInfo) -> Result<String, String> {
    let mut name = ptr::null();
    check(
        api,
        unsafe { (api.GetValueInfoName.unwrap())(value, &mut name) },
        "reading initializer name",
    )?;
    Ok(cstr(name))
}

fn check(api: &ort::OrtApi, status: *mut ort::OrtStatus, what: &str) -> Result<(), String> {
    if status.is_null() {
        return Ok(());
    }
    let message = unsafe { cstr((api.GetErrorMessage.unwrap())(status)) };
    unsafe { release_status(api, status) };
    Err(format!("{what}: {message}"))
}

unsafe fn release_status(api: &ort::OrtApi, status: *mut ort::OrtStatus) {
    if !status.is_null() {
        unsafe { (api.ReleaseStatus.unwrap())(status) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{DataType, Graph, Node};

    fn output(name: &str) -> OutRef {
        OutRef {
            name: name.to_string(),
            external: false,
            ctx_index: 0,
            otype: 0,
        }
    }

    fn node(op_type: &str, inputs: Vec<TensorRef>, outputs: Vec<OutRef>) -> NodeDesc {
        let mut node = NodeDesc::new(
            op_type.to_string(),
            String::new(),
            13,
            crate::registry::CompileShapeSafety::Shapeless,
        );
        node.inputs = inputs;
        node.outputs = outputs;
        node
    }

    #[test]
    fn boundaries_preserve_optional_slots_and_original_indices() {
        let input = ValueId(0);
        let output = ValueId(1);
        let snapshot = OrtGraphSnapshot {
            ir: Graph::new(),
            onnx_ir_version: 0,
            nodes: HashMap::new(),
            values: HashMap::from([
                (
                    input,
                    crate::ort_graph::ValueMetadata {
                        name: "input".to_string(),
                        type_kind: crate::ort_graph::ValueTypeKind::Unknown,
                        tensor: None,
                    },
                ),
                (
                    output,
                    crate::ort_graph::ValueMetadata {
                        name: "output".to_string(),
                        type_kind: crate::ort_graph::ValueTypeKind::Unknown,
                        tensor: None,
                    },
                ),
            ]),
            node_by_ort_id: HashMap::new(),
            value_by_name: HashMap::new(),
            graph_input_slots: vec![None, Some(input)],
            graph_output_slots: vec![None, Some(output)],
            initializers: HashMap::new(),
            node_schema_versions: HashMap::new(),
            subgraphs: HashMap::new(),
        };
        assert_eq!(snapshot.graph_input_slots, vec![None, Some(input)]);
        assert_eq!(snapshot.graph_output_slots, vec![None, Some(output)]);
    }

    #[test]
    fn initializer_copy_owns_bytes() {
        let bytes = [1u8, 2, 3, 4];
        let source = InitData {
            data: bytes.as_ptr().cast(),
            shape: vec![4],
            dtype: ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
            count: bytes.len(),
            owned: None,
        };
        let copied = unsafe { own_init_data(&source) }.unwrap();
        assert_eq!(copied.owned.as_deref().unwrap().as_slice(), bytes);
        assert_ne!(copied.data, source.data);
    }

    #[test]
    fn shape_constants_flow_into_nested_enclosing_capture() {
        let outer = tensor_ref("outer_shape".to_string(), Src::Initializer, 0, None);
        let nested = node(
            "Cast",
            vec![tensor_ref(
                "outer_shape".to_string(),
                Src::Intermediate,
                0,
                None,
            )],
            vec![output("nested_out")],
        );
        let mut parent = node("If", vec![outer], vec![output("parent_out")]);
        parent.subgraphs.push(SubgraphDesc {
            attr_name: "then_branch".to_string(),
            input_names: vec!["formal".to_string()],
            output_names: vec!["nested_out".to_string()],
            nodes: vec![nested],
        });
        let mut nodes = vec![parent];
        mark_shape_constants(&mut nodes, &HashSet::from(["outer_shape".to_string()]));
        assert!(nodes[0].inputs[0].shape_const);
        assert!(nodes[0].subgraphs[0].nodes[0].inputs[0].shape_const);
    }

    #[test]
    fn ir_topological_order_is_stable_over_insertion_order() {
        let mut graph = Graph::new();
        let input = graph.create_named_value("input", DataType::Float32, vec![]);
        let intermediate = graph.create_named_value("intermediate", DataType::Float32, vec![]);
        let output = graph.create_named_value("output", DataType::Float32, vec![]);
        let consumer = graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(intermediate)],
            vec![output],
        ));
        let producer = graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(input)],
            vec![intermediate],
        ));
        assert_eq!(graph.topological_order().unwrap(), vec![producer, consumer]);
    }
}
