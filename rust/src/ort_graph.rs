//! A safe, owned snapshot of the optimized `OrtGraph` exposed to a plugin EP.
//!
//! This is deliberately not an ONNX protobuf loader: ORT has already optimized
//! and partitioned the graph when this boundary is called. Initializer *metadata*
//! is captured here, but initializer bytes remain owned and copied by `plan_builder::build_plan`
//! during Compile because ORT's borrowed initializer storage cannot outlive that call.

use std::collections::{BTreeSet, HashMap};
use std::ffi::{CStr, c_char};

use onnx_runtime_ir::{DataType, Dim, Graph, Node, NodeId, Shape, ValueId, normalize_domain};

use crate::sys::ort;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ValueTypeKind {
    Tensor,
    Sequence,
    Optional,
    Other,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DimensionMetadata {
    Static(usize),
    Symbolic(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TensorMetadata {
    pub dtype: Option<DataType>,
    pub shape: Option<Vec<DimensionMetadata>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ValueMetadata {
    pub name: String,
    pub type_kind: ValueTypeKind,
    pub tensor: Option<TensorMetadata>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NodeMetadata {
    /// ORT's stable node ID, not a borrowed `OrtNode` pointer.
    pub ort_node_id: usize,
    pub name: String,
    pub domain: String,
    pub op_type: String,
    /// The operator schema version selected by ORT. This is not a graph opset import.
    pub since_version: i32,
    pub input_slots: Vec<Option<ValueId>>,
    pub output_slots: Vec<Option<ValueId>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InitializerMetadata {
    pub value: ValueId,
    pub dtype: Option<DataType>,
    pub shape: Option<Vec<DimensionMetadata>>,
}

/// A fully-owned representation of one optimized ORT graph.
///
/// `ir` contains the graph topology. The accompanying maps preserve the stable
/// names and ORT node IDs that identify the optimized ABI objects without
/// extending their lifetimes. `node_schema_versions` records the schema version
/// selected by ORT for each node domain; `ir.opset_imports` is populated from
/// ORT's declared graph operator-set table.
#[derive(Clone, Debug)]
pub(crate) struct OrtGraphSnapshot {
    pub ir: Graph,
    pub onnx_ir_version: i64,
    pub nodes: HashMap<NodeId, NodeMetadata>,
    pub values: HashMap<ValueId, ValueMetadata>,
    pub node_by_ort_id: HashMap<usize, NodeId>,
    pub value_by_name: HashMap<String, ValueId>,
    pub graph_input_slots: Vec<Option<ValueId>>,
    pub graph_output_slots: Vec<Option<ValueId>>,
    pub initializers: HashMap<ValueId, InitializerMetadata>,
    pub node_schema_versions: HashMap<String, BTreeSet<i32>>,
    pub subgraphs: HashMap<(NodeId, String), Box<OrtGraphSnapshot>>,
}

impl OrtGraphSnapshot {
    /// Materialize a safe snapshot while `graph` and all ABI pointers are live.
    ///
    /// # Safety
    /// `api` and `graph` must be valid for this call, as supplied by ORT's
    /// `Compile` callback. No borrowed pointers are retained in the result.
    pub(crate) unsafe fn from_ort(
        api: &ort::OrtApi,
        graph: *const ort::OrtGraph,
    ) -> Result<Self, String> {
        if graph.is_null() {
            return Err("ORT supplied a null graph".to_string());
        }
        unsafe { Capture::new(api).capture_graph(graph) }
    }
}

struct Capture<'a> {
    api: &'a ort::OrtApi,
    snapshot: OrtGraphSnapshot,
}

impl<'a> Capture<'a> {
    fn new(api: &'a ort::OrtApi) -> Self {
        Self {
            api,
            snapshot: OrtGraphSnapshot {
                ir: Graph::new(),
                onnx_ir_version: 0,
                nodes: HashMap::new(),
                values: HashMap::new(),
                node_by_ort_id: HashMap::new(),
                value_by_name: HashMap::new(),
                graph_input_slots: Vec::new(),
                graph_output_slots: Vec::new(),
                initializers: HashMap::new(),
                node_schema_versions: HashMap::new(),
                subgraphs: HashMap::new(),
            },
        }
    }

    unsafe fn capture_graph(
        mut self,
        graph: *const ort::OrtGraph,
    ) -> Result<OrtGraphSnapshot, String> {
        unsafe { self.capture_graph_versions(graph)? };
        let input_values = unsafe {
            self.graph_values(
                graph,
                self.api.Graph_GetNumInputs.unwrap(),
                self.api.Graph_GetInputs.unwrap(),
                "graph inputs",
            )?
        };
        for &value in input_values.iter().flatten() {
            self.snapshot.ir.add_input(value);
        }
        self.snapshot.graph_input_slots = input_values;

        let output_values = unsafe {
            self.graph_values(
                graph,
                self.api.Graph_GetNumOutputs.unwrap(),
                self.api.Graph_GetOutputs.unwrap(),
                "graph outputs",
            )?
        };
        for &value in output_values.iter().flatten() {
            self.snapshot.ir.add_output(value);
        }
        self.snapshot.graph_output_slots = output_values;

        let initializer_values = unsafe {
            self.graph_values(
                graph,
                self.api.Graph_GetNumInitializers.unwrap(),
                self.api.Graph_GetInitializers.unwrap(),
                "graph initializers",
            )?
        };
        for value in initializer_values.into_iter().flatten() {
            let metadata = self.snapshot.values[&value].clone();
            self.snapshot.initializers.insert(
                value,
                InitializerMetadata {
                    value,
                    dtype: metadata.tensor.as_ref().and_then(|tensor| tensor.dtype),
                    shape: metadata
                        .tensor
                        .as_ref()
                        .and_then(|tensor| tensor.shape.clone()),
                },
            );
        }

        let mut count = 0;
        unsafe {
            self.status(
                (self.api.Graph_GetNumNodes.unwrap())(graph, &mut count),
                "reading node count",
            )?
        };
        let mut raw_nodes = vec![std::ptr::null(); count];
        if count > 0 {
            unsafe {
                self.status(
                    (self.api.Graph_GetNodes.unwrap())(graph, raw_nodes.as_mut_ptr(), count),
                    "reading nodes",
                )?
            };
        }
        for raw_node in raw_nodes {
            unsafe { self.capture_node(raw_node)? };
        }

        // `Graph::initializers` requires a `WeightRef`, which would imply an
        // execution-owned byte copy. This metadata-only layer intentionally
        // leaves it empty, so a valid graph that returns an initializer directly
        // cannot satisfy `Graph::validate` until a later ownership layer joins
        // the existing Compile-time initializer copies.
        Ok(self.snapshot)
    }

    unsafe fn capture_node(&mut self, node: *const ort::OrtNode) -> Result<NodeId, String> {
        if node.is_null() {
            return Err("ORT graph contains a null node".to_string());
        }
        let input_slots = unsafe { self.node_values(node, true)? };
        let output_slots = unsafe { self.node_values(node, false)? };
        // `Graph` requires a dense output vector. Keep only present values in
        // that projection, while `output_slots` retains ORT's positional holes.
        let outputs: Vec<ValueId> = output_slots.iter().flatten().copied().collect();

        let ort_node_id = unsafe { self.node_id(node)? };
        let name = unsafe { self.node_string(node, self.api.Node_GetName.unwrap(), "node name")? };
        let domain =
            unsafe { self.node_string(node, self.api.Node_GetDomain.unwrap(), "node domain")? };
        let op_type = unsafe {
            self.node_string(
                node,
                self.api.Node_GetOperatorType.unwrap(),
                "node operator type",
            )?
        };
        let mut since_version = 0;
        unsafe {
            self.status(
                (self.api.Node_GetSinceVersion.unwrap())(node, &mut since_version),
                "reading node schema version",
            )?
        };
        let domain = normalize_domain(&domain).to_string();
        let ir_node = Node::new(NodeId(0), op_type.clone(), input_slots.clone(), outputs);
        let node_id = self.snapshot.ir.insert_node(ir_node);
        {
            let ir_node = self.snapshot.ir.node_mut(node_id);
            ir_node.name = name.clone();
            ir_node.domain = domain.clone();
        }
        self.snapshot.node_by_ort_id.insert(ort_node_id, node_id);
        self.snapshot.nodes.insert(
            node_id,
            NodeMetadata {
                ort_node_id,
                name,
                domain: domain.clone(),
                op_type,
                since_version,
                input_slots,
                output_slots,
            },
        );
        self.snapshot
            .node_schema_versions
            .entry(domain)
            .or_default()
            .insert(since_version);

        let mut subgraph_count = 0;
        unsafe {
            self.status(
                (self.api.Node_GetNumSubgraphs.unwrap())(node, &mut subgraph_count),
                "reading node subgraph count",
            )?
        };
        if subgraph_count > 0 {
            let mut graphs = vec![std::ptr::null(); subgraph_count];
            let mut names = vec![std::ptr::null(); subgraph_count];
            unsafe {
                self.status(
                    (self.api.Node_GetSubgraphs.unwrap())(
                        node,
                        graphs.as_mut_ptr(),
                        subgraph_count,
                        names.as_mut_ptr(),
                    ),
                    "reading node subgraphs",
                )?
            };
            for (raw_graph, raw_name) in graphs.into_iter().zip(names) {
                let attr_name = unsafe { Self::cstr(raw_name) };
                if raw_graph.is_null() || attr_name.is_empty() {
                    return Err("ORT returned an invalid control-flow subgraph".to_string());
                }
                let child = unsafe { OrtGraphSnapshot::from_ort(self.api, raw_graph)? };
                self.snapshot
                    .ir
                    .subgraphs
                    .insert((node_id, attr_name.clone()), child.ir.clone());
                self.snapshot
                    .subgraphs
                    .insert((node_id, attr_name), Box::new(child));
            }
        }
        Ok(node_id)
    }

    unsafe fn capture_graph_versions(&mut self, graph: *const ort::OrtGraph) -> Result<(), String> {
        let mut onnx_ir_version = 0;
        unsafe {
            self.status(
                (self.api.Graph_GetOnnxIRVersion.unwrap())(graph, &mut onnx_ir_version),
                "reading ONNX IR version",
            )?
        };
        self.snapshot.onnx_ir_version = onnx_ir_version;
        let mut count = 0;
        unsafe {
            self.status(
                (self.api.Graph_GetNumOperatorSets.unwrap())(graph, &mut count),
                "reading graph operator-set count",
            )?
        };
        let mut domains = vec![std::ptr::null(); count];
        let mut versions = vec![0i64; count];
        if count > 0 {
            unsafe {
                self.status(
                    (self.api.Graph_GetOperatorSets.unwrap())(
                        graph,
                        domains.as_mut_ptr(),
                        versions.as_mut_ptr(),
                        count,
                    ),
                    "reading graph operator sets",
                )?
            };
        }
        for (domain, version) in domains.into_iter().zip(versions) {
            let domain = normalize_domain(&unsafe { Self::cstr(domain) }).to_string();
            let version = u64::try_from(version)
                .ok()
                .filter(|version| *version != 0)
                .ok_or_else(|| {
                    format!("ORT reported invalid opset version {version} for {domain:?}")
                })?;
            if let Some(previous) = self
                .snapshot
                .ir
                .opset_imports
                .insert(domain.clone(), version)
                && previous != version
            {
                return Err(format!(
                    "ORT reported conflicting opset versions {previous} and {version} for {domain:?}"
                ));
            }
        }
        Ok(())
    }

    unsafe fn graph_values(
        &mut self,
        graph: *const ort::OrtGraph,
        count_fn: unsafe extern "C" fn(*const ort::OrtGraph, *mut usize) -> *mut ort::OrtStatus,
        get_fn: unsafe extern "C" fn(
            *const ort::OrtGraph,
            *mut *const ort::OrtValueInfo,
            usize,
        ) -> *mut ort::OrtStatus,
        what: &str,
    ) -> Result<Vec<Option<ValueId>>, String> {
        let mut count = 0;
        unsafe { self.status(count_fn(graph, &mut count), what)? };
        let mut values = vec![std::ptr::null(); count];
        if count > 0 {
            unsafe { self.status(get_fn(graph, values.as_mut_ptr(), count), what)? };
        }
        values
            .into_iter()
            .map(|value| unsafe { self.ensure_value(value) })
            .collect()
    }

    unsafe fn node_values(
        &mut self,
        node: *const ort::OrtNode,
        inputs: bool,
    ) -> Result<Vec<Option<ValueId>>, String> {
        let mut count = 0;
        if inputs {
            unsafe {
                self.status(
                    (self.api.Node_GetNumInputs.unwrap())(node, &mut count),
                    "reading node input count",
                )?
            };
        } else {
            unsafe {
                self.status(
                    (self.api.Node_GetNumOutputs.unwrap())(node, &mut count),
                    "reading node output count",
                )?
            };
        }
        let mut values = vec![std::ptr::null(); count];
        if count > 0 {
            if inputs {
                unsafe {
                    self.status(
                        (self.api.Node_GetInputs.unwrap())(node, values.as_mut_ptr(), count),
                        "reading node inputs",
                    )?
                };
            } else {
                unsafe {
                    self.status(
                        (self.api.Node_GetOutputs.unwrap())(node, values.as_mut_ptr(), count),
                        "reading node outputs",
                    )?
                };
            }
        }
        values
            .into_iter()
            .map(|value| unsafe { self.ensure_value(value) })
            .collect()
    }

    unsafe fn ensure_value(
        &mut self,
        value_info: *const ort::OrtValueInfo,
    ) -> Result<Option<ValueId>, String> {
        if value_info.is_null() {
            return Ok(None);
        }
        let name = unsafe { self.value_name(value_info)? };
        if name.is_empty() {
            return Ok(None);
        }
        if let Some(&value) = self.snapshot.value_by_name.get(&name) {
            return Ok(Some(value));
        }
        let metadata = unsafe { self.value_metadata(value_info, &name)? };
        let (dtype, shape, type_known, shape_known) = self.ir_value_type(&metadata);
        let value = self
            .snapshot
            .ir
            .create_named_value(name.clone(), dtype, shape);
        if !type_known {
            self.snapshot.ir.mark_value_type_unknown(value);
        }
        if !shape_known {
            self.snapshot.ir.mark_value_shape_unknown(value);
        }
        self.snapshot.value_by_name.insert(name, value);
        self.snapshot.values.insert(value, metadata);
        Ok(Some(value))
    }

    fn ir_value_type(&mut self, metadata: &ValueMetadata) -> (DataType, Shape, bool, bool) {
        let Some(tensor) = &metadata.tensor else {
            return (DataType::Undefined, Vec::new(), false, false);
        };
        let type_known = tensor.dtype.is_some();
        let shape_known = tensor.shape.is_some();
        let shape = tensor.shape.as_ref().map_or_else(Vec::new, |dimensions| {
            dimensions
                .iter()
                .map(|dimension| match dimension {
                    DimensionMetadata::Static(size) => Dim::Static(*size),
                    DimensionMetadata::Symbolic(name) => {
                        Dim::Symbolic(self.snapshot.ir.intern_symbol(name))
                    }
                })
                .collect()
        });
        (
            tensor.dtype.unwrap_or(DataType::Undefined),
            shape,
            type_known,
            shape_known,
        )
    }

    unsafe fn value_metadata(
        &mut self,
        value_info: *const ort::OrtValueInfo,
        value_name: &str,
    ) -> Result<ValueMetadata, String> {
        let mut type_info = std::ptr::null();
        let status =
            unsafe { (self.api.GetValueInfoTypeInfo.unwrap())(value_info, &mut type_info) };
        if !status.is_null() || type_info.is_null() {
            unsafe { self.release_status(status) };
            return Ok(ValueMetadata {
                name: value_name.to_string(),
                type_kind: ValueTypeKind::Unknown,
                tensor: None,
            });
        }
        let mut onnx_type = 0;
        let status =
            unsafe { (self.api.GetOnnxTypeFromTypeInfo.unwrap())(type_info, &mut onnx_type) };
        if !status.is_null() {
            unsafe { self.release_status(status) };
            return Ok(ValueMetadata {
                name: value_name.to_string(),
                type_kind: ValueTypeKind::Unknown,
                tensor: None,
            });
        }
        if onnx_type != ort::ONNXType_ONNX_TYPE_TENSOR {
            return Ok(ValueMetadata {
                name: value_name.to_string(),
                type_kind: match onnx_type {
                    value if value == ort::ONNXType_ONNX_TYPE_SEQUENCE => ValueTypeKind::Sequence,
                    value if value == ort::ONNXType_ONNX_TYPE_OPTIONAL => ValueTypeKind::Optional,
                    _ => ValueTypeKind::Other,
                },
                tensor: None,
            });
        }
        let mut tensor_info = std::ptr::null();
        unsafe {
            self.status(
                (self.api.CastTypeInfoToTensorInfo.unwrap())(type_info, &mut tensor_info),
                "reading tensor type information",
            )?
        };
        if tensor_info.is_null() {
            return Err("ORT returned null tensor type information".to_string());
        }
        let mut raw_dtype = 0;
        unsafe {
            self.status(
                (self.api.GetTensorElementType.unwrap())(tensor_info, &mut raw_dtype),
                "reading tensor element type",
            )?
        };
        let dtype = i32::try_from(raw_dtype).ok().and_then(DataType::from_onnx);
        let shape = unsafe { self.tensor_shape(tensor_info, value_name)? };
        Ok(ValueMetadata {
            name: value_name.to_string(),
            type_kind: ValueTypeKind::Tensor,
            tensor: Some(TensorMetadata { dtype, shape }),
        })
    }

    unsafe fn tensor_shape(
        &mut self,
        tensor_info: *const ort::OrtTensorTypeAndShapeInfo,
        value_name: &str,
    ) -> Result<Option<Vec<DimensionMetadata>>, String> {
        let mut count = 0;
        unsafe {
            self.status(
                (self.api.GetDimensionsCount.unwrap())(tensor_info, &mut count),
                "reading tensor rank",
            )?
        };
        let mut dimensions = vec![0i64; count];
        if count > 0 {
            unsafe {
                self.status(
                    (self.api.GetDimensions.unwrap())(tensor_info, dimensions.as_mut_ptr(), count),
                    "reading tensor dimensions",
                )?
            };
        }
        let mut symbolic = vec![std::ptr::null::<c_char>(); count];
        if count > 0
            && let Some(get_symbolic_dimensions) = self.api.GetSymbolicDimensions
        {
            let status = unsafe {
                get_symbolic_dimensions(tensor_info, symbolic.as_mut_ptr(), symbolic.len())
            };
            if !status.is_null() {
                unsafe { self.release_status(status) };
                symbolic.fill(std::ptr::null());
            }
        }
        Ok(Some(
            dimensions
                .into_iter()
                .enumerate()
                .map(|(index, dimension)| {
                    if dimension >= 0 {
                        DimensionMetadata::Static(dimension as usize)
                    } else {
                        let name = unsafe { Self::cstr(symbolic[index]) };
                        DimensionMetadata::Symbolic(if name.is_empty() {
                            format!("__ort_unknown_{value_name}_{index}")
                        } else {
                            name
                        })
                    }
                })
                .collect(),
        ))
    }

    unsafe fn value_name(&self, value_info: *const ort::OrtValueInfo) -> Result<String, String> {
        let mut name = std::ptr::null();
        unsafe {
            self.status(
                (self.api.GetValueInfoName.unwrap())(value_info, &mut name),
                "reading value name",
            )?
        };
        Ok(unsafe { Self::cstr(name) })
    }

    unsafe fn node_id(&self, node: *const ort::OrtNode) -> Result<usize, String> {
        let mut id = 0;
        unsafe {
            self.status(
                (self.api.Node_GetId.unwrap())(node, &mut id),
                "reading node ID",
            )?
        };
        Ok(id)
    }

    unsafe fn node_string(
        &self,
        node: *const ort::OrtNode,
        get: unsafe extern "C" fn(*const ort::OrtNode, *mut *const c_char) -> *mut ort::OrtStatus,
        what: &str,
    ) -> Result<String, String> {
        let mut value = std::ptr::null();
        unsafe { self.status(get(node, &mut value), what)? };
        Ok(unsafe { Self::cstr(value) })
    }

    unsafe fn status(&self, status: *mut ort::OrtStatus, what: &str) -> Result<(), String> {
        if status.is_null() {
            return Ok(());
        }
        let message = unsafe {
            let raw = (self.api.GetErrorMessage.unwrap())(status);
            Self::cstr(raw)
        };
        unsafe { self.release_status(status) };
        Err(format!("{what}: {message}"))
    }

    unsafe fn release_status(&self, status: *mut ort::OrtStatus) {
        if !status.is_null() {
            unsafe { (self.api.ReleaseStatus.unwrap())(status) };
        }
    }

    unsafe fn cstr(value: *const c_char) -> String {
        if value.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(value).to_string_lossy().into_owned() }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{SymbolConstraints, SymbolId};

    fn tensor(dtype: DataType, shape: &[DimensionMetadata]) -> ValueMetadata {
        ValueMetadata {
            name: String::new(),
            type_kind: ValueTypeKind::Tensor,
            tensor: Some(TensorMetadata {
                dtype: Some(dtype),
                shape: Some(shape.to_vec()),
            }),
        }
    }

    fn synthetic_snapshot() -> OrtGraphSnapshot {
        let mut ir = Graph::new();
        ir.opset_imports.insert(String::new(), 13);
        let batch = ir.intern_symbol("batch");
        let input = ir.create_named_value(
            "input",
            DataType::Float32,
            vec![Dim::Symbolic(batch), Dim::Static(4)],
        );
        let weight = ir.create_named_value("weight", DataType::Float32, vec![Dim::Static(4)]);
        let output = ir.create_named_value(
            "output",
            DataType::Float32,
            vec![Dim::Symbolic(batch), Dim::Static(4)],
        );
        ir.add_input(input);
        ir.add_output(output);
        let node = ir.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(input), None, Some(weight)],
            vec![output],
        ));
        let mut snapshot = OrtGraphSnapshot {
            ir,
            onnx_ir_version: 9,
            nodes: HashMap::new(),
            values: HashMap::new(),
            node_by_ort_id: HashMap::from([(42, node)]),
            value_by_name: HashMap::from([
                ("input".to_string(), input),
                ("weight".to_string(), weight),
                ("output".to_string(), output),
            ]),
            graph_input_slots: vec![Some(input)],
            graph_output_slots: vec![Some(output)],
            initializers: HashMap::from([(
                weight,
                InitializerMetadata {
                    value: weight,
                    dtype: Some(DataType::Float32),
                    shape: Some(vec![DimensionMetadata::Static(4)]),
                },
            )]),
            node_schema_versions: HashMap::from([(String::new(), BTreeSet::from([13]))]),
            subgraphs: HashMap::new(),
        };
        snapshot.values.insert(
            input,
            ValueMetadata {
                name: "input".to_string(),
                ..tensor(
                    DataType::Float32,
                    &[
                        DimensionMetadata::Symbolic("batch".to_string()),
                        DimensionMetadata::Static(4),
                    ],
                )
            },
        );
        snapshot.values.insert(
            weight,
            ValueMetadata {
                name: "weight".to_string(),
                ..tensor(DataType::Float32, &[DimensionMetadata::Static(4)])
            },
        );
        snapshot.values.insert(
            output,
            ValueMetadata {
                name: "output".to_string(),
                ..tensor(
                    DataType::Float32,
                    &[
                        DimensionMetadata::Symbolic("batch".to_string()),
                        DimensionMetadata::Static(4),
                    ],
                )
            },
        );
        snapshot.nodes.insert(
            node,
            NodeMetadata {
                ort_node_id: 42,
                name: "add".to_string(),
                domain: String::new(),
                op_type: "Add".to_string(),
                since_version: 13,
                input_slots: vec![Some(input), None, Some(weight)],
                output_slots: vec![Some(output)],
            },
        );
        snapshot
    }

    #[test]
    fn preserves_topology_optional_slots_and_boundaries() {
        let snapshot = synthetic_snapshot();
        let node = snapshot.node_by_ort_id[&42];
        assert_eq!(snapshot.ir.topological_order().unwrap(), vec![node]);
        assert_eq!(snapshot.nodes[&node].input_slots[1], None);
        assert_eq!(
            snapshot.ir.inputs,
            snapshot
                .graph_input_slots
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            snapshot.ir.outputs,
            snapshot
                .graph_output_slots
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            snapshot.ir.uses(snapshot.value_by_name["input"]),
            vec![(node, 0)]
        );
        assert!(snapshot.ir.initializers.is_empty());
        snapshot.ir.validate().unwrap();
    }

    #[test]
    fn preserves_symbolic_shapes_domains_versions_and_initializers() {
        let snapshot = synthetic_snapshot();
        let input = snapshot.value_by_name["input"];
        assert_eq!(
            snapshot.values[&input].tensor.as_ref().unwrap().shape,
            Some(vec![
                DimensionMetadata::Symbolic("batch".to_string()),
                DimensionMetadata::Static(4),
            ])
        );
        assert_eq!(
            snapshot.ir.symbol_constraints.values().next(),
            Some(&SymbolConstraints::new(
                SymbolId(0),
                Some("batch".to_string())
            ))
        );
        assert_eq!(snapshot.node_schema_versions[""], BTreeSet::from([13]));
        assert_eq!(snapshot.onnx_ir_version, 9);
        assert_eq!(snapshot.ir.opset_imports[""], 13);
        let weight = snapshot.value_by_name["weight"];
        assert_eq!(
            snapshot.initializers[&weight].shape,
            Some(vec![DimensionMetadata::Static(4)])
        );
    }

    #[test]
    fn indexes_control_flow_subgraphs_in_both_views() {
        let mut snapshot = synthetic_snapshot();
        let parent = snapshot.node_by_ort_id[&42];
        let child = synthetic_snapshot();
        snapshot
            .ir
            .subgraphs
            .insert((parent, "then_branch".to_string()), child.ir.clone());
        snapshot
            .subgraphs
            .insert((parent, "then_branch".to_string()), Box::new(child));
        assert_eq!(snapshot.ir.subgraphs.len(), 1);
        assert_eq!(
            snapshot.subgraphs[&(parent, "then_branch".to_string())]
                .ir
                .num_nodes(),
            1
        );
    }

    #[test]
    fn preserves_optional_output_slots_without_ir_placeholders() {
        let mut snapshot = synthetic_snapshot();
        let node = snapshot.node_by_ort_id[&42];
        let output = snapshot.value_by_name["output"];
        snapshot.nodes.get_mut(&node).unwrap().output_slots = vec![None, Some(output)];

        assert_eq!(snapshot.nodes[&node].output_slots, vec![None, Some(output)]);
        assert_eq!(snapshot.ir.node(node).outputs, vec![output]);
        snapshot.ir.validate().unwrap();
    }
}
