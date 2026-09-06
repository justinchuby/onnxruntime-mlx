//! GetCapability policy and partition orchestration.
//!
//! This module owns the short-lived ORT graph inspection required to apply registry
//! policy. Topology always comes from [`OrtGraphSnapshot`]; raw `OrtNode` handles are
//! used only while this callback is active for registry attribute predicates.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, c_char};
use std::ptr;

use crate::engine::is_separate_qkv_attention_op;
use crate::factory::ORT_API_VERSION;
use crate::ort_graph::{NodeMetadata, OrtGraphSnapshot};
use crate::partition::{
    build_contiguous_clusters, build_convex_clusters, infer_layer_boundary_values,
    split_annotated_layer_clusters,
};
use crate::registry::{CompilePartitionClass, NodeView, claimable};
use crate::sys::ort;

/// Apply EP claim policy to one optimized graph.
///
/// # Safety
/// All ABI handles are borrowed from ORT and must remain valid for this call. No
/// borrowed pointer is retained after this function returns.
pub(crate) unsafe fn get_capability(
    ort_api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> *mut ort::OrtStatus {
    unsafe {
        let api = &*ort_api;
        let ep_api = &*ep_api;
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

        // ORT partitions control-flow bodies before their parent. Never create a
        // private-domain fusion inside a nested body; the parent is claimed whole or
        // ORT retains the body on CPU.
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

        // Foundry Q8 Whisper keeps its dynamic position-ID Range/Tile pair together
        // with the decoder so ORT does not introduce CPU islands.
        let native_q8_attention_graph = node_order.len() > 32
            && node_order.iter().any(|&node| {
                let metadata = &snapshot.nodes[&node];
                is_separate_qkv_attention_op(&metadata.domain, &metadata.op_type)
            })
            && node_order.iter().any(|&node| {
                let view = NodeView::new(ort_api, raw_nodes_by_id[&node]);
                view.op_type() == "MatMulNBits" && view.int_attr("bits", 4) == 8
            })
            && node_order
                .iter()
                .filter(|&&node| {
                    let view = NodeView::new(ort_api, raw_nodes_by_id[&node]);
                    view.op_type() == "Range"
                        && view.read_const_scalar_f64(2) == Some(1.0)
                        && (0..3).any(|index| view.read_const_scalar_f64(index).is_none())
                })
                .count()
                == 1
            && node_order
                .iter()
                .filter(|&&node| {
                    let view = NodeView::new(ort_api, raw_nodes_by_id[&node]);
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

        let supported: Vec<bool> = node_order
            .iter()
            .map(|&node_id| {
                if in_cf_body {
                    return false;
                }
                let view = NodeView::new(ort_api, raw_nodes_by_id[&node_id]);
                if native_q8_attention_graph && matches!(view.op_type().as_str(), "Range" | "Tile")
                {
                    return true;
                }
                claimable(&view)
            })
            .collect();
        let float64: Vec<bool> = node_order
            .iter()
            .map(|&node| snapshot_node_uses_float64(&snapshot, node))
            .collect();
        let attention_anchors = node_order
            .iter()
            .copied()
            .filter(|node| is_decoder_attention_anchor(&snapshot.nodes[node]))
            .collect::<Vec<_>>();
        let decoder_graph = !attention_anchors.is_empty();
        let compile_class: Vec<CompilePartitionClass> = node_order
            .iter()
            .map(|&node| {
                let view = NodeView::new(ort_api, raw_nodes_by_id[&node]);
                if decoder_graph {
                    crate::registry::compile_shape_safety(&view).partition_class()
                } else {
                    CompilePartitionClass::Shapeless
                }
            })
            .collect();

        let tr = crate::trace::tracer();
        let mut rejected: Vec<(String, usize, String, Vec<String>)> = Vec::new();
        if tr.active() || std::env::var_os("ONNXRUNTIME_EP_MLX_CLAIM_DEBUG").is_some() {
            use std::collections::BTreeMap;
            let mut acc: BTreeMap<String, (usize, String, Vec<String>)> = BTreeMap::new();
            for (&node, &ok) in node_order.iter().zip(supported.iter()) {
                if !ok {
                    let view = NodeView::new(ort_api, raw_nodes_by_id[&node]);
                    let entry = acc
                        .entry(crate::registry::qualified_op_name(&view))
                        .or_insert((0, String::new(), Vec::new()));
                    entry.0 += 1;
                    if entry.1.is_empty() {
                        entry.1 = if in_cf_body {
                            "inside a control-flow subgraph body — claimed as part of the parent \
                             If/Loop/Scan, not individually"
                                .to_string()
                        } else {
                            crate::registry::claim_decision(&view)
                                .err()
                                .map(|cause| cause.into_owned())
                                .unwrap_or_else(|| "declined (no reason reported)".to_string())
                        };
                    }
                    if entry.2.len() < 16 {
                        let name = view.name();
                        if !name.is_empty() {
                            entry.2.push(name);
                        }
                    }
                }
            }
            rejected = acc
                .into_iter()
                .map(|(op, (count, reason, names))| (op, count, reason, names))
                .collect();
            rejected.sort_by_key(|entry| std::cmp::Reverse(entry.1));
            if std::env::var_os("ONNXRUNTIME_EP_MLX_CLAIM_DEBUG").is_some() {
                for (op, count, reason, names) in &rejected {
                    log::debug!("unclaimed {op} x{count} ({reason}): {names:?}");
                }
            }
        }

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
        let layer_partition_span = select_layer_partition_span(
            std::env::var("ONNXRUNTIME_EP_MLX_LAYER_PARTITIONS")
                .ok()
                .as_deref(),
            layer_boundary_outputs.len(),
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
            let mut options: ort::OrtNodeFusionOptions = std::mem::zeroed();
            options.ort_version_supported = ORT_API_VERSION;
            options.drop_constant_initializers = true;
            let st = add_fuse(support, group.as_ptr(), group.len(), &options);
            if !st.is_null() {
                return st;
            }
            claimed += cluster.len();
        }
        tr.record_claim(claimed, node_order.len(), clusters.len(), &rejected);
        ptr::null_mut()
    }
}

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

unsafe fn release_status(api: &ort::OrtApi, status: *mut ort::OrtStatus) {
    if !status.is_null() {
        unsafe { (api.ReleaseStatus.unwrap())(status) };
    }
}

unsafe fn graph_metadata_value(
    api: &ort::OrtApi,
    graph: *const ort::OrtGraph,
    key: &CStr,
) -> Result<Option<String>, *mut ort::OrtStatus> {
    unsafe {
        let mut metadata: *mut ort::OrtModelMetadata = ptr::null_mut();
        let status = (api.Graph_GetModelMetadata.unwrap())(graph, &mut metadata);
        if !status.is_null() {
            return Err(status);
        }
        let mut allocator: *mut ort::OrtAllocator = ptr::null_mut();
        let status = (api.GetAllocatorWithDefaultOptions.unwrap())(&mut allocator);
        if !status.is_null() {
            (api.ReleaseModelMetadata.unwrap())(metadata);
            return Err(status);
        }
        let mut value: *mut c_char = ptr::null_mut();
        let status = (api.ModelMetadataLookupCustomMetadataMap.unwrap())(
            metadata,
            allocator,
            key.as_ptr(),
            &mut value,
        );
        if !status.is_null() {
            (api.ReleaseModelMetadata.unwrap())(metadata);
            return Err(status);
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

#[cfg(test)]
mod tests {
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
