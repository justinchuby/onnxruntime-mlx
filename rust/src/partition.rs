//! Pure capability-partition algorithms over the owned graph IR.

use std::collections::{HashMap, HashSet};

use onnx_runtime_ir::{Graph, NodeId, ValueId};

use crate::registry::CompilePartitionClass;

pub(crate) fn build_contiguous_clusters(
    order: &[NodeId],
    supported: &[bool],
    float64: &[bool],
    compile_class: &[CompilePartitionClass],
) -> Vec<Vec<NodeId>> {
    debug_assert_eq!(order.len(), supported.len());
    let mut clusters = Vec::new();
    let mut start = 0usize;
    while start < supported.len() {
        while start < supported.len() && !supported[start] {
            start += 1;
        }
        if start == supported.len() {
            break;
        }
        let mut end = start + 1;
        while end < supported.len()
            && supported[end]
            && float64[end] == float64[start]
            && compile_class[end] == compile_class[start]
        {
            end += 1;
        }
        clusters.push(order[start..end].to_vec());
        start = end;
    }
    clusters
}

/// Groups supported nodes into maximal, convex, connected clusters. A set S is convex iff no node
/// outside S lies on a path between two members of S.
pub(crate) fn build_convex_clusters(
    graph: &Graph,
    order: &[NodeId],
    supported: &[bool],
    float64: &[bool],
    compile_class: &[CompilePartitionClass],
) -> Vec<Vec<NodeId>> {
    let positions: HashMap<NodeId, usize> = order
        .iter()
        .enumerate()
        .map(|(index, &node)| (node, index))
        .collect();
    let succ = order
        .iter()
        .map(|&node| {
            graph
                .successors(node)
                .into_iter()
                .filter_map(|successor| positions.get(&successor).copied())
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    let pred = order
        .iter()
        .map(|&node| {
            graph
                .predecessors(node)
                .into_iter()
                .filter_map(|predecessor| positions.get(&predecessor).copied())
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    cluster_edges(supported, float64, compile_class, &succ, &pred)
        .into_iter()
        .map(|cluster| cluster.into_iter().map(|index| order[index]).collect())
        .collect()
}

pub(crate) fn infer_layer_boundary_values(
    graph: &Graph,
    order: &[NodeId],
    attention_anchors: &[NodeId],
) -> HashSet<ValueId> {
    let positions: HashMap<NodeId, usize> = order
        .iter()
        .enumerate()
        .map(|(index, &node)| (node, index))
        .collect();
    let anchors = attention_anchors
        .iter()
        .filter_map(|node| positions.get(node).copied())
        .collect::<Vec<_>>();
    let op_types = order
        .iter()
        .map(|&node| graph.node(node).op_type.clone())
        .collect::<Vec<_>>();
    let successors = order
        .iter()
        .map(|&node| {
            graph
                .successors(node)
                .into_iter()
                .filter_map(|successor| positions.get(&successor).copied())
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    let predecessors = order
        .iter()
        .map(|&node| {
            graph
                .predecessors(node)
                .into_iter()
                .filter_map(|predecessor| positions.get(&predecessor).copied())
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    infer_boundary_indices(&op_types, &successors, &predecessors, &anchors)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|index| graph.node(order[index]).outputs.first().copied())
        .collect()
}

pub(crate) fn split_annotated_layer_clusters(
    graph: &Graph,
    clusters: Vec<Vec<NodeId>>,
    layer_boundary_outputs: &HashSet<ValueId>,
    layers_per_partition: usize,
) -> Vec<Vec<NodeId>> {
    let mut split = Vec::with_capacity(clusters.len());
    for cluster in clusters {
        let mut part = Vec::new();
        let mut layers_in_part = 0usize;
        for node in cluster {
            part.push(node);
            let layer_end = graph
                .node(node)
                .outputs
                .iter()
                .any(|output| layer_boundary_outputs.contains(output));
            if layer_end {
                layers_in_part += 1;
                if layers_in_part == layers_per_partition {
                    split.push(std::mem::take(&mut part));
                    layers_in_part = 0;
                }
            }
        }
        if !part.is_empty() {
            split.push(part);
        }
    }
    split
}

fn infer_boundary_indices(
    op_types: &[String],
    successors: &[Vec<usize>],
    predecessors: &[Vec<usize>],
    attention_anchors: &[usize],
) -> Option<Vec<usize>> {
    if attention_anchors.len() < 2 {
        return None;
    }
    let mut boundaries = Vec::with_capacity(attention_anchors.len() - 1);
    for pair in attention_anchors.windows(2) {
        let (current, next) = (pair[0], pair[1]);
        let mut downstream = vec![false; op_types.len()];
        let mut stack = vec![current];
        while let Some(index) = stack.pop() {
            for &successor in &successors[index] {
                if successor <= next && !downstream[successor] {
                    downstream[successor] = true;
                    stack.push(successor);
                }
            }
        }
        let mut upstream = vec![false; op_types.len()];
        let mut stack = vec![next];
        while let Some(index) = stack.pop() {
            for &predecessor in &predecessors[index] {
                if predecessor >= current && !upstream[predecessor] {
                    upstream[predecessor] = true;
                    stack.push(predecessor);
                }
            }
        }
        boundaries.push(
            (current + 1..next)
                .rev()
                .find(|&index| op_types[index] == "Add" && downstream[index] && upstream[index])?,
        );
    }
    Some(boundaries)
}

fn cluster_edges(
    supported: &[bool],
    float64: &[bool],
    compile_class: &[CompilePartitionClass],
    succ: &[Vec<usize>],
    pred: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let n = supported.len();
    debug_assert_eq!(n, float64.len());
    debug_assert_eq!(n, compile_class.len());
    debug_assert_eq!(n, succ.len());
    debug_assert_eq!(n, pred.len());
    let words = n.div_ceil(64);
    let mut indeg: Vec<usize> = pred.iter().map(Vec::len).collect();
    let mut stack: Vec<usize> = (0..n).filter(|&index| indeg[index] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(node) = stack.pop() {
        order.push(node);
        for &successor in &succ[node] {
            indeg[successor] -= 1;
            if indeg[successor] == 0 {
                stack.push(successor);
            }
        }
    }
    if order.len() != n {
        order = (0..n).collect();
    }

    let mut reach = vec![vec![0u64; words]; n];
    for &node in order.iter().rev() {
        for &successor in &succ[node] {
            bit_set(&mut reach[node], successor);
            let successor_reach = reach[successor].clone();
            bit_or_into(&mut reach[node], &successor_reach);
        }
    }

    let mut parent: Vec<usize> = (0..n).collect();
    let mut cluster_bits = vec![vec![0u64; words]; n];
    let mut reach_bits = vec![vec![0u64; words]; n];
    for index in 0..n {
        if supported[index] {
            bit_set(&mut cluster_bits[index], index);
            reach_bits[index] = reach[index].clone();
        }
    }

    let mut edges = Vec::new();
    for node in 0..n {
        if !supported[node] {
            continue;
        }
        for &successor in &succ[node] {
            if supported[successor]
                && float64[node] == float64[successor]
                && compile_class[node] == compile_class[successor]
            {
                edges.push((node, successor));
            }
        }
    }
    let is_convex = |cluster: &[u64], cluster_reach: &[u64]| {
        reach.iter().enumerate().all(|(node, reaches)| {
            bit_test(cluster, node)
                || !bit_test(cluster_reach, node)
                || !bit_intersects(reaches, cluster)
        })
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &(left, right) in &edges {
            let left_root = uf_find(&mut parent, left);
            let right_root = uf_find(&mut parent, right);
            if left_root == right_root {
                continue;
            }
            let mut merged = cluster_bits[left_root].clone();
            bit_or_into(&mut merged, &cluster_bits[right_root]);
            let mut merged_reach = reach_bits[left_root].clone();
            bit_or_into(&mut merged_reach, &reach_bits[right_root]);
            if !is_convex(&merged, &merged_reach) {
                continue;
            }
            parent[right_root] = left_root;
            cluster_bits[left_root] = merged;
            reach_bits[left_root] = merged_reach;
            changed = true;
        }
    }

    let mut grouped = HashMap::<usize, Vec<usize>>::new();
    for (node, &is_supported) in supported.iter().enumerate() {
        if is_supported {
            grouped
                .entry(uf_find(&mut parent, node))
                .or_default()
                .push(node);
        }
    }
    let mut clusters = grouped
        .into_values()
        .map(|mut cluster| {
            cluster.sort_unstable();
            cluster
        })
        .collect::<Vec<_>>();

    loop {
        let mut qid = vec![usize::MAX; n];
        for (cluster_id, cluster) in clusters.iter().enumerate() {
            for &node in cluster {
                qid[node] = cluster_id;
            }
        }
        let mut next = clusters.len();
        for id in &mut qid {
            if *id == usize::MAX {
                *id = next;
                next += 1;
            }
        }
        let mut qsucc = vec![HashSet::new(); next];
        let mut qindeg = vec![0usize; next];
        for node in 0..n {
            for &successor in &succ[node] {
                let (from, to) = (qid[node], qid[successor]);
                if from != to && qsucc[from].insert(to) {
                    qindeg[to] += 1;
                }
            }
        }
        let mut stack: Vec<usize> = (0..next).filter(|&node| qindeg[node] == 0).collect();
        let mut visited = 0usize;
        while let Some(node) = stack.pop() {
            visited += 1;
            for &successor in &qsucc[node] {
                qindeg[successor] -= 1;
                if qindeg[successor] == 0 {
                    stack.push(successor);
                }
            }
        }
        if visited == next {
            break;
        }
        let victim = (0..next)
            .filter(|&node| qindeg[node] > 0 && node < clusters.len())
            .min_by_key(|&cluster| clusters[cluster].len());
        match victim {
            Some(cluster) => {
                clusters.remove(cluster);
            }
            None => break,
        }
    }
    clusters.sort_by_key(|cluster| cluster[0]);
    clusters
}

fn uf_find(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

fn bit_set(bits: &mut [u64], index: usize) {
    bits[index >> 6] |= 1u64 << (index & 63);
}

fn bit_test(bits: &[u64], index: usize) -> bool {
    (bits[index >> 6] >> (index & 63)) & 1 != 0
}

fn bit_or_into(destination: &mut [u64], source: &[u64]) {
    for index in 0..destination.len() {
        destination[index] |= source[index];
    }
}

fn bit_intersects(left: &[u64], right: &[u64]) -> bool {
    left.iter()
        .zip(right)
        .any(|(left, right)| left & right != 0)
}

#[cfg(test)]
mod tests {
    use super::{
        build_contiguous_clusters, build_convex_clusters, cluster_edges, infer_boundary_indices,
    };
    use crate::registry::CompilePartitionClass::{Eager, ShapeKeyed, Shapeless};
    use onnx_runtime_ir::{DataType, Graph, Node};

    fn chain(n: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut succ = vec![Vec::new(); n];
        let mut pred = vec![Vec::new(); n];
        for index in 0..n.saturating_sub(1) {
            succ[index].push(index + 1);
            pred[index + 1].push(index);
        }
        (succ, pred)
    }

    #[test]
    fn convex_clustering_preserves_colours_and_contiguity() {
        let n = 8;
        let (succ, pred) = chain(n);
        let supported = vec![true; n];
        let float64 = vec![false, false, true, true, true, true, false, false];
        assert_eq!(
            cluster_edges(&supported, &float64, &vec![Shapeless; n], &succ, &pred),
            vec![vec![0, 1], vec![2, 3, 4, 5], vec![6, 7]]
        );
        let order = (0..n)
            .map(|index| onnx_runtime_ir::NodeId(index as u32))
            .collect::<Vec<_>>();
        assert_eq!(
            build_contiguous_clusters(&order, &supported, &float64, &vec![Shapeless; n]),
            vec![
                vec![onnx_runtime_ir::NodeId(0), onnx_runtime_ir::NodeId(1)],
                vec![
                    onnx_runtime_ir::NodeId(2),
                    onnx_runtime_ir::NodeId(3),
                    onnx_runtime_ir::NodeId(4),
                    onnx_runtime_ir::NodeId(5)
                ],
                vec![onnx_runtime_ir::NodeId(6), onnx_runtime_ir::NodeId(7)]
            ]
        );
    }

    #[test]
    fn ir_topology_matches_legacy_adjacency_clustering() {
        let mut graph = Graph::new();
        let input = graph.create_named_value("input", DataType::Float32, Vec::new());
        let values = (0..5)
            .map(|index| {
                graph.create_named_value(format!("value_{index}"), DataType::Float32, Vec::new())
            })
            .collect::<Vec<_>>();
        let mut previous = input;
        for output in &values {
            graph.insert_node(Node::new(
                onnx_runtime_ir::NodeId(0),
                "Identity",
                vec![Some(previous)],
                vec![*output],
            ));
            previous = *output;
        }
        let order = graph.topological_order().unwrap();
        let (successors, predecessors) = chain(order.len());
        let supported = vec![true; order.len()];
        let float64 = vec![false, true, true, false, false];
        let classes = vec![Shapeless; order.len()];
        let legacy = cluster_edges(&supported, &float64, &classes, &successors, &predecessors);
        let ir = build_convex_clusters(&graph, &order, &supported, &float64, &classes)
            .into_iter()
            .map(|cluster| {
                cluster
                    .into_iter()
                    .map(|node| {
                        order
                            .iter()
                            .position(|&candidate| candidate == node)
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(ir, legacy);
    }

    #[test]
    fn compile_boundaries_remain_separate() {
        let (succ, pred) = chain(5);
        assert_eq!(
            cluster_edges(
                &vec![true; 5],
                &vec![false; 5],
                &[Shapeless, ShapeKeyed, Eager, ShapeKeyed, Shapeless],
                &succ,
                &pred,
            ),
            vec![vec![0], vec![1], vec![2], vec![3], vec![4]]
        );
    }

    #[test]
    fn uniformly_coloured_graphs_remain_maximal() {
        let (succ, pred) = chain(8);
        assert_eq!(
            cluster_edges(
                &vec![true; 8],
                &vec![false; 8],
                &vec![Shapeless; 8],
                &succ,
                &pred,
            ),
            vec![(0..8).collect::<Vec<_>>()]
        );
        let succ = vec![vec![1, 2], vec![3], vec![3], vec![]];
        let pred = vec![vec![], vec![0], vec![0], vec![1, 2]];
        assert_eq!(
            cluster_edges(
                &vec![true; 4],
                &vec![true; 4],
                &vec![Shapeless; 4],
                &succ,
                &pred,
            ),
            vec![vec![0, 1, 2, 3]]
        );
    }

    #[test]
    fn shape_keyed_regions_and_contiguous_fallback_preserve_boundaries() {
        let (succ, pred) = chain(5);
        assert_eq!(
            cluster_edges(
                &vec![true; 5],
                &vec![false; 5],
                &[Shapeless, Shapeless, ShapeKeyed, Shapeless, Shapeless],
                &succ,
                &pred,
            ),
            vec![vec![0, 1], vec![2], vec![3, 4]]
        );
        let order = (0..5)
            .map(|index| onnx_runtime_ir::NodeId(index as u32))
            .collect::<Vec<_>>();
        assert_eq!(
            build_contiguous_clusters(
                &order,
                &vec![true; 5],
                &vec![false; 5],
                &[Shapeless, ShapeKeyed, ShapeKeyed, Shapeless, Shapeless],
            ),
            vec![
                vec![onnx_runtime_ir::NodeId(0)],
                vec![onnx_runtime_ir::NodeId(1), onnx_runtime_ir::NodeId(2)],
                vec![onnx_runtime_ir::NodeId(3), onnx_runtime_ir::NodeId(4)]
            ]
        );
    }

    #[test]
    fn structural_boundaries_use_the_last_residual_add() {
        let layers = 24;
        let mut op_types = Vec::new();
        let mut successors = Vec::new();
        let mut predecessors = Vec::new();
        let mut anchors = Vec::new();
        for _ in 0..layers {
            let anchor = op_types.len();
            anchors.push(anchor);
            op_types.extend(["GroupQueryAttention", "Add", "MatMul", "Add"].map(String::from));
            successors.extend([vec![], vec![], vec![], vec![]]);
            predecessors.extend([vec![], vec![], vec![], vec![]]);
        }
        for &anchor in anchors.iter().take(layers - 1) {
            let next = anchor + 4;
            for (from, to) in [
                (anchor, anchor + 1),
                (anchor + 1, anchor + 2),
                (anchor + 2, anchor + 3),
                (anchor + 3, next),
            ] {
                successors[from].push(to);
                predecessors[to].push(from);
            }
        }
        let boundaries =
            infer_boundary_indices(&op_types, &successors, &predecessors, &anchors).unwrap();
        assert_eq!(boundaries.len(), layers - 1);
        assert_eq!(boundaries[0], 3);
        assert_eq!(boundaries[layers - 2], (layers - 2) * 4 + 3);
    }
}
