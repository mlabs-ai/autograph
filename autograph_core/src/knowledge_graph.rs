use std::cmp::min;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt::Display;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::mem;
use std::ops::{Range, RangeInclusive};
use std::path::Path;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index::sample;

enum ClusterStrength {
    Strong,
    Border,
    Weak
}

/// `KnowledgeGraph` stores a sparse graph in an edge list format. For now, the
/// graph will be undirected and uncolored -- these features will be added later.
/// 
/// Vertices can be represented by anything hashable (e.g., a `String`), but in
/// the internal edge representations, they will be stored as `usize` integer
/// IDs that are mapped by `vertex_mapping`. Edges will be stored in a `Vec`, 
/// where each item is a tuple `(usize, usize)`, where the first item is the ID
/// of the vertex at one end of the edge, and the second item is the ID of the
/// other vertex.
#[derive(Eq, Debug)]
pub struct KnowledgeGraph<V: Ord> {
    edges: Vec<(usize, usize)>,
    vertex_mapping: BTreeMap<V, usize>,
    clusters: Vec<RangeInclusive<usize>>
}

impl<V: Ord> PartialEq for KnowledgeGraph<V> {
    /// Implements `eq` in such a way that the order of edges is not important.
    fn eq(&self, other: &Self) -> bool {
        let self_edge_set: HashSet<_> = self.edges.iter().collect();
        let other_edge_set: HashSet<_> = other.edges.iter().collect();
        self_edge_set == other_edge_set && self.vertex_mapping == other.vertex_mapping
    }
}

impl<V: Ord> KnowledgeGraph<V> {
    /// Create a new, empty `KnowledgeGraph`.
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            vertex_mapping: BTreeMap::new(),
            clusters: Vec::new()
        }
    }

    /// Returns the number of vertices in this graph.
    pub fn num_vertices(&self) -> usize {
        self.vertex_mapping.len()
    }

    /// Adds the given vertex to the `vertex_mapping` if not already present.
    /// If it is already present, no vertex will be added. Either way, the 
    /// `usize` integer ID used to represent the vertex in edges will be
    /// returned.
    pub fn add_vertex(&mut self, vertex: V) -> usize {
        let num_vertices = self.vertex_mapping.len();
        *self.vertex_mapping
            .entry(vertex)
            .or_insert(num_vertices)
    }

    /// Gets the `usize` integer ID used to represent the given vertex in edges,
    /// if the vertex has been added to the graph.
    pub fn get_vertex_id(&self, vertex: &V) -> Option<usize> {
        self.vertex_mapping.get(vertex).copied()
    }

    /// Adds an edge to the graph. If any of the vertices on either side of the
    /// edge is not already in the graph, it will be added.
    pub fn add_edge(&mut self, v1: V, v2: V) {
        let mut index1 = self.add_vertex(v1);
        let mut index2 = self.add_vertex(v2);

        // TODO: For determinism purposes, it helps to order these indices.
        // This only works since the graph is unirected; once we add direction,
        // this won't work.
        if index2 < index1 {
            mem::swap(&mut index1, &mut index2);
        }

        self.edges.push((index1, index2));
    }

    /// Remaps the internal vertex IDs. The input slice `new_mapping` will have
    /// one entry per vertex. Each entry's index will correspond to the vertex's
    /// old ID, while the entry itself will correspond to the new ID.
    pub fn remap_vertices(&mut self, new_mapping: &HashMap<usize, usize>) {
        for old_id in self.vertex_mapping.values_mut() {
            if let Some(&new_id) = new_mapping.get(old_id) {
                *old_id = new_id;
            }
        }

        for (src, dst) in self.edges.iter_mut() {
            if let Some(&new_src) = new_mapping.get(src) {
                *src = new_src;
            }
            if let Some(&new_dst) = new_mapping.get(dst) {
                *dst = new_dst;
            }

            // TODO: For determinism purposes, it helps to order these indices.
            // This only works since the graph is unirected; once we add direction,
            // this won't work.
            if *dst < *src {
                mem::swap(dst, src);
            }
        }
    }

    /// Applies a random ordering to the internal vertex IDs. While this does
    /// not meaningfully alter the actual structure of the graph, it does
    /// effectively ensure that the graph is unclustered for the purposes of
    /// the clustering algorithm of this project.
    pub fn shuffle_vertex_ids(&mut self, seed: u64) {
        let num_verts = self.vertex_mapping.len();

        let mut rng = StdRng::seed_from_u64(seed);
        let new_mapping = sample(&mut rng, num_verts, num_verts)
            .into_iter()
            .enumerate()
            .collect();

        self.remap_vertices(&new_mapping);
    }

    /// Performs one iteration of the block factorization algorithm.
    pub fn cluster_step(&mut self, factor: f64, range: Range<usize>) -> Vec<f64> {
        // Get the weight per vertex
        let mut weights = vec![0.0; range.end - range.start];

        for &(id1, id2) in &self.edges {
            if !range.contains(&id1) || !range.contains(&id2) {
                continue;
            }

            let id1 = id1 - range.start;
            let id2 = id2 - range.start;

            // TODO: Assumes undigraph
            weights[id1] += (-factor * id2 as f64).exp();
            weights[id2] += (-factor * id1 as f64).exp();

            // TODO: notes for if 0 case
            //weights[id1] += (factor * (num_verts - id2) as f64).exp();
        }

        // Order by weight descending. Equal weights are a virtual impossibility,
        // so we can use an unstable sort to save memory
        let mut weights: Vec<_> = weights.into_iter().enumerate().collect();
        weights.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap().reverse());

        // Get new mapping and apply it
        let new_mapping = weights.iter()
            .enumerate()
            .map(|(new, &(old, _))| (old + range.start, new + range.start))
            .collect();
        self.remap_vertices(&new_mapping);

        weights.into_iter().map(|(_, w)| w).collect()
    }

    pub fn cluster(
        &mut self,
        factor: f64,
        steps_before_subdivide: usize,
        mut boundary_threshold: f64,
        min_cluster_size: usize
    ) {
        // Perform sanity checks
        assert!(self.vertex_mapping.len() > 0, "Cannot cluster an empty graph");
        assert!(min_cluster_size > 0, "Min cluster size cannot be 0");

        // Ensure boundary threshold is negative
        if boundary_threshold > 0.0 {
            boundary_threshold = -boundary_threshold;
        }

        // Step 0: Split disconnected nodes from rest of graph
        let mut last_connected = self.vertex_mapping.len() - 1;
        let mut disconnected_nodes: HashSet<_> = (0..self.vertex_mapping.len()).collect();
        for (v1, v2) in &self.edges {
            disconnected_nodes.remove(v1);
            disconnected_nodes.remove(v2);
        }
        if !disconnected_nodes.is_empty() {
            let mut new_mapping = HashMap::new();
            for disconnected_node in disconnected_nodes {
                new_mapping.insert(disconnected_node, last_connected);
                new_mapping.insert(last_connected, disconnected_node);

                self.clusters.push(last_connected..=last_connected);
                last_connected -= 1;
            }

            self.remap_vertices(&new_mapping);
        }

        // If not enough nodes remaining, return
        let num_nodes = last_connected + 1;
        if num_nodes <= min_cluster_size {
            if num_nodes > 0 {
                self.clusters.push(0..=last_connected);
            }
        }

        // Otherwise, continue to cluster
        else {
            self.cluster_worker(
                0..num_nodes,
                factor,
                steps_before_subdivide,
                boundary_threshold,
                min_cluster_size
            );
        }
    }

    fn cluster_worker(
        &mut self,
        range: Range<usize>,
        factor: f64,
        steps_before_subdivide: usize,
        boundary_threshold: f64,
        min_cluster_size: usize
    ) -> Vec<(Range<usize>, ClusterStrength)> {
        assert!(
            steps_before_subdivide >= 1, 
            "Must perform at least one subdivision step"
        );

        // If the number of nodes is low, return that this is a weak cluster
        if range.end - range.start <= min_cluster_size {
            return vec![(range, ClusterStrength::Weak)];
        }

        // Step 1: Do some cluster steps
        for _ in 0..steps_before_subdivide-1 {
            self.cluster_step(factor, range.clone());
        }

        // Step 2: Do subdivision
        let weights = self.cluster_step(factor, range);
        let log_weights: Vec<_> = weights
            .iter()
            .take_while(|&&w| w > 0.0)
            .map(|w| w.log2())
            .collect();
        let log_derivatives: Vec<_> = log_weights
            .iter()
            .zip(log_weights.iter().skip(1))
            .map(|(a, b)| b - a)
            .collect();

        let mut clusters: Vec<Range<usize>> = Vec::new();
        let mut curr_cluster = 0..min_cluster_size;
        let mut curr_cluster_strength = 
            if log_derivatives[min_cluster_size - 1] > boundary_threshold {
                ClusterStrength::Strong
            }
            else {
                ClusterStrength::Border
            };
        for d in log_derivatives.iter().skip(min_cluster_size - 1) {

        }

        vec![]
    }

    pub fn split_density(&self) -> Vec<f64> {
        // TODO: `split_density` assumes undigraph

        // Helper function to reduce code reuse
        fn get_density_and_move_to_upper_left(
            index: &mut usize,
            upper_right: &mut Vec<usize>,
            lower_left: &mut Vec<usize>,
            num_verts: usize,
            upper_left_count: &mut usize,
            lower_right_count: &mut usize,
            point_densities: &mut [f64]
        ) {
            // Calculate density value
            let upper_right_count: usize = upper_right[*index + 1..].iter().sum();
            let lower_left_count: usize = lower_left[*index + 1..].iter().sum();
            let in_verts = *index as f64 + 1.0;
            let out_verts = num_verts as f64 - in_verts;
            point_densities[*index] = 
                *upper_left_count as f64 / in_verts.powi(2) +
                *lower_right_count as f64 / out_verts.powi(2) -
                upper_right_count as f64 / (in_verts * out_verts) -
                lower_left_count as f64 / (in_verts * out_verts);

            // Move to next index and move points to upper left
            *index += 1;
            *upper_left_count += upper_right[*index];
            *upper_left_count += lower_left[*index];
        }

        let mut point_densities = vec![0.0; self.vertex_mapping.len()];
        let mut upper_left_count = 0;
        let mut upper_right: Vec<usize> = vec![0; self.vertex_mapping.len()];
        let mut lower_left: Vec<usize> = vec![0; self.vertex_mapping.len()];
        let mut lower_right: Vec<_> = self.edges.iter().collect();
        lower_right.sort_unstable();
        let mut lower_right_count = lower_right
            .iter()
            .map(|(src, dst)| if src == dst { 1 } else { 2 })
            .sum();
        let total_edges = lower_right_count;

        // Iterate through all edges in order from smallest src to greatest
        let mut i = 0;
        for &(src, dst) in lower_right {
            // Calculate density of split point
            while i < src {
                get_density_and_move_to_upper_left(
                    &mut i, 
                    &mut upper_right, 
                    &mut lower_left, 
                    self.vertex_mapping.len(), 
                    &mut upper_left_count, 
                    &mut lower_right_count, 
                    &mut point_densities
                );
            }

            // Move edge to upper left or "wings"
            if src == dst {
                lower_right_count -= 1;
                upper_left_count += 1;
            }
            else {
                lower_right_count -= 2;
                upper_right[dst] += 1;
                lower_left[dst] += 1;
            }
        }
        
        // Calculate remaining densities, if there are any
        while i < self.vertex_mapping.len() - 1 {
            get_density_and_move_to_upper_left(
                &mut i,
                &mut upper_right,
                &mut lower_left,
                self.vertex_mapping.len(),
                &mut upper_left_count,
                &mut lower_right_count,
                &mut point_densities
            );
        }

        // We can calculate last density at this point
        point_densities[self.vertex_mapping.len() - 1] = 
            total_edges as f64 / (self.vertex_mapping.len() as f64).powi(2);

        point_densities
    }

    /// Write the graph as a Graphviz dot file to the given path.
    pub fn write_to_dot_file<P>(&self, path: P) -> Result<(), Box<dyn Error>> 
    where 
        P: AsRef<Path>,
        V: Display
    {
        // Ensure vertices are written in order of IDs
        let id_to_vertex_mapping: BTreeMap<_, _> = self.vertex_mapping.iter()
            .map(|(v, &id)| (id, v))
            .collect();

        // Open file and write file header
        if let Some(parent_dir) = path.as_ref().parent() {
            create_dir_all(parent_dir)?;
        }
        let mut file = File::create(path)?;
        file.write_all(b"graph {\n")?;

        // Write vertex labels, ensuring they are sorted by ID
        for (v, id) in &self.vertex_mapping {
            file.write_fmt(format_args!("\t{} [label=\"{} ({})\"]\n", v, v, id))?;
        }
        file.write_all(b"\n")?;

        // Write graph edges
        for (id1, id2) in &self.edges {
            let v1 = id_to_vertex_mapping
                .get(id1)
                .ok_or("Graph contained an edge with an unindexed vertex")?;
            let v2 = id_to_vertex_mapping
                .get(id2)
                .ok_or("Graph contained an edge with an unindexed vertex")?;
            file.write_fmt(format_args!("\t{} -- {}\n", v1, v2))?;
        }

        // Write file footer
        file.write_all(b"}")?;

        Ok(())
    }

    /// Returns the graph as an adjacency matrix, in row major format.
    pub fn as_matrix(&self) -> Vec<Vec<usize>> {
        let num_verts = self.vertex_mapping.len();
        let num_bins = min(1000, num_verts);
        let mut adj_mat = vec![vec![0; num_bins]; num_bins];

        // TODO: Assumes undirected graph
        for &(v1, v2) in &self.edges {
            let v1 = v1 * num_bins / num_verts;
            let v2 = v2 * num_bins / num_verts;
            adj_mat[v1][v2] += 1;
            adj_mat[v2][v1] += 1;
        }

        adj_mat
    }
}

impl<V: Ord> Default for KnowledgeGraph<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeGraph<String> {
    /// Read the graph from the given dot file.
    pub fn from_dot_file<P>(path: P) -> Result<Self, Box<dyn Error>> 
    where 
        P: AsRef<Path>
    {
        // Read and parse graph
        let ast_graph = dot_parser::ast::Graph::from_file(path)?;
        let can_graph = dot_parser::canonical::Graph::from(ast_graph);

        // Sort the vertices for determinism's sake
        let mut vs: Vec<_> = can_graph.nodes.set.keys().collect();
        vs.sort_unstable();

        // Convert the graph to this format
        let mut graph = Self::new();
        for v in vs {
            graph.add_vertex(v.to_string());
        }
        for e in can_graph.edges.set {
            graph.add_edge(e.from, e.to);
        }

        Ok(graph)
    }
}

impl<V: Ord> FromIterator<(V, V)> for KnowledgeGraph<V> {
    /// Constructs a `KnowledgeGraph<V>` from an iterator of tuples representing
    /// edges in the form `(V, V)`, where the first `V` represents the source 
    /// vertex of the edge, and the second represents the destination.
    fn from_iter<T: IntoIterator<Item = (V, V)>>(iter: T) -> Self {
        let mut graph = Self::new();
        for (v1, v2) in iter {
            graph.add_edge(v1, v2);
        }

        graph
    }
}

impl<V: Ord> From<&KnowledgeGraph<V>> for KnowledgeGraph<usize> {
    fn from(value: &KnowledgeGraph<V>) -> Self {
        Self {
            edges: value.edges.clone(),
            vertex_mapping: value.vertex_mapping.values().cloned().enumerate().collect(),
            clusters: Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::KnowledgeGraph;
    use tempfile::tempdir;

    #[test]
    fn empty_graph() {
        let g: KnowledgeGraph<i32> = KnowledgeGraph::new();
        assert!(g.edges.is_empty());
        assert!(g.vertex_mapping.is_empty());
    }

    #[test]
    fn one_vertex() {
        let mut g = KnowledgeGraph::new();
        let idx = g.add_vertex("v1");

        assert!(g.edges.is_empty());
        assert_eq!(g.vertex_mapping.len(), 1);
        assert_eq!(idx, 0);
    }

    #[test]
    fn three_vertices() {
        let mut g = KnowledgeGraph::new();
        let idx1 = g.add_vertex("v1");
        let idx2 = g.add_vertex("v2");
        let idx3 = g.add_vertex("v3");

        assert!(g.edges.is_empty());
        assert_eq!(g.vertex_mapping.len(), 3);
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(idx3, 2);
    }

    #[test]
    fn one_vertex_cycle() {
        let g: KnowledgeGraph<_> = [("v1", "v1")].into_iter().collect();

        assert_eq!(g.edges.len(), 1);
        assert_eq!(g.vertex_mapping.len(), 1);
        assert_eq!(g.edges[0], (0, 0));
    }

    #[test]
    fn three_vertices_tree() {
        let g: KnowledgeGraph<_> = [
            ("v1", "v2"),
            ("v1", "v3")
        ].into_iter().collect();

        assert_eq!(g.edges.len(), 2);
        assert_eq!(g.vertex_mapping.len(), 3);
        assert_eq!(g.edges[0], (0, 1));
        assert_eq!(g.edges[1], (0, 2));
    }

    #[test]
    fn rename_vertices() {
        let mut g: KnowledgeGraph<_> = [
            ("v1", "v2"),
            ("v1", "v3")
        ].into_iter().collect();

        assert_eq!(g.edges.len(), 2);
        assert_eq!(g.vertex_mapping.len(), 3);
        assert_eq!(g.edges[0], (0, 1));
        assert_eq!(g.edges[1], (0, 2));
        
        let remapping = [
            (0, 1),
            (1, 2),
            (2, 0)
        ].into_iter().collect();
        g.remap_vertices(&remapping);

        assert_eq!(g.get_vertex_id(&"v1"), Some(1));
        assert_eq!(g.get_vertex_id(&"v2"), Some(2));
        assert_eq!(g.get_vertex_id(&"v3"), Some(0));
        assert_eq!(g.edges.len(), 2);
        assert_eq!(g.vertex_mapping.len(), 3);
        assert_eq!(g.edges[0], (1, 2));
        assert_eq!(g.edges[1], (0, 1));
    }

    #[test]
    fn dot_file_writing() {
        let g: KnowledgeGraph<_> = [
            ("v1", "v2"),
            ("v1", "v3")
        ].into_iter().collect();

        // Write graph to temporary dot file
        let temp_dir = tempdir().unwrap();
        let temp_path = temp_dir.path().join("test.dot");
        g.write_to_dot_file(&temp_path).unwrap();

        // Read the temporary dot file and make sure it is correct
        let temp_contents = fs::read_to_string(temp_path).unwrap();
        let corr_contents = fs::read_to_string("tests/goldens/test.dot").unwrap();

        assert_eq!(temp_contents, corr_contents);
    }

    #[test]
    fn dot_file_reading() {
        let g1: KnowledgeGraph<String> = [
            ("v1".to_string(), "v2".to_string()),
            ("v1".to_string(), "v3".to_string())
        ].into_iter().collect();
        let g2 = KnowledgeGraph::from_dot_file("tests/goldens/test.dot").unwrap();

        assert_eq!(g1, g2);
    }

    #[test]
    fn adj_mat_small() {
        let g: KnowledgeGraph<_> = [
            ("v1", "v2"),
            ("v1", "v3")
        ].into_iter().collect();

        let adj_mat = vec![
            vec![0, 1, 1],
            vec![1, 0, 0],
            vec![1, 0, 0]
        ];
        assert_eq!(adj_mat, g.as_matrix());
    }

    #[test]
    fn adj_mat_large() {
        let mut g: KnowledgeGraph<_> = [
            (0, 1),
            (0, 2)
        ].into_iter().collect();
        for i in 3..2000 {
            g.add_vertex(i);
        }

        let mut adj_mat = vec![vec![0;1000];1000];
        adj_mat[0][0] = 2;
        adj_mat[0][1] = 1;
        adj_mat[1][0] = 1;

        assert_eq!(adj_mat, g.as_matrix());
    }

    #[test]
    fn empty_density() {
        let mut g = KnowledgeGraph::new();

        g.add_vertex(0);
        g.add_vertex(1);

        assert_eq!(g.split_density(), vec![0.0; 2]);
    }

    fn feq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn one_edge_density() {
        let mut g = KnowledgeGraph::new();

        g.add_vertex(0);
        g.add_vertex(1);
        g.add_vertex(2);
        g.add_edge(0, 0);

        let density = g.split_density();

        assert_eq!(density[0], 1.0);
        assert_eq!(density[1], 0.25);
        assert!(feq(density[2], 1.0 / 9.0, 1e-9));
    }

    #[test]
    fn two_edge_density() {
        let mut g = KnowledgeGraph::new();

        g.add_vertex(0);
        g.add_vertex(1);
        g.add_vertex(2);
        g.add_edge(0, 0);
        g.add_edge(0, 1);

        let density = g.split_density();

        assert_eq!(density[0], 0.0);
        assert_eq!(density[1], 0.75);
        assert!(feq(density[2], 1.0 / 3.0, 1e-9));
    }

    #[test]
    fn three_edge_density() {
        let mut g = KnowledgeGraph::new();

        g.add_vertex(0);
        g.add_vertex(1);
        g.add_vertex(2);
        g.add_edge(0, 0);
        g.add_edge(0, 1);
        g.add_edge(1, 2);

        let density = g.split_density();

        assert_eq!(density[0], 0.5);
        assert_eq!(density[1], -0.25);
        assert!(feq(density[2], 5.0 / 9.0, 1e-9));
    }

    #[test]
    fn four_edge_density() {
        let mut g = KnowledgeGraph::new();

        g.add_vertex(0);
        g.add_vertex(1);
        g.add_vertex(2);
        g.add_edge(0, 0);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 2);

        let density = g.split_density();

        assert_eq!(density[0], 0.75);
        assert_eq!(density[1], 0.75);
        assert!(feq(density[2], 2.0 / 3.0, 1e-9));
    }
}