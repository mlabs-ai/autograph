use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Display;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// KnowledgeGraph stores a sparse graph in an edge list format. For now, the
/// graph will be undirected and uncolored -- these features will be added later.
/// 
/// Vertices can be represented by anything hashable (e.g., a `String`), but in
/// the internal edge representations, they will be stored as `usize` integer
/// IDs that are mapped by `vertex_mapping`. Edges will be stored in a `Vec`, 
/// where each item is a tuple `(usize, usize)`, where the first item is the ID
/// of the vertex at one end of the edge, and the second item is the ID of the
/// other vertex.
#[derive(PartialEq, Eq, Debug)]
pub struct KnowledgeGraph<V: Ord> {
    edges: Vec<(usize, usize)>,
    vertex_mapping: BTreeMap<V, usize>
}

impl<V: Ord> KnowledgeGraph<V> {
    /// Create a new, empty `KnowledgeGraph`.
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            vertex_mapping: BTreeMap::new()
        }
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
        let index1 = self.add_vertex(v1);
        let index2 = self.add_vertex(v2);
        self.edges.push((index1, index2));
    }

    /// Remaps the internal vertex IDs. The input slice `new_mapping` will have
    /// one entry per vertex. Each entry's index will correspond to the vertex's
    /// old ID, while the entry itself will correspond to the new ID.
    pub fn remap_vertices(&mut self, new_mapping: &[usize]) {
        for old_id in self.vertex_mapping.values_mut() {
            let new_id = new_mapping[*old_id];
            *old_id = new_id;
        }

        for (old_src, old_dst) in self.edges.iter_mut() {
            *old_src = new_mapping[*old_src];
            *old_dst = new_mapping[*old_dst];
        }
    }

    /// Write the graph as a dot file to the given path.
    pub fn write_to_dot_file<P>(&self, path: P) -> Result<(), Box<dyn Error>> 
    where 
        P: AsRef<Path>,
        V: Display
    {
        let mut file = File::create(path)?;

        // Write file header
        file.write_all(b"graph {\n")?;

        // Write vertex labels, ensuring they are sorted by ID
        let mut id_to_vertex_mapping = BTreeMap::new();
        for (v, id) in &self.vertex_mapping {
            file.write_fmt(format_args!("\t{} [label=\"{} ({})\"]\n", v, v, id))?;
            id_to_vertex_mapping.insert(*id, v);
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
        vs.sort();

        // Convert the graph to this format
        let mut g = Self::new();
        for v in vs {
            g.add_vertex(v.to_string());
        }
        for e in can_graph.edges.set {
            g.add_edge(e.from, e.to);
        }

        Ok(g)
    }
}

impl<V: Ord> FromIterator<(V, V)> for KnowledgeGraph<V> {
    /// Constructs a `KnowledgeGraph<V>` from an iterator of tuples representing
    /// edges in the form `(V, V)`, where the first `V` represents the source 
    /// vertex of the edge, and the second represents the destination.
    /// 
    /// # Example
    /// 
    /// ```
    /// use crate::KnowledgeGraph;
    /// 
    /// let g: KnowledgeGraph<_, _> = [
    ///     ("v1", "v2"),
    ///     ("v1", "v3")
    /// ].into_iter().collect();
    /// 
    /// assert_eq!(g.edges.len(), 2);
    /// assert_eq!(g.vertex_mapping.len(), 3);
    /// assert_eq!(g.edges[0], (0, 1));
    /// assert_eq!(g.edges[1], (0, 2));
    /// ```
    fn from_iter<T: IntoIterator<Item = (V, V)>>(iter: T) -> Self {
        let mut graph = Self::new();
        for (v1, v2) in iter {
            graph.add_edge(v1, v2);
        }

        graph
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
        
        let remapping = vec![1, 2, 0];
        g.remap_vertices(&remapping);

        assert_eq!(g.get_vertex_id(&"v1"), Some(1));
        assert_eq!(g.get_vertex_id(&"v2"), Some(2));
        assert_eq!(g.get_vertex_id(&"v3"), Some(0));
        assert_eq!(g.edges.len(), 2);
        assert_eq!(g.vertex_mapping.len(), 3);
        assert_eq!(g.edges[0], (1, 2));
        assert_eq!(g.edges[1], (1, 0));
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
}