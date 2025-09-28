use crate::knowledge_graph::KnowledgeGraph;

use itertools::Itertools;

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::mem;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;

/// `GraphBuilder` is a struct that allows one to easily construct a complex
/// `KnowledgeGraph<String>`.
pub struct GraphBuilder {
    graph: KnowledgeGraph<String>,
    clusters: Vec<Vec<String>>,
    links: HashMap<usize, HashMap<usize, HashSet<(usize, usize)>>>,
    rng: StdRng
}

macro_rules! check_condition {
    ($cond:expr, $($msg:expr),+) => {
        if !($cond) {
            return Err(format!($($msg),+).into());
        }
    };
}

impl GraphBuilder {
    /// Create a new `GraphBuilder`. A `seed` is given for reproducability.
    pub fn new(seed: u64) -> Self {
        Self {
            graph: KnowledgeGraph::new(),
            clusters: Vec::new(),
            links: HashMap::new(),
            rng: StdRng::seed_from_u64(seed)
        }
    }

    /// Create a cluster. Each node has a probility `edge_density` of being
    /// connected to any other node in the cluster.
    /// 
    /// Fails if `num_nodes` is 0 or `edge_density` is not in the range [0.0,
    /// 1.0].
    pub fn add_dense_cluster(
        &mut self,
        num_nodes: usize,
        edge_density: f64
    ) -> Result<usize, Box<dyn Error>> {
        // Perform some sanity checks
        check_condition!(num_nodes > 0, "Cluster can't be empty");
        check_condition!(
            (0.0..=1.0).contains(&edge_density),
            "Edge density must be between 0 and 1"
        );

        // Determine the ID for the new cluster
        let cluster_id = self.clusters.len();

        // Create and name all nodes in this cluster
        let mut cluster_nodes = Vec::new();
        for i in 0..num_nodes {
            let node_id = format!("d{}_n{}", cluster_id, i);
            cluster_nodes.push(node_id.clone());
            self.graph.add_vertex(node_id);
        }

        // Add edges probabilistically
        cluster_nodes.iter()
            .array_combinations()
            .filter(|_| self.rng.random_bool(edge_density))
            .for_each(|[v1, v2]| {
                self.graph.add_edge(v1.clone(), v2.clone())
            });

        // Record the cluster
        self.clusters.push(cluster_nodes);

        Ok(cluster_id)
    }

    /// Create a bipartite cluster. One side will have `num_nodes_a` nodes, while
    /// the other will have `num_nodes_b`. Each node on side a will be connected
    /// to every node on side b.
    /// 
    /// Fails if either `num_nodes_a` or `num_nodes_b` is 0.
    pub fn add_bipartite_cluster(
        &mut self,
        num_nodes_a: usize,
        num_nodes_b: usize
    ) -> Result<usize, Box<dyn Error>> {
        // Perform some sanity checks
        check_condition!(
            num_nodes_a > 0 && num_nodes_b > 0,
            "Cluster can't be empty"
        );

        // Determine the ID for the new cluster
        let cluster_id = self.clusters.len();

        // Create all nodes on side A
        let mut cluster_nodes_a = Vec::new();
        for i in 0..num_nodes_a {
            let node_id = format!("b{}_a_n{}", cluster_id, i);
            cluster_nodes_a.push(node_id.clone());
            self.graph.add_vertex(node_id);
        }

        // Create all nodes on side B
        let mut cluster_nodes_b = Vec::new();
        for i in 0..num_nodes_b {
            let node_id = format!("b{}_b_n{}", cluster_id, i);
            cluster_nodes_b.push(node_id.clone());
            self.graph.add_vertex(node_id);
        }

        // Create all edges
        for v1 in &cluster_nodes_a {
            for v2 in &cluster_nodes_b {
                self.graph.add_edge(v1.clone(), v2.clone());
            }
        }

        // Record the cluster
        cluster_nodes_a.append(&mut cluster_nodes_b);
        self.clusters.push(cluster_nodes_a);

        Ok(cluster_id)
    }

    /// Create a cluster where all the nodes are connected in a ring structure.
    /// 
    /// Fails if `num_nodes` is 0.
    pub fn add_circle_cluster(
        &mut self,
        num_nodes: usize
    ) -> Result<usize, Box<dyn Error>> {
        check_condition!(num_nodes > 0, "Cluster can't be empty");

        // Determine the ID for the new cluster
        let cluster_id = self.clusters.len();

        // Create and name all nodes in this cluster
        let mut cluster_nodes = Vec::new();
        for i in 0..num_nodes {
            let node_id = format!("c{}_n{}", cluster_id, i);
            cluster_nodes.push(node_id.clone());
            self.graph.add_vertex(node_id);
        }

        // Add edges
        for i in 0..cluster_nodes.len() {
            let j = (i + 1) % cluster_nodes.len();
            self.graph.add_edge(cluster_nodes[i].clone(), cluster_nodes[j].clone());
        }

        // Record the cluster
        self.clusters.push(cluster_nodes);

        Ok(cluster_id)
    }

    /// Create a link between two clusters. The link will be randomly chosen from
    /// any two nodes that do not already have an edge.
    /// 
    /// Fails under any of the following cicumstances:
    /// - `cluster1_id` and `cluster2_id` are the same.
    /// - Either cluster does not already exist.
    /// - All possible links have already been created.
    pub fn add_random_link(
        &mut self,
        mut cluster1_id: usize,
        mut cluster2_id: usize
    ) -> Result<(), Box<dyn Error>> {
        // Perform some sanity checks
        check_condition!(
            cluster1_id != cluster2_id,
            "Links must be made between separate clusters"
        );
        check_condition!(
            cluster1_id < self.clusters.len(),
            "Cluster {} does not exist",
            cluster1_id
        );
        check_condition!(
            cluster2_id < self.clusters.len(),
            "Cluster {} does not exist",
            cluster2_id
        );

        // Make sure c1 < c2
        if cluster2_id < cluster1_id {
            mem::swap(&mut cluster1_id, &mut cluster2_id);
        }

        // Determine number of links present vs number of possible links
        let num_existing_links = self.links
            .get(&cluster1_id)
            .and_then(|clusters| clusters.get(&cluster2_id))
            .map(|links| links.len())
            .unwrap_or(0);
        let cluster1_nodes = &self.clusters[cluster1_id];
        let cluster2_nodes = &self.clusters[cluster2_id];
        let num_possible_links = cluster1_nodes.len() * cluster2_nodes.len();

        // Make sure there are links that can be made
        check_condition!(
            num_existing_links < num_possible_links,
            "All possible links between clusters {} and {} have been made",
            cluster1_id,
            cluster2_id
        );

        // Since we are not memoizing the list of remaining links between every
        // cluster, the fastest approach for finding a random link that can be
        // added is to randomly choose new edges until we find one that has not 
        // been created. The other main approach of creating a list of links
        // that have not yet been created and randomly choosing from that is
        // only faster if the number of remaining links is precisely 1.
        let (cluster1_node_id, cluster2_node_id) = loop {
            // Select random nodes from each cluster
            let cluster1_node_id = (0..cluster1_nodes.len())
                .choose(&mut self.rng)
                .unwrap();
            let cluster2_node_id = (0..cluster2_nodes.len())
                .choose(&mut self.rng)
                .unwrap();
            let link_candidate = (cluster1_node_id, cluster2_node_id);

            // If the link is not already present, we have found our new link
            let link_present = self.links
                .get(&cluster1_id)
                .and_then(|clusters| clusters.get(&cluster2_id))
                .map(|links| links.contains(&link_candidate))
                .unwrap_or(false);
            if !link_present {
                break link_candidate;
            }
        };
        
        // Record the link
        self.add_link_unchecked(
            cluster1_id, 
            cluster2_id, 
            cluster1_node_id, 
            cluster2_node_id
        );
        Ok(())
    }

    /// Add a link between a specific node in one cluster and a specific node in
    /// another cluster.
    /// 
    /// Fails under any of the following cicumstances:
    /// - `cluster1_id` and `cluster2_id` are the same.
    /// - Either cluster does not already exist.
    /// - Either specified node does not exist in its cluster.
    /// - The specified link has already been created.
    pub fn add_link(
        &mut self,
        mut cluster1_id: usize,
        mut cluster2_id: usize,
        mut cluster1_node_id: usize,
        mut cluster2_node_id: usize
    ) -> Result<(), Box<dyn Error>> {
        // Perform some sanity checks
        check_condition!(
            cluster1_id != cluster2_id,
            "Links must be made between separate clusters"
        );
        check_condition!(
            cluster1_id < self.clusters.len(),
            "Cluster {} does not exist",
            cluster1_id
        );
        check_condition!(
            cluster2_id < self.clusters.len(),
            "Cluster {} does not exist",
            cluster2_id
        );
        check_condition!(
            cluster1_node_id < self.clusters[cluster1_id].len(),
            "Cluster {} does not have node {}",
            cluster1_id,
            cluster1_node_id
        );
        check_condition!(
            cluster2_node_id < self.clusters[cluster2_id].len(),
            "Cluster {} does not have node {}",
            cluster2_id,
            cluster2_node_id
        );

        // Make sure c1 < c2
        if cluster2_id < cluster1_id {
            mem::swap(&mut cluster1_id, &mut cluster2_id);
            mem::swap(&mut cluster1_node_id, &mut cluster2_node_id);
        }

        // Determine if the link already exists
        let link = (cluster1_node_id, cluster2_node_id);
        let link_present = self.links
            .get(&cluster1_id)
            .and_then(|clusters| clusters.get(&cluster2_id))
            .map(|links| links.contains(&link))
            .unwrap_or(false);
        check_condition!(!link_present, "The specified link already exists");

        // Record the link and add it to the graph
        self.add_link_unchecked(
            cluster1_id, 
            cluster2_id, 
            cluster1_node_id, 
            cluster2_node_id
        );
        Ok(())
    }

    /// Get the list of nodes in the specified cluster.
    pub fn get_cluster(&self, cluster_id: usize) -> Option<&Vec<String>> {
        self.clusters.get(cluster_id)
    }

    /// Consume the builder and return the resulting `KnowledgeGraph<String>`.
    pub fn finalize_graph(self) -> KnowledgeGraph<String> {
        self.graph
    }

    fn add_link_unchecked(
        &mut self,
        cluster1_id: usize,
        cluster2_id: usize,
        cluster1_node_id: usize,
        cluster2_node_id: usize
    ) {
        // Record the link
        self.links
            .entry(cluster1_id).or_default()
            .entry(cluster2_id).or_default()
            .insert((cluster1_node_id, cluster2_node_id));

        // Add it to the graph
        let cluser1_node_name = &self.clusters[cluster1_id][cluster1_node_id];
        let cluser2_node_name = &self.clusters[cluster2_id][cluster2_node_id];
        self.graph.add_edge(cluser1_node_name.clone(), cluser2_node_name.clone());
    }
}

#[cfg(test)]
mod tests {
    use crate::graph_builder::GraphBuilder;
    use crate::knowledge_graph::KnowledgeGraph;

    #[test]
    fn empty_graph() {
        let builder = GraphBuilder::new(0);
        let graph = KnowledgeGraph::new();

        assert_eq!(graph, builder.finalize_graph());
    }

    #[test]
    fn full_single_cluster() {
        for seed in 0..10 {
            let mut builder = GraphBuilder::new(seed);
            builder.add_dense_cluster(3, 1.0).unwrap();

            let graph: KnowledgeGraph<_> = [
                ("d0_n0".to_owned(), "d0_n1".to_owned()),
                ("d0_n0".to_owned(), "d0_n2".to_owned()),
                ("d0_n1".to_owned(), "d0_n2".to_owned())
            ].into_iter().collect();

            assert_eq!(graph, builder.finalize_graph());
        }
    }

    #[test]
    fn single_cluster_no_edges() {
        for seed in 0..10 {
            let mut builder = GraphBuilder::new(seed);
            builder.add_dense_cluster(3, 0.0).unwrap();

            let mut graph = KnowledgeGraph::new();
            graph.add_vertex("d0_n0".to_owned());
            graph.add_vertex("d0_n1".to_owned());
            graph.add_vertex("d0_n2".to_owned());

            assert_eq!(graph, builder.finalize_graph());
        }
    }

    #[test]
    fn circle() {
        let mut builder = GraphBuilder::new(0);
        builder.add_circle_cluster(3).unwrap();

        let graph: KnowledgeGraph<_> = [
            ("c0_n0".to_owned(), "c0_n1".to_owned()),
            ("c0_n1".to_owned(), "c0_n2".to_owned()),
            ("c0_n0".to_owned(), "c0_n2".to_owned())
        ].into_iter().collect();

        assert_eq!(graph, builder.finalize_graph());
    }

    #[test]
    fn bipartite() {
        let mut builder = GraphBuilder::new(0);
        builder.add_bipartite_cluster(3, 2).unwrap();

        let mut graph = KnowledgeGraph::new();
        graph.add_vertex("b0_a_n0".to_owned());
        graph.add_vertex("b0_a_n1".to_owned());
        graph.add_vertex("b0_a_n2".to_owned());
        graph.add_vertex("b0_b_n0".to_owned());
        graph.add_vertex("b0_b_n1".to_owned());

        let edges = [
            ("b0_a_n0".to_owned(), "b0_b_n0".to_owned()),
            ("b0_a_n0".to_owned(), "b0_b_n1".to_owned()),
            ("b0_a_n1".to_owned(), "b0_b_n0".to_owned()),
            ("b0_a_n1".to_owned(), "b0_b_n1".to_owned()),
            ("b0_a_n2".to_owned(), "b0_b_n0".to_owned()),
            ("b0_a_n2".to_owned(), "b0_b_n1".to_owned()),
        ];
        for (v1, v2) in edges {
            graph.add_edge(v1, v2);
        }

        assert_eq!(graph, builder.finalize_graph());
    }

    #[test]
    fn full_two_clusters_unconnected() {
        for seed in 0..10 {
            let mut builder = GraphBuilder::new(seed);
            builder.add_dense_cluster(2, 1.0).unwrap();
            builder.add_dense_cluster(2, 1.0).unwrap();

            let graph: KnowledgeGraph<_> = [
                ("d0_n0".to_owned(), "d0_n1".to_owned()),
                ("d1_n0".to_owned(), "d1_n1".to_owned()),
            ].into_iter().collect();

            assert_eq!(graph, builder.finalize_graph());
        }
    }

    #[test]
    fn full_two_clusters_connected() {
        for seed in 0..10 {
            let mut builder = GraphBuilder::new(seed);
            builder.add_dense_cluster(2, 1.0).unwrap();
            builder.add_dense_cluster(2, 1.0).unwrap();
            builder.add_random_link(0, 1).unwrap();
            builder.add_random_link(0, 1).unwrap();
            builder.add_random_link(0, 1).unwrap();
            builder.add_random_link(0, 1).unwrap();
            assert!(builder.add_random_link(0, 1).is_err());

            let graph: KnowledgeGraph<_> = [
                ("d0_n0".to_owned(), "d0_n1".to_owned()),
                ("d1_n0".to_owned(), "d1_n1".to_owned()),
                ("d0_n0".to_owned(), "d1_n0".to_owned()),
                ("d0_n0".to_owned(), "d1_n1".to_owned()),
                ("d0_n1".to_owned(), "d1_n0".to_owned()),
                ("d0_n1".to_owned(), "d1_n1".to_owned())
            ].into_iter().collect();

            assert_eq!(graph, builder.finalize_graph());
        }
    }

    #[test]
    fn failures() {
        let mut builder = GraphBuilder::new(0);
        assert!(builder.add_dense_cluster(0, 1.0).is_err());
        assert!(builder.add_dense_cluster(2, 1.1).is_err());
        assert!(builder.add_dense_cluster(2, -0.1).is_err());
        assert!(builder.add_dense_cluster(0, 1.1).is_err());
        assert!(builder.add_dense_cluster(0, -0.1).is_err());

        assert!(builder.add_bipartite_cluster(2, 0).is_err());
        assert!(builder.add_bipartite_cluster(0, 2).is_err());
        assert!(builder.add_bipartite_cluster(0, 0).is_err());

        assert!(builder.add_circle_cluster(0).is_err());

        assert!(builder.add_random_link(0, 0).is_err());
        assert!(builder.add_link(0, 0, 0, 0).is_err());

        builder.add_dense_cluster(5, 1.0).unwrap();

        assert!(builder.add_random_link(0, 1).is_err());
        assert!(builder.add_random_link(1, 0).is_err());
        assert!(builder.add_link(0, 1, 0, 0).is_err());
        assert!(builder.add_link(1, 0, 0, 0).is_err());

        builder.add_dense_cluster(5, 1.0).unwrap();

        assert!(builder.add_link(0, 1, 0, 5).is_err());
        assert!(builder.add_link(1, 0, 5, 0).is_err());

        builder.add_link(0, 1, 0, 0).unwrap();
        assert!(builder.add_link(0, 1, 0, 0).is_err());
    }
}