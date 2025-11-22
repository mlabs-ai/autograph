#[pyo3::pymodule]
mod autograph {
    use pyo3::exceptions::{PyIOError, PyValueError};
    use pyo3::prelude::*;

    use autograph_core::graph_builder::GraphBuilder;
    use autograph_core::knowledge_graph::KnowledgeGraph;

    #[pyclass(name = "KnowledgeGraph", subclass)]
    pub struct KnowledgeGraphWrapper {
        graph: KnowledgeGraph<usize>
    }

    #[pymethods]
    impl KnowledgeGraphWrapper {
        #[new]
        fn new() -> Self {
            Self {
                graph: KnowledgeGraph::new()
            }
        }

        #[staticmethod]
        fn from_dot_file(path: &str) -> PyResult<Self> {
            KnowledgeGraph::from_dot_file(path)
                .map(|graph| {
                    let graph: KnowledgeGraph<usize> = (&graph).into();
                    KnowledgeGraphWrapper { graph }
                })
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyIOError, _>(error)
                })
        }

        fn write_to_dot_file(&self, path: &str) -> PyResult<()> {
            self.graph.write_to_dot_file(path).map_err(|e| {
                let error = format!("Error: {}", e);
                PyErr::new::<PyIOError, _>(error)
            })
        }

        fn num_vertices(&self) -> usize {
            self.graph.num_vertices()
        }

        fn shuffle_vertex_ids(&mut self, seed: u64) {
            self.graph.shuffle_vertex_ids(seed);
        }

        fn as_matrix(&self) -> Vec<Vec<usize>> {
            self.graph.as_matrix()
        }

        fn cluster_step(&mut self, factor: f64) -> Vec<f64> {
            self.graph.cluster_step(factor)
        }

        fn split_density(&self) -> Vec<f64> {
            self.graph.split_density()
        }
    }

    #[pyclass(name = "GraphBuilder", subclass)]
    pub struct GraphBuilderWrapper {
        builder: Option<GraphBuilder>
    }

    #[pymethods]
    impl GraphBuilderWrapper {
        #[new]
        fn new(seed: u64) -> Self {
            Self {
                builder: Some(GraphBuilder::new(seed))
            }
        }

        fn add_scale_free_cluster(
            &mut self,
            num_nodes: usize,
            new_edges: usize
        ) -> PyResult<usize> {
            self.builder
                .as_mut()
                .ok_or("Builder has been finalized and should not be used".into())
                .and_then(|b| b.add_scale_free_cluster(num_nodes, new_edges))
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyValueError, _>(error)
                })
        }

        fn add_dense_cluster(
            &mut self,
            num_nodes: usize,
            edge_density: f64
        ) -> PyResult<usize> {
            self.builder
                .as_mut()
                .ok_or("Builder has been finalized and should not be used".into())
                .and_then(|b| b.add_dense_cluster(num_nodes, edge_density))
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyValueError, _>(error)
                })
        }

        fn add_bipartite_cluster(
            &mut self,
            num_nodes_a: usize,
            num_nodes_b: usize
        ) -> PyResult<usize> {
            self.builder
                .as_mut()
                .ok_or("Builder has been finalized and should not be used".into())
                .and_then(|b| b.add_bipartite_cluster(num_nodes_a, num_nodes_b))
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyValueError, _>(error)
                })
        }

        fn add_circle_cluster(
            &mut self,
            num_nodes: usize
        ) -> PyResult<usize> {
            self.builder
                .as_mut()
                .ok_or("Builder has been finalized and should not be used".into())
                .and_then(|b| b.add_circle_cluster(num_nodes))
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyValueError, _>(error)
                })
        }

        fn add_random_link(
            &mut self,
            cluster1_id: usize,
            cluster2_id: usize
        ) -> PyResult<()> {
            self.builder
                .as_mut()
                .ok_or("Builder has been finalized and should not be used".into())
                .and_then(|b| b.add_random_link(cluster1_id, cluster2_id))
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyValueError, _>(error)
                })
        }

        fn add_link(
            &mut self,
            cluster1_id: usize,
            cluster2_id: usize,
            cluster1_node_id: usize,
            cluster2_node_id: usize
        ) -> PyResult<()> {
            self.builder
                .as_mut()
                .ok_or("Builder has been finalized and should not be used".into())
                .and_then(|b| 
                    b.add_link(cluster1_id, cluster2_id, cluster1_node_id, cluster2_node_id)
                )
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyValueError, _>(error)
                })
        }

        fn get_cluster(&self, cluster_id: usize) -> Option<&Vec<usize>> {
            self.builder.as_ref().and_then(|b| b.get_cluster(cluster_id))
        }

        fn finalize_graph(&mut self) -> PyResult<KnowledgeGraphWrapper> {
            self.builder
                .take()
                .ok_or("Builder has been finalized and should not be used")
                .map(|graph| KnowledgeGraphWrapper { graph: graph.finalize_graph() })
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyValueError, _>(error)
                })
        }
    }
}