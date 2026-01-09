#[pyo3::pymodule]
mod autograph {
    use pyo3::exceptions::{PyIOError, PyValueError};
    use pyo3::prelude::*;

    use autograph_core::graph_builder::GraphBuilder;
    use autograph_core::knowledge_graph::KnowledgeGraph;

    #[pyclass(name = "KnowledgeGraph", subclass)]
    pub struct KnowledgeGraphWrapper {
        graph: KnowledgeGraph<String>
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
                    KnowledgeGraphWrapper { graph }
                })
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyIOError, _>(error)
                })
        }

        #[staticmethod]
        fn from_wikidata(path: &str, relationship: &str) -> PyResult<Self> {
            KnowledgeGraph::from_wikidata(path, relationship)
                .map(|graph| {
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

        fn cluster_step(
            &mut self,
            factor: f64,
            from_idx: usize, 
            to_idx: usize
        ) -> Vec<f64> {
            let range = from_idx..to_idx;
            self.graph.cluster_step(factor, &range)
        }

        fn cluster(
            &mut self,
            factor: f64,
            steps_before_subdivide: usize,
            boundary_threshold: f64,
            min_cluster_size: usize
        ) {
            self.graph.cluster(
                factor,
                steps_before_subdivide,
                boundary_threshold,
                min_cluster_size
            );
        }

        fn get_clusters(&self) -> Vec<Vec<String>> {
            self.graph.get_clusters()
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
                .map(|graph| {
                    let graph = graph.finalize_graph();
                    KnowledgeGraphWrapper { graph: (&graph).into() }
                })
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyValueError, _>(error)
                })
        }
    }
}