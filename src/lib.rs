pub mod graph_builder;
pub mod knowledge_graph;
pub mod renderers;

#[pyo3::pymodule]
mod autograph {
    use pyo3::exceptions::{PyIOError, PyValueError};
    use pyo3::prelude::*;

    use crate::graph_builder::GraphBuilder;
    use crate::knowledge_graph::KnowledgeGraph;

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
                .map(|graph| KnowledgeGraphWrapper { graph })
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyIOError, _>(error)
                })
        }

        #[staticmethod]
        fn from_edge_list(edge_list: Vec<(String, String)>) -> Self {
            let graph: KnowledgeGraph<_> = edge_list.into_iter().collect();
            Self {
                graph
            }
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

        fn cluster(&mut self, factor: f64) -> Vec<f64> {
            self.graph.cluster(factor)
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

        fn get_cluster(&self, cluster_id: usize) -> Option<&Vec<String>> {
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