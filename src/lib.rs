#[pyo3::pymodule]
mod autograph {
    use pyo3::exceptions::{PyIOError, PyValueError};
    use pyo3::{PyResult, prelude::*};

    use autograph_core::assignment::{FrameGraph, Memberships};
    use autograph_core::graph_builder::GraphBuilder;
    use autograph_core::knowledge_graph::KnowledgeGraph;

    #[pyclass(name = "KnowledgeGraph", subclass)]
    pub struct KnowledgeGraphWrapper {
        graph: KnowledgeGraph<String>,
    }

    #[pymethods]
    impl KnowledgeGraphWrapper {
        #[new]
        fn new() -> Self {
            Self {
                graph: KnowledgeGraph::new(),
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
        fn from_wikidata(path: &str, relationship: &str) -> PyResult<Self> {
            KnowledgeGraph::from_wikidata(path, relationship)
                .map(|graph| KnowledgeGraphWrapper { graph })
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyIOError, _>(error)
                })
        }

        #[staticmethod]
        fn from_wikidata_bz2(path: &str, relationship: &str) -> PyResult<Self> {
            KnowledgeGraph::from_wikidata_bz2(path, relationship)
                .map(|graph| KnowledgeGraphWrapper { graph })
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyIOError, _>(error)
                })
        }

        fn edge_list(&self) -> Vec<(String, String)> {
            self.graph.edge_list()
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

        fn num_edges(&self) -> usize {
            self.graph.num_edges()
        }

        fn shuffle_vertex_ids(&mut self, seed: u64) {
            self.graph.shuffle_vertex_ids(seed);
        }

        fn as_matrix(&self) -> Vec<Vec<usize>> {
            self.graph.as_matrix()
        }

        fn cluster_step(&mut self, factor: f64, from_idx: usize, to_idx: usize) -> Vec<f64> {
            let range = from_idx..to_idx;
            self.graph.cluster_step(factor, &range)
        }

        fn cluster(
            &mut self,
            factor: f64,
            steps_before_subdivide: usize,
            boundary_threshold: f64,
            min_cluster_size: usize,
        ) {
            self.graph.cluster(
                factor,
                steps_before_subdivide,
                boundary_threshold,
                min_cluster_size,
            );
        }

        fn get_clusters(&self) -> Vec<Vec<String>> {
            self.graph.get_clusters()
        }
    }

    #[pyclass(name = "GraphBuilder", subclass)]
    pub struct GraphBuilderWrapper {
        builder: Option<GraphBuilder>,
    }

    #[pymethods]
    impl GraphBuilderWrapper {
        #[new]
        fn new(seed: u64) -> Self {
            Self {
                builder: Some(GraphBuilder::new(seed)),
            }
        }

        fn add_scale_free_cluster(
            &mut self,
            num_nodes: usize,
            new_edges: usize,
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

        fn add_dense_cluster(&mut self, num_nodes: usize, edge_density: f64) -> PyResult<usize> {
            self.builder
                .as_mut()
                .ok_or("Builder has been finalized and should not be used".into())
                .and_then(|b| b.add_dense_cluster(num_nodes, edge_density))
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyValueError, _>(error)
                })
        }

        fn add_random_link(&mut self, cluster1_id: usize, cluster2_id: usize) -> PyResult<()> {
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
            cluster2_node_id: usize,
        ) -> PyResult<()> {
            self.builder
                .as_mut()
                .ok_or("Builder has been finalized and should not be used".into())
                .and_then(|b| {
                    b.add_link(cluster1_id, cluster2_id, cluster1_node_id, cluster2_node_id)
                })
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyValueError, _>(error)
                })
        }

        fn get_cluster(&self, cluster_id: usize) -> Option<&Vec<usize>> {
            self.builder
                .as_ref()
                .and_then(|b| b.get_cluster(cluster_id))
        }

        fn finalize_graph(&mut self) -> PyResult<KnowledgeGraphWrapper> {
            self.builder
                .take()
                .ok_or("Builder has been finalized and should not be used")
                .map(|graph| {
                    let graph = graph.finalize_graph();
                    KnowledgeGraphWrapper {
                        graph: (&graph).into(),
                    }
                })
                .map_err(|e| {
                    let error = format!("Error: {}", e);
                    PyErr::new::<PyValueError, _>(error)
                })
        }
    }

    #[pyclass(name = "FrameGraph", subclass)]
    pub struct FrameGraphWrapper {
        graph: FrameGraph,
    }

    #[pymethods]
    impl FrameGraphWrapper {
        #[new]
        fn new() -> Self {
            Self {
                graph: FrameGraph::new(),
            }
        }

        #[staticmethod]
        #[pyo3(signature = (path, predicates, limit=None))]
        fn from_wikidata_bz2(
            path: &str,
            predicates: Vec<String>,
            limit: Option<usize>,
        ) -> PyResult<Self> {
            let preds: Vec<&str> = predicates.iter().map(String::as_str).collect();
            FrameGraph::from_wikidata_multi_bz2(path, &preds, limit)
                .map(|graph| FrameGraphWrapper { graph })
                .map_err(|e| PyErr::new::<PyIOError, _>(format!("Error: {}", e)))
        }

        fn add_edge(&mut self, src: &str, predicate: &str, dst: &str) -> PyResult<()> {
            self.graph
                .add_edge(src, predicate, dst)
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("Error: {}", e)))
        }

        fn num_entities(&self) -> usize {
            self.graph.num_entities()
        }

        fn num_predicates(&self) -> usize {
            self.graph.num_predicates()
        }

        /// Entity QIDs in ID order, so `argmax_entities()` can be interpreted.
        fn entity_labels(&self) -> Vec<String> {
            self.graph.all_entity_labels().to_vec()
        }

        /// Predicate property IDs in ID order, so `argmax_predicates()` can be
        /// interpreted.
        fn predicate_labels(&self) -> Vec<String> {
            self.graph.all_predicate_labels().to_vec()
        }

        fn num_edges(&self) -> usize {
            self.graph.num_edges()
        }

        fn cluster(
            &self,
            factor: f64,
            steps_before_subdivide: usize,
            boundary_threshold: f64,
            min_cluster_size: usize,
        ) -> Vec<Vec<String>> {
            self.graph.cluster(
                factor,
                steps_before_subdivide,
                boundary_threshold,
                min_cluster_size,
            )
        }

        /// Runs EM assignment seeded by the given frames (the output of
        /// `cluster`, or a previously-computed factorization), returning soft
        /// memberships.
        #[pyo3(signature = (frames, epsilon=1e-3, tol=1e-6, max_iters=100))]
        fn em_assign(
            &self,
            frames: Vec<Vec<String>>,
            epsilon: f64,
            tol: f64,
            max_iters: usize,
        ) -> PyResult<MembershipsWrapper> {
            let seed = self
                .graph
                .seed_from_frames(&frames)
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("Error: {}", e)))?;
            let memberships = self.graph.em_assign(&seed, epsilon, tol, max_iters);
            Ok(MembershipsWrapper { memberships })
        }
    }

    #[pyclass(name = "Memberships")]
    pub struct MembershipsWrapper {
        memberships: Memberships,
    }

    #[pymethods]
    impl MembershipsWrapper {
        fn num_frames(&self) -> usize {
            self.memberships.num_frames
        }

        fn num_entities(&self) -> usize {
            self.memberships.num_entities
        }

        fn num_predicates(&self) -> usize {
            self.memberships.num_predicates
        }

        #[getter]
        fn theta(&self) -> Vec<Vec<f64>> {
            self.memberships.theta.clone()
        }

        #[getter]
        fn phi(&self) -> Vec<Vec<f64>> {
            self.memberships.phi.clone()
        }

        fn argmax_entities(&self) -> Vec<usize> {
            self.memberships.argmax_entities()
        }

        fn argmax_predicates(&self) -> Vec<usize> {
            self.memberships.argmax_predicates()
        }
    }
}
