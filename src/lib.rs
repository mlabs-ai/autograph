mod graph_builder;
mod knowledge_graph;
mod renderers;

#[pyo3::pymodule]
mod autograph {
    use pyo3::prelude::*;

    #[pyfunction]
    fn sum_as_string(a: i64, b: i64) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}