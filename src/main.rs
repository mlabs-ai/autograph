#[allow(dead_code)]
mod knowledge_graph;

use std::error::Error;
use std::path::Path;

use knowledge_graph::KnowledgeGraph;

fn show_progression<P: AsRef<Path>>(
    path: P,
    num_iterations: usize,
    graph_shuffle_seed: u64
) -> Result<(), Box<dyn Error>> {
    // Load a graph, shuffle it, and write it
    let mut graph = KnowledgeGraph::from_dot_file(path)?;
    graph.shuffle_vertex_ids(graph_shuffle_seed);
    graph.write_to_dot_file("data/test/0.dot")?;

    // Perform the specified number of clusters, writing the results of each iteration
    for i in 0..num_iterations {
        graph.cluster();
        let filename = format!("data/test/{}.dot", i + 1);
        graph.write_to_dot_file(filename)?;
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    show_progression("tests/verifiers/small.dot", 10, 2)?;

    Ok(())
}
