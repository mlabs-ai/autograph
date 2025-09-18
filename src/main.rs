#[allow(dead_code)]
mod knowledge_graph;

use std::error::Error;
use std::path::Path;

use knowledge_graph::KnowledgeGraph;

fn format_file_name(
    dot_file_name: &str,
    seed: u64,
    factor: f64,
    iteration: usize
) -> String {
    format!(
        "data/test/graph_{}/seed_{}/factor_{}/iter_{}.dot",
        dot_file_name,
        seed,
        factor,
        iteration
    )
}

fn show_progression<P: AsRef<Path>>(
    path: P,
    num_iterations: usize,
    shuffle_seed: u64,
    factor: f64
) -> Result<(), Box<dyn Error>> {
    // Get dot file base name
    let dot_file_name = path
        .as_ref()
        .file_stem()
        .and_then(|p| p.to_str())
        .ok_or("Could not get dot file base name")?;

    // Load the graph, shuffle it, and write it. This is the starting point
    let mut graph = KnowledgeGraph::from_dot_file(&path)?;
    graph.shuffle_vertex_ids(shuffle_seed);
    let filename = format_file_name(dot_file_name, shuffle_seed, factor, 0);
    graph.write_to_dot_file(filename)?;

    // Perform the specified number of clusters, writing the results of each iteration
    for i in 0..num_iterations {
        graph.cluster(factor);
        let filename = format_file_name(dot_file_name, shuffle_seed, factor, i);
        graph.write_to_dot_file(filename)?;
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    show_progression("tests/verifiers/small_1.dot", 10, 2, 1.0)?;

    Ok(())
}
