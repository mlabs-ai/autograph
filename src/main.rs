#[allow(dead_code)]
mod knowledge_graph;

use std::error::Error;

use knowledge_graph::KnowledgeGraph;

fn main() -> Result<(), Box<dyn Error>> {
    // Load a graph, shuffle it, and write it
    let mut graph = KnowledgeGraph::from_dot_file("tests/verifiers/tiny.dot")?;
    graph.shuffle_vertex_ids(2);
    graph.write_to_dot_file("data/test/0.dot")?;

    // Perform 10 clusters, writing the results of each iteration
    for i in 0..10 {
        graph.cluster();
        let filename = format!("data/test/{}.dot", i + 1);
        graph.write_to_dot_file(filename)?;
    }

    Ok(())
}
