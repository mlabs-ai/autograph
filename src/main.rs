#[allow(dead_code)]
mod knowledge_graph;

use knowledge_graph::KnowledgeGraph;

fn main() {
    // Draw a non-descript graph
    let mut graph: KnowledgeGraph<_> = [
        ("v1", "v2"),
        ("v1", "v3")
    ].into_iter().collect();
    graph.write_to_dot_file("data/test/g.dot").unwrap();

    // Rename its vertices
    graph.remap_vertices(&[1, 2, 0]);
    graph.write_to_dot_file("data/test/h.dot").unwrap();
}
