#[allow(dead_code)]
mod knowledge_graph;

use knowledge_graph::KnowledgeGraph;

fn main() {
    // Draw a non-descript graph
    let mut graph: KnowledgeGraph<_> = [
        ("v1", "v2"),
        ("v1", "v3"),
        ("v1", "v4"),
        ("v2", "v3"),
        ("v2", "v4"),
        ("v3", "v4"),
        ("v4", "v5"),
        ("v5", "v6"),
        ("v6", "v7"),
        ("v7", "v5")
    ].into_iter().collect();
    graph.shuffle_vertex_ids();
    graph.write_to_dot_file("tests/verifiers/tiny.dot").unwrap();
}
