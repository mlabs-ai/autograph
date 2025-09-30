#[allow(dead_code)]
mod knowledge_graph;
#[allow(dead_code)]
mod graph_builder;
mod renderers;

use std::env;
use std::error::Error;
use std::path::Path;

use graph_builder::GraphBuilder;
use knowledge_graph::KnowledgeGraph;
use renderers::StdRendererOutput;
use renderers::dot_files::DotFileRenderer;
use renderers::adjacency_matrix_images::AdjacencyMatrixImageRenderer;

fn show_progression<P: AsRef<Path>>(
    path: P,
    num_iterations: usize,
    shuffle_seed: u64,
    factor: f64
) -> Result<(), Box<dyn Error>> {
    // Get dot file base name and set up the renderers
    let dot_file_name = path
        .as_ref()
        .file_stem()
        .and_then(|p| p.to_str())
        .ok_or("Could not get dot file base name")?;
    let mut dot_renderer = DotFileRenderer::from_graph_parameters(
        dot_file_name,
        shuffle_seed,
        factor
    )?;
    let mut adj_img_renderer = AdjacencyMatrixImageRenderer::from_graph_parameters(
        dot_file_name,
        shuffle_seed,
        factor
    )?;

    // Load the graph, shuffle it, and write it. This is the starting point
    let mut graph = KnowledgeGraph::from_dot_file(&path)?;
    graph.shuffle_vertex_ids(shuffle_seed);
    dot_renderer.render(&graph)?;
    adj_img_renderer.render(1024, 1024, &graph)?;

    // Perform the specified number of clusters, writing the results of each iteration
    for _ in 0..num_iterations {
        graph.cluster(factor);
        dot_renderer.render(&graph)?;
        adj_img_renderer.render(1024, 1024, &graph)?;
    }

    Ok(())
}

fn generate_graph() -> Result<(), Box<dyn Error>> {
    let mut builder = GraphBuilder::new(1);
    builder.add_dense_cluster(50, 0.95)?;
    builder.add_dense_cluster(40, 0.8)?;
    builder.add_dense_cluster(30, 0.75)?;
    builder.add_dense_cluster(75, 0.6)?;
    builder.add_random_link(0, 1)?;
    builder.add_random_link(1, 2)?;
    builder.add_random_link(2, 3)?;
    builder.add_random_link(0, 3)?;
    builder.finalize_graph().write_to_dot_file("tests/verifiers/medium_1.dot")
}

fn progressions() -> Result<(), Box<dyn Error>> {
    let graphs = [
        "tests/verifiers/tiny.dot",
        "tests/verifiers/small_0.dot",
        "tests/verifiers/small_1.dot",
        "tests/verifiers/medium_0.dot",
        "tests/verifiers/medium_1.dot",
    ];
    let factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0];

    for graph in graphs {
        for seed in 0..10 {
            for factor in factors {
                show_progression(graph, 10, seed, factor)?;
            }
        }
    }

    Ok(())
}



fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 2 || (args[1] != "-p" && args[1] != "-g") {
        panic!("Currently, this program requires the \"-p\" or \"-g\" argument");
    }

    if args[1] == "-p" {
        progressions()
    }
    else {
        generate_graph()
    }
}
