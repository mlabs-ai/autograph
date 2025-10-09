use std::error::Error;
use std::path::{Path, PathBuf};

pub mod adjacency_matrix_images;
pub mod dot_files;

pub trait StdRendererOutput {
    fn with_directory<P>(directory: P) -> Result<Self, Box<dyn Error>>
    where 
        Self: Sized,
        P: AsRef<Path>;
    fn output_name() -> &'static str;

    fn containing_folder(
        graph_name: &str,
        seed: u64,
        factor: f64
    ) -> PathBuf {
        format!(
            "data/test/graph_{}/seed_{}/factor_{}/",
            graph_name,
            seed,
            factor
        ).into()
    }

    fn from_graph_parameters(
        graph_name: &str,
        seed: u64,
        factor: f64
    ) -> Result<Self, Box<dyn Error>> 
    where 
        Self: Sized
    {
        let containing_folder = Self::containing_folder(graph_name, seed, factor);
        let directory = containing_folder.join(Self::output_name());

        Self::with_directory(directory)
    }
}