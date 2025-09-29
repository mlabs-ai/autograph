use std::error::Error;
use std::fmt::Display;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};

use crate::knowledge_graph::KnowledgeGraph;

pub struct DotFileRenderer {
    directory: PathBuf,
    count: usize
}

impl DotFileRenderer {
    pub fn new<P: AsRef<Path>>(directory: P) -> Result<Self, Box<dyn Error>> {
        // Make sure directory exists
        let directory = directory.as_ref().to_path_buf();
        if !directory.is_dir() {
            create_dir_all(&directory)?;
        }

        Ok(
            Self {
                directory,
                count: 0
            }
        )
    }

    pub fn new_from_default(
        graph_name: &str,
        seed: u64,
        factor: f64
    ) -> Result<Self, Box<dyn Error>> {
        let directory = format!(
            "data/test/graph_{}/seed_{}/factor_{}/dot_files/",
            graph_name,
            seed,
            factor
        );

        Self::new(directory)
    }

    pub fn render<V: Display + Ord>(
        &mut self, 
        graph: &KnowledgeGraph<V>
    ) -> Result<(), Box<dyn Error>> {
        let curr_iter = self.count;

        let filename = format!("iter_{}.dot", curr_iter);
        let filepath = self.directory.join(filename);
        graph.write_to_dot_file(filepath)?;

        self.count += 1;
        Ok(())
    }
}