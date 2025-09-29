use std::error::Error;
use std::fmt::Display;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};

use crate::knowledge_graph::KnowledgeGraph;
use crate::renderers::StdRendererOutput;

pub struct DotFileRenderer {
    directory: PathBuf,
    count: usize
}

impl StdRendererOutput for DotFileRenderer {
    fn with_directory<P>(directory: P) -> Result<Self, Box<dyn Error>>
    where 
        Self: Sized,
        P: AsRef<Path>
    {
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
    
    fn output_name() -> &'static str {
        "dot_files"
    }
}

impl DotFileRenderer {
    pub fn render<V: Display + Ord>(
        &mut self,
        graph: &KnowledgeGraph<V>
    ) -> Result<(), Box<dyn Error>> {
        let curr_iter = self.count;
        self.count += 1;

        let filename = format!("iter_{}.dot", curr_iter);
        let filepath = self.directory.join(filename);
        
        graph.write_to_dot_file(filepath)
    }
}