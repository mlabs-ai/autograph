use crate::knowledge_graph::KnowledgeGraph;
use crate::renderers::StdRendererOutput;

use std::error::Error;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};

use plotters::prelude::*;

pub struct AdjacencyMatrixImageRenderer {
    directory: PathBuf,
    count: usize
}

impl StdRendererOutput for AdjacencyMatrixImageRenderer {
    fn with_directory<P>(directory: P) -> Result<Self, Box<dyn std::error::Error>>
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
        "adj_matrix_images"
    }
}

impl AdjacencyMatrixImageRenderer {
    pub fn render<V: Ord>(
        &mut self,
        width: usize,
        height: usize,
        graph: &KnowledgeGraph<V>
    ) -> Result<(), Box<dyn Error>> {
        let curr_iter = self.count;
        self.count += 1;

        let filename = format!("iter_{}.png", curr_iter);
        let filepath = self.directory.join(filename);

        let adj_mat = graph.as_matrix();

        let root = BitMapBackend::new(
            &filepath,
            (width as u32, height as u32)
        ).into_drawing_area();
        root.fill(&WHITE)?;

        let cells = root.split_evenly((adj_mat.len(), adj_mat.len()));
        for (edge, cell) in adj_mat.into_iter()
            .flatten()
            .zip(cells.into_iter())
        {
            if edge {
                cell.fill(&BLACK)?;
            }
        }

        root.present()?;
        Ok(())
    }
}