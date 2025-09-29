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
        let num_verts = adj_mat.len();

        let root = BitMapBackend::new(
            &filepath,
            (width as u32, height as u32)
        ).into_drawing_area();
        root.fill(&WHITE)?;

        let title = format!("Iteration {}", curr_iter);
        let mut chart = ChartBuilder::on(&root)
            .caption(&title, ("sans-serif", 80))
            .margin(5)
            .top_x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0..num_verts, num_verts..0)?;

        chart
            .configure_mesh()
            .x_labels(20)
            .y_labels(20)
            .max_light_lines(4)
            .x_label_offset(35)
            .y_label_offset(25)
            .disable_x_mesh()
            .disable_y_mesh()
            .label_style(("sans-serif", 20))
            .draw()?;

        chart.draw_series(
            adj_mat
                .into_iter()
                .enumerate()
                .flat_map(|(y, row)| 
                    row
                        .into_iter()
                        .map(|cell| if cell { &BLACK } else { &WHITE })
                        .enumerate()
                        .map(move |(x, color)| (x, y, color))
                )
                .map(|(x, y, color)| Rectangle::new(
                    [(x, y), (x + 1, y + 1)],
                    color.filled()
                ))
        )?;

        root.present()?;
        Ok(())
    }
}