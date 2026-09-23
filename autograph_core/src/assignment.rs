//! Frame-of-reference assignment (Milestone 2).
//!
//! Milestone 1 (`knowledge_graph`) discovers *frames of reference* — the
//! clusters produced by the block-factorization algorithm — from an unlabeled
//! knowledge graph. Milestone 2 assigns **entities** and **predicates** to
//! those frames, treating the two assignments as a collaborative inference
//! problem solved with Expectation-Maximisation.
//!
//! This module provides the multi-relational data layer that Milestone 2 needs
//! and which the existing `KnowledgeGraph` (an unlabeled graph) does not:
//!
//! * [`FrameGraph`] stores entities, predicates, and *typed* edges of the form
//!   `(entity, predicate, entity)`.
//! * [`FrameGraph::union_graph`] collapses typed edges to an unlabeled
//!   [`KnowledgeGraph`], which is what the Milestone-1 clustering operates on.

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use bzip2::read::BzDecoder;
use serde_json::Value;

use crate::knowledge_graph::KnowledgeGraph;

/// The multi-relational predicate set chosen for Milestone 2: the "knowledge
/// production + careers" ontology. Each entry is a Wikidata property ID.
pub const PEOPLE_AND_WORKS_PREDICATES: [&str; 7] = [
    "P106", // occupation
    "P108", // employer
    "P69",  // educated at
    "P185", // doctoral student
    "P101", // field of work
    "P50",  // author
    "P921", // main subject
];

/// A hard, pre-computed assignment of entities to reference frames, used to
/// seed the (soft) Milestone-2 EM assignment without re-clustering.
///
/// Produced either by [`FrameGraph::cluster`] or by loading the JSON output of
/// `python/cluster_wiki.py`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameSeed {
    /// Number of frames (0-based indices from 0 to `num_frames - 1`).
    pub num_frames: usize,
    /// Entity label (QID) → frame index.
    pub entity_to_frame: HashMap<String, usize>,
}

/// Soft memberships produced by the Milestone-2 EM assignment.
///
/// `theta[f][e]` is the degree to which entity `e` belongs to frame `f`, and
/// `phi[f][p]` the degree to which predicate `p` belongs to frame `f`. Both are
/// in `[0, 1]`, and — per `Project.md` — the sum across frames is intentionally
/// *not* constrained (an entity may fully belong to several frames).
#[derive(Debug, Clone, PartialEq)]
pub struct Memberships {
    pub num_frames: usize,
    pub num_entities: usize,
    pub num_predicates: usize,
    /// `theta[frame][entity]` in `[0, 1]`.
    pub theta: Vec<Vec<f64>>,
    /// `phi[frame][predicate]` in `[0, 1]`.
    pub phi: Vec<Vec<f64>>,
}

impl Memberships {
    /// The frame with the largest membership for each entity.
    pub fn argmax_entities(&self) -> Vec<usize> {
        (0..self.num_entities)
            .map(|eid| {
                (0..self.num_frames)
                    .max_by(|&a, &b| {
                        self.theta[a][eid]
                            .partial_cmp(&self.theta[b][eid])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            })
            .collect()
    }

    /// The frame with the largest membership for each predicate.
    pub fn argmax_predicates(&self) -> Vec<usize> {
        (0..self.num_predicates)
            .map(|pid| {
                (0..self.num_frames)
                    .max_by(|&a, &b| {
                        self.phi[a][pid]
                            .partial_cmp(&self.phi[b][pid])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            })
            .collect()
    }

    /// The largest absolute change in any membership between `self` and `other`
    /// (used as the EM convergence criterion).
    pub fn max_delta(&self, other: &Memberships) -> f64 {
        let mut d: f64 = 0.0;
        for f in 0..self.num_frames {
            for e in 0..self.num_entities {
                d = d.max((self.theta[f][e] - other.theta[f][e]).abs());
            }
            for p in 0..self.num_predicates {
                d = d.max((self.phi[f][p] - other.phi[f][p]).abs());
            }
        }
        d
    }
}

/// A knowledge graph with typed edges, the foundation for frame-of-reference
/// assignment.
///
/// Entities and predicates are interned to dense `usize` IDs. Edges are stored
/// as `(src_entity, predicate, dst_entity)` triples over those IDs.
#[derive(Debug, Eq, PartialEq, Clone, Default)]
pub struct FrameGraph {
    entity_labels: Vec<String>,
    entity_ids: HashMap<String, usize>,
    predicate_labels: Vec<String>,
    predicate_ids: HashMap<String, usize>,
    edges: Vec<(usize, usize, usize)>,
}

impl FrameGraph {
    /// Creates an empty `FrameGraph`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Interns an entity, returning its dense ID. Adding an already-present
    /// entity returns its existing ID and does not create a duplicate.
    pub fn add_entity(&mut self, label: &str) -> usize {
        if let Some(&id) = self.entity_ids.get(label) {
            id
        } else {
            let id = self.entity_labels.len();
            self.entity_labels.push(label.to_string());
            self.entity_ids.insert(label.to_string(), id);
            id
        }
    }

    /// Interns a predicate, returning its dense ID. Like [`Self::add_entity`],
    /// this deduplicates by label.
    pub fn add_predicate(&mut self, label: &str) -> usize {
        if let Some(&id) = self.predicate_ids.get(label) {
            id
        } else {
            let id = self.predicate_labels.len();
            self.predicate_labels.push(label.to_string());
            self.predicate_ids.insert(label.to_string(), id);
            id
        }
    }

    /// Adds a typed edge by ID, panicking-free: returns an error if any of the
    /// entity or predicate IDs is out of range.
    pub fn add_edge_by_id(
        &mut self,
        src: usize,
        predicate: usize,
        dst: usize,
    ) -> Result<(), Box<dyn Error>> {
        if src >= self.entity_labels.len() {
            return Err(format!("unknown entity id: {src}").into());
        }
        if dst >= self.entity_labels.len() {
            return Err(format!("unknown entity id: {dst}").into());
        }
        if predicate >= self.predicate_labels.len() {
            return Err(format!("unknown predicate id: {predicate}").into());
        }
        self.edges.push((src, predicate, dst));
        Ok(())
    }

    /// Adds a typed edge by label, interning any entity or predicate that is
    /// not already present.
    pub fn add_edge(
        &mut self,
        src: &str,
        predicate: &str,
        dst: &str,
    ) -> Result<(), Box<dyn Error>> {
        let src_id = self.add_entity(src);
        let dst_id = self.add_entity(dst);
        let pred_id = self.add_predicate(predicate);
        self.add_edge_by_id(src_id, pred_id, dst_id)
    }

    /// Number of distinct entities.
    pub fn num_entities(&self) -> usize {
        self.entity_labels.len()
    }

    /// Number of distinct predicates.
    pub fn num_predicates(&self) -> usize {
        self.predicate_labels.len()
    }

    /// Number of typed edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// The label of the entity with the given ID, if it exists.
    pub fn entity_label(&self, id: usize) -> Option<&str> {
        self.entity_labels.get(id).map(String::as_str)
    }

    /// The ID of the entity with the given label, if it exists.
    pub fn entity_id(&self, label: &str) -> Option<usize> {
        self.entity_ids.get(label).copied()
    }

    /// The label of the predicate with the given ID, if it exists.
    pub fn predicate_label(&self, id: usize) -> Option<&str> {
        self.predicate_labels.get(id).map(String::as_str)
    }

    /// The ID of the predicate with the given label, if it exists.
    pub fn predicate_id(&self, label: &str) -> Option<usize> {
        self.predicate_ids.get(label).copied()
    }

    /// All entity labels (QIDs) in ID order.
    pub fn all_entity_labels(&self) -> &[String] {
        &self.entity_labels
    }

    /// All predicate labels (property IDs) in ID order.
    pub fn all_predicate_labels(&self) -> &[String] {
        &self.predicate_labels
    }

    /// The typed edges as `(src_entity, predicate, dst_entity)` ID triples.
    pub fn edges(&self) -> &[(usize, usize, usize)] {
        &self.edges
    }

    /// Collapses typed edges to an unlabeled `KnowledgeGraph` over the same
    /// entities, dropping predicate identity. This is the graph the
    /// Milestone-1 clustering runs on to discover frames of reference.
    pub fn union_graph(&self) -> KnowledgeGraph<String> {
        let mut graph = KnowledgeGraph::new();
        for label in &self.entity_labels {
            graph.add_vertex(label.clone());
        }
        for &(src, _pred, dst) in &self.edges {
            graph.add_edge(
                self.entity_labels[src].clone(),
                self.entity_labels[dst].clone(),
            );
        }
        graph
    }

    /// Discovers reference frames by clustering the union graph (all predicates
    /// collapsed) with the Milestone-1 block-factorization algorithm, returning
    /// the frames as sets of entity labels.
    pub fn cluster(
        &self,
        factor: f64,
        steps_before_subdivide: usize,
        boundary_threshold: f64,
        min_cluster_size: usize,
    ) -> Vec<Vec<String>> {
        let mut union = self.union_graph();
        union.cluster(
            factor,
            steps_before_subdivide,
            boundary_threshold,
            min_cluster_size,
        );
        union.get_clusters()
    }

    /// Parses a list of reference frames from JSON, exactly as written by
    /// `python/cluster_wiki.py` (a JSON array of arrays of entity labels).
    pub fn frames_from_json(json: &str) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
        let frames: Vec<Vec<String>> = serde_json::from_str(json)?;
        Ok(frames)
    }

    /// Builds a hard entity → frame seed from pre-computed frames, without
    /// re-clustering. Fails if any entity is assigned to more than one frame.
    pub fn seed_from_frames(&self, frames: &[Vec<String>]) -> Result<FrameSeed, Box<dyn Error>> {
        let num_frames = frames.len();
        let mut entity_to_frame = HashMap::new();
        for (frame_id, frame) in frames.iter().enumerate() {
            for entity in frame {
                if entity_to_frame.contains_key(entity) {
                    return Err(format!("entity {entity} appears in more than one frame").into());
                }
                entity_to_frame.insert(entity.clone(), frame_id);
            }
        }
        Ok(FrameSeed {
            num_frames,
            entity_to_frame,
        })
    }

    /// Initialises memberships from a hard frame seed: an entity's own frame
    /// gets `theta = 1.0`, all other frames `epsilon`; predicates start uniform
    /// (`1 / num_frames`). Entities absent from the seed start at `epsilon` in
    /// every frame.
    pub fn init_memberships(&self, seed: &FrameSeed, epsilon: f64) -> Memberships {
        let nf = seed.num_frames;
        let ne = self.num_entities();
        let np = self.num_predicates();

        let mut theta = vec![vec![epsilon; ne]; nf];
        for (label, &frame) in &seed.entity_to_frame {
            if frame >= nf {
                continue;
            }
            if let Some(&eid) = self.entity_ids.get(label) {
                theta[frame][eid] = 1.0;
            }
        }

        let phi = vec![vec![1.0 / nf as f64; np]; nf];

        Memberships {
            num_frames: nf,
            num_entities: ne,
            num_predicates: np,
            theta,
            phi,
        }
    }

    /// Performs one EM iteration: an E-step that assigns each typed edge a
    /// responsibility over frames, and an M-step that re-estimates memberships
    /// from those responsibilities and saturates them at 1.
    ///
    /// Membership update rule (currently): `theta[f][e] = min(1, raw)` where
    /// `raw` is the sum of that frame's responsibilities over edges incident on
    /// `e`. This keeps each entry in `[0, 1]` and does *not* normalize the sum
    /// across frames, honouring `Project.md`'s "≤ 1, sum unconstrained"
    /// semantics.
    pub fn em_step(&self, m: &Memberships) -> Memberships {
        let nf = m.num_frames;
        let ne = m.num_entities;
        let np = m.num_predicates;

        let mut theta_raw = vec![vec![0.0f64; ne]; nf];
        let mut phi_raw = vec![vec![0.0f64; np]; nf];

        for &(u, pred, v) in &self.edges {
            // E-step: score each frame, then normalise to a responsibility.
            let mut scores = Vec::with_capacity(nf);
            let mut total = 0.0f64;
            for f in 0..nf {
                let s = m.theta[f][u] * m.theta[f][v] * m.phi[f][pred];
                scores.push(s);
                total += s;
            }
            if total <= 0.0 {
                continue;
            }

            // M-step accumulation: distribute the responsibility to the two
            // endpoint entities and the predicate.
            for f in 0..nf {
                let gamma = scores[f] / total;
                theta_raw[f][u] += gamma;
                theta_raw[f][v] += gamma;
                phi_raw[f][pred] += gamma;
            }
        }

        let mut theta = vec![vec![0.0f64; ne]; nf];
        for f in 0..nf {
            for e in 0..ne {
                theta[f][e] = theta_raw[f][e].min(1.0);
            }
        }
        let mut phi = vec![vec![0.0f64; np]; nf];
        for f in 0..nf {
            for p in 0..np {
                phi[f][p] = phi_raw[f][p].min(1.0);
            }
        }

        Memberships {
            num_frames: nf,
            num_entities: ne,
            num_predicates: np,
            theta,
            phi,
        }
    }

    /// Runs the EM assignment to convergence (or the iteration budget) and
    /// returns the soft memberships.
    pub fn em_assign(
        &self,
        seed: &FrameSeed,
        epsilon: f64,
        tol: f64,
        max_iters: usize,
    ) -> Memberships {
        let mut m = self.init_memberships(seed, epsilon);
        for _ in 0..max_iters {
            let next = self.em_step(&m);
            let delta = m.max_delta(&next);
            m = next;
            if delta < tol {
                break;
            }
        }
        m
    }

    /// Builds a `FrameGraph` from a Wikidata JSON dump, ingesting **multiple**
    /// predicates as typed edges.
    ///
    /// Entities are keyed by their Wikidata QID (e.g. `"Q42"`), predicates by
    /// their property ID (e.g. `"P106"`). For each entity and each predicate in
    /// `predicates`, every `(entity --predicate--> target)` claim becomes a
    /// typed edge. Only `wikibase-entityid` targets are kept (literal values
    /// such as dates and strings are skipped).
    pub fn from_wikidata_multi<R>(
        reader: R,
        predicates: &[&str],
        limit: Option<usize>,
    ) -> Result<Self, Box<dyn Error>>
    where
        R: BufRead,
    {
        let limit = limit.unwrap_or(usize::MAX);
        let mut graph = FrameGraph::new();
        let mut count = 0usize;

        for line in reader.lines() {
            let line = line?;

            // Each entity is a single JSON object (possibly wrapped in a larger
            // array); isolate the `{ ... }` portion.
            let Some(start) = line.find('{') else {
                continue;
            };
            let Some(end) = line.rfind('}') else { continue };
            if start > end {
                continue;
            }
            let json: Value = match serde_json::from_str(&line[start..=end]) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let Some(src) = json.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            let Some(claims) = json.get("claims").and_then(|v| v.as_object()) else {
                continue;
            };

            for &pred in predicates {
                let Some(stmts) = claims.get(pred).and_then(|v| v.as_array()) else {
                    continue;
                };
                for stmt in stmts {
                    let dst = stmt
                        .get("mainsnak")
                        .and_then(|m| m.get("datavalue"))
                        .and_then(|d| d.get("value"))
                        .and_then(|v| v.get("id"))
                        .and_then(|i| i.as_str());
                    if let Some(dst) = dst {
                        graph.add_edge(src, pred, dst)?;
                    }
                }
            }

            count += 1;
            if count >= limit {
                break;
            }
        }

        Ok(graph)
    }

    /// Reads the full (or `limit`-capped) Wikidata dump from a bzip2-compressed
    /// stream, ingesting the given predicates as typed edges.
    pub fn from_wikidata_multi_bz2<P>(
        path: P,
        predicates: &[&str],
        limit: Option<usize>,
    ) -> Result<Self, Box<dyn Error>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(path)?;
        let decoder = BzDecoder::new(file);
        let reader = BufReader::new(decoder);
        Self::from_wikidata_multi(reader, predicates, limit)
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameGraph, FrameSeed};

    fn two_clique_graph() -> FrameGraph {
        let mut g = FrameGraph::new();
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    g.add_edge(&format!("E{i}"), "P", &format!("E{j}")).unwrap();
                }
            }
        }
        for i in 4..16 {
            for j in 4..16 {
                if i != j {
                    g.add_edge(&format!("E{i}"), "P", &format!("E{j}")).unwrap();
                }
            }
        }
        g
    }

    #[test]
    fn empty() {
        let g = FrameGraph::new();
        assert_eq!(g.num_entities(), 0);
        assert_eq!(g.num_predicates(), 0);
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn entity_dedup() {
        let mut g = FrameGraph::new();
        let a = g.add_entity("Q1");
        let b = g.add_entity("Q1");
        let c = g.add_entity("Q2");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(g.num_entities(), 2);
    }

    #[test]
    fn predicate_dedup() {
        let mut g = FrameGraph::new();
        let p1 = g.add_predicate("P31");
        let p2 = g.add_predicate("P31");
        let p3 = g.add_predicate("P279");
        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
        assert_eq!(g.num_predicates(), 2);
    }

    #[test]
    fn add_edge_by_id_rejects_unknown_ids() {
        let mut g = FrameGraph::new();
        let e = g.add_entity("Q1");
        let p = g.add_predicate("P31");

        // Unknown dst entity.
        assert!(g.add_edge_by_id(e, p, 99).is_err());
        // Unknown predicate.
        assert!(g.add_edge_by_id(e, 99, e).is_err());
        // Unknown src entity.
        assert!(g.add_edge_by_id(99, p, e).is_err());

        // Valid edge succeeds.
        assert!(g.add_edge_by_id(e, p, e).is_ok());
        assert_eq!(g.num_edges(), 1);
    }

    #[test]
    fn add_edge_auto_interns() {
        let mut g = FrameGraph::new();
        g.add_edge("Q1", "P31", "Q2").unwrap();
        g.add_edge("Q2", "P279", "Q3").unwrap();

        assert_eq!(g.num_entities(), 3);
        assert_eq!(g.num_predicates(), 2);
        assert_eq!(g.num_edges(), 2);
    }

    #[test]
    fn union_graph_preserves_vertices_and_edges() {
        let mut g = FrameGraph::new();
        g.add_edge("Q1", "P31", "Q2").unwrap();
        g.add_edge("Q2", "P31", "Q3").unwrap();
        // An isolated entity that must still appear in the union graph.
        g.add_entity("Q9");

        let union = g.union_graph();

        // All entities are present as vertices.
        assert_eq!(union.num_vertices(), 4);
        // Two typed edges collapse to two unlabeled edges (predicate dropped).
        assert_eq!(union.num_edges(), 2);
    }

    /// A small committed Wikidata fixture: people with occupations/fields/
    /// education, works with authors/subjects, and an out-of-scope `instance
    /// of` (P31) claim that must be ignored.
    const WIKI_FIXTURE: &str = concat!(
        r#"{"type":"item","id":"Q1","claims":{"P106":[{"mainsnak":{"datavalue":{"value":{"id":"Q10"}}}}],"P101":[{"mainsnak":{"datavalue":{"value":{"id":"Q20"}}}}],"P69":[{"mainsnak":{"datavalue":{"value":{"id":"Q30"}}}}],"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q5"}}}}]}}"#,
        "\n",
        r#"{"type":"item","id":"Q2","claims":{"P106":[{"mainsnak":{"datavalue":{"value":{"id":"Q11"}}}}],"P108":[{"mainsnak":{"datavalue":{"value":{"id":"Q40"}}}}],"P185":[{"mainsnak":{"datavalue":{"value":{"id":"Q50"}}}}]}}"#,
        "\n",
        r#"{"type":"item","id":"Q3","claims":{"P50":[{"mainsnak":{"datavalue":{"value":{"id":"Q1"}}}}],"P921":[{"mainsnak":{"datavalue":{"value":{"id":"Q20"}}}}]}}"#,
        "\n",
        r#"{"type":"item","id":"Q4","claims":{"P50":[{"mainsnak":{"datavalue":{"value":{"id":"Q2"}}}}]}}"#
    );

    #[test]
    fn from_wikidata_multi_ingests_typed_edges() {
        use std::io::Cursor;

        let predicates = ["P106", "P108", "P69", "P185", "P101", "P50", "P921"];
        let graph = FrameGraph::from_wikidata_multi(
            Cursor::new(WIKI_FIXTURE.as_bytes()),
            &predicates,
            None,
        )
        .unwrap();

        // Entities: Q1..Q4 (sources) + Q10, Q11, Q20, Q30, Q40, Q50 (targets).
        // Note Q5 (instance-of target) is NOT ingested because P31 is excluded.
        assert_eq!(graph.num_entities(), 10);
        assert_eq!(graph.num_predicates(), 7);
        assert_eq!(graph.num_edges(), 9);
    }

    #[test]
    fn from_wikidata_multi_ignores_non_whitelisted_predicates() {
        use std::io::Cursor;

        // Only P106 is whitelisted; P101/P31/etc. must be ignored.
        let graph =
            FrameGraph::from_wikidata_multi(Cursor::new(WIKI_FIXTURE.as_bytes()), &["P106"], None)
                .unwrap();

        // Two P106 edges (Q1->Q10, Q2->Q11); no other predicate ingested.
        assert_eq!(graph.num_edges(), 2);
        assert_eq!(graph.num_predicates(), 1);
        assert_eq!(graph.num_entities(), 4); // Q1, Q2, Q10, Q11
    }

    #[test]
    fn from_wikidata_multi_respects_limit() {
        use std::io::Cursor;

        let predicates = ["P106", "P108", "P69", "P185", "P101", "P50", "P921"];
        let graph = FrameGraph::from_wikidata_multi(
            Cursor::new(WIKI_FIXTURE.as_bytes()),
            &predicates,
            Some(2),
        )
        .unwrap();

        // Only the first two entities (Q1, Q2) are read.
        // Entities: Q1, Q2, Q10, Q20, Q30, Q11, Q40, Q50 = 8.
        assert_eq!(graph.num_entities(), 8);
    }

    #[test]
    fn cluster_separates_two_disconnected_cliques() {
        let mut g = FrameGraph::new();
        // Clique 1: entities E0..E3 (4 nodes), fully connected.
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    g.add_edge(&format!("E{i}"), "P", &format!("E{j}")).unwrap();
                }
            }
        }
        // Clique 2: entities E4..E15 (12 nodes), fully connected.
        for i in 4..16 {
            for j in 4..16 {
                if i != j {
                    g.add_edge(&format!("E{i}"), "P", &format!("E{j}")).unwrap();
                }
            }
        }

        let frames = g.cluster(0.01, 3, 0.1, 2);

        assert_eq!(frames.len(), 2, "expected two frames, got {frames:?}");
        let mut sizes: Vec<usize> = frames.iter().map(Vec::len).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![4, 12]);
    }

    #[test]
    fn frames_round_trip_through_json() {
        let g = two_clique_graph();
        let frames = g.cluster(0.01, 3, 0.1, 2);

        // Serialize like cluster_wiki.py, then parse back.
        let json = serde_json::to_string(&frames).unwrap();
        let parsed = FrameGraph::frames_from_json(&json).unwrap();

        assert_eq!(parsed, frames);
    }

    #[test]
    fn seed_from_frames_maps_entities() {
        let g = two_clique_graph();
        let frames = g.cluster(0.01, 3, 0.1, 2);
        let seed = g.seed_from_frames(&frames).unwrap();

        assert_eq!(seed.num_frames, 2);
        // Every entity is assigned to exactly one frame.
        assert_eq!(seed.entity_to_frame.len(), 16);
        // The two cliques land in different frames.
        assert_ne!(seed.entity_to_frame["E0"], seed.entity_to_frame["E4"]);
    }

    #[test]
    fn seed_from_frames_rejects_duplicate_entity() {
        let g = FrameGraph::new();
        let frames = vec![
            vec!["E0".to_string()],
            vec!["E0".to_string(), "E1".to_string()],
        ];
        assert!(g.seed_from_frames(&frames).is_err());
    }

    /// Two disjoint "regimes" plus a bridge entity:
    ///   frame 0 = A0..A9 fully connected via predicate PA,
    ///   frame 1 = B0..B9 fully connected via predicate PB,
    ///   bridge X connects to A0 (PA) and B0 (PB).
    /// The seed assigns A* -> 0 and B* -> 1 (X is unseeded).
    fn two_regime_graph_and_seed() -> (FrameGraph, FrameSeed) {
        use std::collections::HashMap;

        let mut g = FrameGraph::new();
        for i in 0..10 {
            for j in 0..10 {
                if i != j {
                    g.add_edge(&format!("A{i}"), "PA", &format!("A{j}"))
                        .unwrap();
                    g.add_edge(&format!("B{i}"), "PB", &format!("B{j}"))
                        .unwrap();
                }
            }
        }
        g.add_edge("X", "PA", "A0").unwrap();
        g.add_edge("X", "PB", "B0").unwrap();

        let mut entity_to_frame = HashMap::new();
        for i in 0..10 {
            entity_to_frame.insert(format!("A{i}"), 0);
            entity_to_frame.insert(format!("B{i}"), 1);
        }
        let seed = FrameSeed {
            num_frames: 2,
            entity_to_frame,
        };
        (g, seed)
    }

    #[test]
    fn init_memberships_seeds_theta() {
        let (g, seed) = two_regime_graph_and_seed();
        let m = g.init_memberships(&seed, 1e-3);

        let a0 = g.entity_id("A0").unwrap();
        let b0 = g.entity_id("B0").unwrap();
        let x = g.entity_id("X").unwrap();

        // Seeded entity: own frame = 1, other frame = epsilon.
        assert_eq!(m.theta[0][a0], 1.0);
        assert_eq!(m.theta[1][a0], 1e-3);
        assert_eq!(m.theta[1][b0], 1.0);
        assert_eq!(m.theta[0][b0], 1e-3);
        // Unseeded entity: epsilon everywhere.
        assert_eq!(m.theta[0][x], 1e-3);
        assert_eq!(m.theta[1][x], 1e-3);
        // Predicates start uniform.
        let pa = g.predicate_id("PA").unwrap();
        assert_eq!(m.phi[0][pa], 0.5);
        assert_eq!(m.phi[1][pa], 0.5);
    }

    #[test]
    fn em_keeps_memberships_within_unit_interval() {
        let (g, seed) = two_regime_graph_and_seed();
        let m = g.em_assign(&seed, 1e-3, 1e-9, 50);

        for f in 0..m.num_frames {
            for e in 0..m.num_entities {
                let t = m.theta[f][e];
                assert!(
                    (0.0..=1.0).contains(&t),
                    "theta[{f}][{e}] = {t} out of [0,1]"
                );
            }
            for p in 0..m.num_predicates {
                let phi = m.phi[f][p];
                assert!(
                    (0.0..=1.0).contains(&phi),
                    "phi[{f}][{p}] = {phi} out of [0,1]"
                );
            }
        }
    }

    #[test]
    fn em_is_deterministic() {
        let (g, seed) = two_regime_graph_and_seed();
        let m1 = g.em_assign(&seed, 1e-3, 1e-9, 50);
        let m2 = g.em_assign(&seed, 1e-3, 1e-9, 50);
        assert_eq!(m1, m2);
    }

    #[test]
    fn em_recovers_ground_truth_frames() {
        let (g, seed) = two_regime_graph_and_seed();
        let m = g.em_assign(&seed, 1e-3, 1e-9, 50);
        let argmax = m.argmax_entities();

        for i in 0..10 {
            let a = g.entity_id(&format!("A{i}")).unwrap();
            let b = g.entity_id(&format!("B{i}")).unwrap();
            assert_eq!(argmax[a], 0, "entity A{i} should be in frame 0");
            assert_eq!(argmax[b], 1, "entity B{i} should be in frame 1");
        }
    }

    #[test]
    fn em_concentrates_predicates() {
        let (g, seed) = two_regime_graph_and_seed();
        let m = g.em_assign(&seed, 1e-3, 1e-9, 50);
        let argmax = m.argmax_predicates();

        let pa = g.predicate_id("PA").unwrap();
        let pb = g.predicate_id("PB").unwrap();
        assert_eq!(argmax[pa], 0, "predicate PA should be frame 0");
        assert_eq!(argmax[pb], 1, "predicate PB should be frame 1");
    }

    #[test]
    fn bridge_entity_belongs_to_multiple_frames() {
        let (g, seed) = two_regime_graph_and_seed();
        let m = g.em_assign(&seed, 1e-3, 1e-9, 50);
        let x = g.entity_id("X").unwrap();

        // X bridges both regimes, so its membership must be non-trivial in both
        // frames, and the sum is allowed to exceed 1 (no sum normalisation).
        assert!(m.theta[0][x] > 0.5, "X should belong to frame 0");
        assert!(m.theta[1][x] > 0.5, "X should belong to frame 1");
        assert!(
            m.theta[0][x] + m.theta[1][x] > 1.0,
            "sum of X's memberships should exceed 1, got {:.3}",
            m.theta[0][x] + m.theta[1][x]
        );
    }
}
