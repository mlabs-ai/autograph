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

    /// The label of the predicate with the given ID, if it exists.
    pub fn predicate_label(&self, id: usize) -> Option<&str> {
        self.predicate_labels.get(id).map(String::as_str)
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
    use super::FrameGraph;

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
}
