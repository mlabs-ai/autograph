//! Semantic divisive clustering (Milestone 2).
//!
//! A faithful Rust reimplementation of `python/semantic_clustering.py::Clusterer`
//! (the 8 core edge predicates + P31/P106 qualities; the same greedy
//! dominant-quality / dominant-edge evaluation). Kept in a standalone module so
//! the `semantic_cluster` binary can drive it end-to-end and the results can be
//! compared 1:1 against the Python output.
//!
//! The recursion's split groups are mutually independent, so `solve` recurses
//! over them in parallel (`rayon::par_iter`). Mutable counting state is held in
//! *thread-local* scratch (generation-tagged arrays) so each worker counts
//! without synchronisation, and each node returns its clusters as owned values.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};

use rayon::prelude::*;

/// Tunable knobs, mirroring the `semantic_clustering.py` CLI defaults.
#[derive(Debug, Clone)]
pub struct Config {
    pub min_size: usize,
    pub max_depth: usize,
    pub fanout: usize,
    pub coverage_target: f64,
    pub min_lift: f64,
    pub min_fraction: f64,
    pub max_df_frac: f64,
    pub max_picks: usize,
    pub max_steps_cluster: usize,
    /// If true, qualities (P31/P106) are excluded from the split & stop logic and
    /// used only to *evaluate* the resulting clusters (non-circular comparison).
    pub edges_only: bool,
    /// Nodes at least this many members recurse over their children in parallel.
    pub parallel_min: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            min_size: 20,
            max_depth: 40,
            fanout: 200,
            coverage_target: 0.6,
            min_lift: 2.0,
            min_fraction: 0.02,
            max_df_frac: 0.1,
            max_picks: 10,
            max_steps_cluster: 100_000,
            edges_only: false,
            parallel_min: 20_000,
        }
    }
}

/// One dominant quality or dominant edge, as reported per cluster.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Dominant {
    pub predicate: String,
    pub value: String,
    pub count: u32,
    pub fraction: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lift: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ClusterOut {
    pub size: usize,
    pub num_qualities: usize,
    pub quality_coverage: f64,
    pub dominant_qualities: Vec<Dominant>,
    pub num_edges: usize,
    pub edge_coverage: f64,
    pub dominant_edges: Vec<Dominant>,
    pub truncated: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SemanticResult {
    pub num_clusters: usize,
    pub num_entities: usize,
    pub clusters: Vec<ClusterOut>,
}

/// The interned, immutable semantic model built during ingestion.
pub struct SemanticModel {
    pub num_entities: usize,
    // Per-entity (indexed 0..num_entities): interned key ids.
    qkeys: Vec<Vec<u32>>,
    ekeys: Vec<Vec<u32>>,
    // Per-entity: interned *value* ids of the edges (for the "steps" BFS).
    evalues: Vec<Vec<u32>>,
    // Global document-frequency per key id.
    df: Vec<u32>,
    // key id -> (pred id, value id).
    key_pred: Vec<u32>,
    key_value: Vec<u32>,
    // Interner decode tables.
    pred_str: Vec<String>,
    value_str: Vec<String>,
}

/// Incremental builder used by the ingestion loop.
pub struct ModelBuilder {
    pred_str: Vec<String>,
    pred_id: HashMap<String, u32>,
    value_str: Vec<String>,
    value_id: HashMap<String, u32>,
    key_map: HashMap<u64, u32>,
    key_pred: Vec<u32>,
    key_value: Vec<u32>,
    df: Vec<u32>,
    qkeys: Vec<Vec<u32>>,
    ekeys: Vec<Vec<u32>>,
    evalues: Vec<Vec<u32>>,
}

impl ModelBuilder {
    pub fn new() -> Self {
        ModelBuilder {
            pred_str: Vec::new(),
            pred_id: HashMap::new(),
            value_str: Vec::new(),
            value_id: HashMap::new(),
            key_map: HashMap::new(),
            key_pred: Vec::new(),
            key_value: Vec::new(),
            df: Vec::new(),
            qkeys: Vec::new(),
            ekeys: Vec::new(),
            evalues: Vec::new(),
        }
    }

    fn pred(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.pred_id.get(s) {
            return id;
        }
        let id = self.pred_str.len() as u32;
        self.pred_str.push(s.to_string());
        self.pred_id.insert(s.to_string(), id);
        id
    }

    fn value(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.value_id.get(s) {
            return id;
        }
        let id = self.value_str.len() as u32;
        self.value_str.push(s.to_string());
        self.value_id.insert(s.to_string(), id);
        id
    }

    fn key(&mut self, p: u32, v: u32) -> u32 {
        let packed = ((p as u64) << 32) | (v as u64);
        if let Some(&id) = self.key_map.get(&packed) {
            return id;
        }
        let id = self.key_pred.len() as u32;
        self.key_map.insert(packed, id);
        self.key_pred.push(p);
        self.key_value.push(v);
        self.df.push(0);
        id
    }

    /// Register one entity. `qualities` / `edges` are raw `(predicate, value)`
    /// string pairs (duplicates filtered here, matching the Python dedupe).
    pub fn add_entity(&mut self, qualities: &[(&str, &str)], edges: &[(&str, &str)]) {
        // Intern + collect quality key ids (deduped).
        let mut qs: Vec<u32> = qualities
            .iter()
            .map(|&(p, v)| {
                let pi = self.pred(p);
                let vi = self.value(v);
                self.key(pi, vi)
            })
            .collect();
        qs.sort_unstable();
        qs.dedup();

        // Intern + collect edge key ids (deduped) and edge value ids (deduped).
        let mut es: Vec<u32> = Vec::with_capacity(edges.len());
        let mut evs: Vec<u32> = Vec::with_capacity(edges.len());
        for &(p, v) in edges {
            let pi = self.pred(p);
            let vi = self.value(v);
            es.push(self.key(pi, vi));
            evs.push(vi);
        }
        es.sort_unstable();
        es.dedup();
        evs.sort_unstable();
        evs.dedup();

        for &k in &qs {
            self.df[k as usize] += 1;
        }
        for &k in &es {
            self.df[k as usize] += 1;
        }

        self.qkeys.push(qs);
        self.ekeys.push(es);
        self.evalues.push(evs);
    }

    pub fn finish(self) -> SemanticModel {
        SemanticModel {
            num_entities: self.qkeys.len(),
            qkeys: self.qkeys,
            ekeys: self.ekeys,
            evalues: self.evalues,
            df: self.df,
            key_pred: self.key_pred,
            key_value: self.key_value,
            pred_str: self.pred_str,
            value_str: self.value_str,
        }
    }
}

fn idf(dfv: f64, n_total: usize) -> f64 {
    ((n_total as f64 + 1.0) / (dfv + 1.0)).ln()
}

fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}
fn round2(x: f64) -> f64 {
    (x * 1e2).round() / 1e2
}

/// Per-thread counting scratch: generation-tagged arrays so `build_counts` does
/// O(keys-in-node) work and never rescans or zeroes the whole key space.
struct Scratch {
    counts: Vec<u32>,
    seen: Vec<u32>,
    generation: u32,
}

thread_local! {
    static SCRATCH: RefCell<Option<Scratch>> = const { RefCell::new(None) };
}

fn with_scratch<R>(num_keys: usize, f: impl FnOnce(&mut Scratch) -> R) -> R {
    SCRATCH.with(|slot| {
        let mut b = slot.borrow_mut();
        let needs_init = b.as_ref().is_none_or(|s| s.counts.len() != num_keys);
        if needs_init {
            *b = Some(Scratch {
                counts: vec![0u32; num_keys],
                seen: vec![0u32; num_keys],
                generation: 0,
            });
        }
        f(b.as_mut().unwrap())
    })
}

/// A ranked dominant tag: `(key_id, count, fraction, lift)`.
type Row = (u32, u32, f64, Option<f64>);

/// The outcome of processing a node before recursion.
enum Step {
    Emit(Vec<Row>, Vec<Row>, f64, f64, bool),
    Split(Vec<Vec<u32>>),
}

struct Solver<'a> {
    model: &'a SemanticModel,
    cfg: &'a Config,
}

impl<'a> Solver<'a> {
    fn new(model: &'a SemanticModel, cfg: &'a Config) -> Self {
        Solver { model, cfg }
    }

    fn solve(&self, members: &[u32], depth: usize) -> Vec<ClusterOut> {
        let n = members.len();
        if n <= self.cfg.min_size {
            return vec![self.make_cluster(members, &[], &[], 0.0, 0.0, false)];
        }

        let step = with_scratch(self.model.df.len(), |scratch| {
            let (q_touched, e_touched) = self.build_counts(members, scratch);
            let qual_rows = self.top_keys(scratch, &q_touched, n);
            let edge_rows = self.top_keys(scratch, &e_touched, n);
            let qual_cov = self.union_coverage(members, &qual_rows, true);
            let edge_cov = self.union_coverage(members, &edge_rows, false);

            let q_ok = !self.cfg.edges_only
                && qual_cov >= self.cfg.coverage_target
                && self.specific(&qual_rows);
            let e_ok = edge_cov >= self.cfg.coverage_target && self.specific(&edge_rows);

            if q_ok || e_ok {
                return Step::Emit(qual_rows, edge_rows, qual_cov, edge_cov, false);
            }
            if depth >= self.cfg.max_depth {
                return Step::Emit(qual_rows, edge_rows, qual_cov, edge_cov, true);
            }
            let split_keys = self.pick_split_keys(scratch, &q_touched, &e_touched, n);
            if split_keys.is_empty() {
                return Step::Emit(qual_rows, edge_rows, qual_cov, edge_cov, true);
            }
            Step::Split(self.assign_groups(members, &split_keys))
        });

        match step {
            Step::Emit(qr, er, qcov, ecov, trunc) => {
                vec![self.make_cluster(members, &qr, &er, qcov, ecov, trunc)]
            }
            Step::Split(groups) => {
                if n < self.cfg.parallel_min || groups.len() <= 1 {
                    let mut out = Vec::new();
                    for g in &groups {
                        out.extend(self.solve(g, depth + 1));
                    }
                    out
                } else {
                    groups
                        .par_iter()
                        .flat_map_iter(|g| self.solve(g, depth + 1))
                        .collect()
                }
            }
        }
    }

    /// Accumulate counts + touched-key lists; returns (quality-keys, edge-keys).
    fn build_counts(&self, members: &[u32], scratch: &mut Scratch) -> (Vec<u32>, Vec<u32>) {
        scratch.generation += 1;
        let cur = scratch.generation;
        let mut q_touched: Vec<u32> = Vec::new();
        let mut e_touched: Vec<u32> = Vec::new();
        for &m in members {
            for &k in &self.model.qkeys[m as usize] {
                if scratch.seen[k as usize] != cur {
                    scratch.seen[k as usize] = cur;
                    scratch.counts[k as usize] = 0;
                    q_touched.push(k);
                }
                scratch.counts[k as usize] += 1;
            }
            for &k in &self.model.ekeys[m as usize] {
                if scratch.seen[k as usize] != cur {
                    scratch.seen[k as usize] = cur;
                    scratch.counts[k as usize] = 0;
                    e_touched.push(k);
                }
                scratch.counts[k as usize] += 1;
            }
        }
        (q_touched, e_touched)
    }

    fn top_keys(&self, scratch: &Scratch, touched: &[u32], n: usize) -> Vec<Row> {
        let mut scored: Vec<(f64, u32, u32, f64, Option<f64>)> = Vec::with_capacity(touched.len());
        let nf = n as f64;
        for &k in touched {
            let c = scratch.counts[k as usize];
            let frac = c as f64 / nf;
            if frac < self.cfg.min_fraction {
                continue;
            }
            let dfv = self.model.df[k as usize] as f64;
            let idfv = idf(dfv, self.model.num_entities);
            let lift = if dfv > 0.0 {
                Some(frac / (dfv / self.model.num_entities as f64))
            } else {
                None
            };
            scored.push((c as f64 * idfv, k, c, frac, lift));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let mut kept: Vec<Row> = Vec::new();
        for (_score, k, c, frac, lift) in scored {
            if kept.len() >= self.cfg.max_picks {
                break;
            }
            if let Some(l) = lift {
                if l < self.cfg.min_lift {
                    continue;
                }
            }
            kept.push((k, c, frac, lift));
        }
        kept
    }

    fn union_coverage(&self, members: &[u32], kept: &[Row], qualities: bool) -> f64 {
        if kept.is_empty() {
            return 0.0;
        }
        let ks: HashSet<u32> = kept.iter().map(|(k, ..)| *k).collect();
        let mut covered = 0usize;
        for &m in members {
            let keys = if qualities {
                &self.model.qkeys[m as usize]
            } else {
                &self.model.ekeys[m as usize]
            };
            if keys.iter().any(|k| ks.contains(k)) {
                covered += 1;
            }
        }
        covered as f64 / members.len() as f64
    }

    fn specific(&self, rows: &[Row]) -> bool {
        match rows.first() {
            Some((k, ..)) => {
                self.model.df[*k as usize] as f64 / self.model.num_entities as f64
                    <= self.cfg.max_df_frac
            }
            None => false,
        }
    }

    fn pick_split_keys(
        &self,
        scratch: &Scratch,
        q_touched: &[u32],
        e_touched: &[u32],
        n: usize,
    ) -> HashMap<u32, f64> {
        let mut cands: Vec<(f64, u32)> = Vec::new();
        let mut consider = |k: u32| {
            let c = scratch.counts[k as usize] as usize;
            if c < self.cfg.min_size || c > n - self.cfg.min_size {
                return;
            }
            let dfv = self.model.df[k as usize] as f64;
            cands.push((c as f64 * idf(dfv, self.model.num_entities), k));
        };
        for &k in e_touched {
            consider(k);
        }
        if !self.cfg.edges_only {
            for &k in q_touched {
                consider(k);
            }
        }
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        cands.truncate(self.cfg.fanout);
        cands.into_iter().map(|(s, k)| (k, s)).collect()
    }

    fn assign_groups(&self, members: &[u32], split_keys: &HashMap<u32, f64>) -> Vec<Vec<u32>> {
        let mut groups: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut remainder: Vec<u32> = Vec::new();
        for &m in members {
            let mut best_key: Option<u32> = None;
            let mut best_score: f64 = -1.0;
            if !self.cfg.edges_only {
                for &k in &self.model.qkeys[m as usize] {
                    if let Some(&s) = split_keys.get(&k) {
                        if s > best_score {
                            best_score = s;
                            best_key = Some(k);
                        }
                    }
                }
            }
            for &k in &self.model.ekeys[m as usize] {
                if let Some(&s) = split_keys.get(&k) {
                    if s > best_score {
                        best_score = s;
                        best_key = Some(k);
                    }
                }
            }
            if let Some(k) = best_key {
                groups.entry(k).or_default().push(m);
            } else {
                remainder.push(m);
            }
        }
        let mut out: Vec<Vec<u32>> = groups.into_values().collect();
        if !remainder.is_empty() {
            out.push(remainder);
        }
        out
    }

    fn make_cluster(
        &self,
        members: &[u32],
        qual_rows: &[Row],
        edge_rows: &[Row],
        qual_cov: f64,
        edge_cov: f64,
        truncated: bool,
    ) -> ClusterOut {
        let n = members.len();

        let dominant_qualities: Vec<Dominant> = qual_rows
            .iter()
            .map(|&(k, c, frac, lift)| Dominant {
                predicate: self.model.pred_str[self.model.key_pred[k as usize] as usize].clone(),
                value: self.model.value_str[self.model.key_value[k as usize] as usize].clone(),
                count: c,
                fraction: round4(frac),
                lift: lift.map(round2),
                steps: None,
            })
            .collect();

        let steps_map = if !edge_rows.is_empty() && n <= self.cfg.max_steps_cluster {
            self.mean_steps(members, edge_rows)
        } else {
            HashMap::new()
        };

        let dominant_edges: Vec<Dominant> = edge_rows
            .iter()
            .map(|&(k, c, frac, lift)| Dominant {
                predicate: self.model.pred_str[self.model.key_pred[k as usize] as usize].clone(),
                value: self.model.value_str[self.model.key_value[k as usize] as usize].clone(),
                count: c,
                fraction: round4(frac),
                lift: lift.map(round2),
                steps: steps_map.get(&k).and_then(|s| *s),
            })
            .collect();

        ClusterOut {
            size: n,
            num_qualities: dominant_qualities.len(),
            quality_coverage: round4(qual_cov),
            dominant_qualities,
            num_edges: dominant_edges.len(),
            edge_coverage: round4(edge_cov),
            dominant_edges,
            truncated,
        }
    }

    fn mean_steps(&self, members: &[u32], edge_rows: &[Row]) -> HashMap<u32, Option<f64>> {
        let offset = self.model.num_entities as u32;
        let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
        for &m in members {
            adj.entry(m).or_default();
            for &v in &self.model.evalues[m as usize] {
                let vn = offset + v;
                adj.entry(m).or_default().push(vn);
                adj.entry(vn).or_default().push(m);
            }
        }

        let mut out = HashMap::new();
        for &(k, _c, _f, _l) in edge_rows {
            let v = self.model.key_value[k as usize];
            let vn = offset + v;
            if !adj.contains_key(&vn) {
                out.insert(k, None);
                continue;
            }
            let mut dist: HashMap<u32, u32> = HashMap::new();
            dist.insert(vn, 0);
            let mut dq: VecDeque<u32> = VecDeque::new();
            dq.push_back(vn);
            while let Some(u) = dq.pop_front() {
                let du = dist[&u];
                if let Some(nbrs) = adj.get(&u) {
                    for &w in nbrs {
                        if !dist.contains_key(&w) {
                            dist.insert(w, du + 1);
                            dq.push_back(w);
                        }
                    }
                }
            }
            let ds: Vec<u32> = members
                .iter()
                .filter_map(|m| dist.get(m).copied())
                .collect();
            if ds.is_empty() {
                out.insert(k, None);
            } else {
                let mean = ds.iter().map(|&d| d as f64).sum::<f64>() / ds.len() as f64;
                out.insert(k, Some(round2(mean)));
            }
        }
        out
    }
}

/// Run the recursion over the full entity set and return the result.
pub fn cluster(model: &SemanticModel, cfg: &Config) -> SemanticResult {
    let members: Vec<u32> = (0..model.num_entities as u32).collect();
    let solver = Solver::new(model, cfg);
    let clusters = solver.solve(&members, 0);
    let mut clusters = clusters;
    clusters.sort_by(|a, b| b.size.cmp(&a.size));
    SemanticResult {
        num_clusters: clusters.len(),
        num_entities: model.num_entities,
        clusters,
    }
}
