//! End-to-end semantic-clustering binary (Milestone 2).
//!
//! Reads the filtered Wikidata career dump (`person_career.json.bz2`), extracts
//! the same P31/P106 qualities + 8 core edge predicates as
//! `python/semantic_clustering.py`, runs the Rust `semantic::cluster` recursion,
//! and writes the identical JSON result.
//!
//! Ingest is parallelised to match the Python driver: `lbzip2` (block-parallel
//! bzip2) feeds decompressed lines into a rayon worker pool that parses the JSON
//! and extracts `(predicate, value)` pairs; interning stays on the main thread.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc::sync_channel;
use std::time::Instant;

use autograph_core::semantic::{Config, ModelBuilder, SemanticResult, cluster};
use rayon::prelude::*;
use serde_json::Value;

const QUALITY_PREDICATES: [&str; 2] = ["P31", "P106"];
const EDGE_PREDICATES: [&str; 8] = ["P39", "P69", "P108", "P463", "P1344", "P54", "P102", "P101"];

/// One parsed entity, as owned strings (the parallel path cannot borrow the
/// serde values across rayon workers).
struct OwnedEntity {
    quals: Vec<(String, String)>,
    edges: Vec<(String, String)>,
}

fn parse_line(line: &str) -> Option<OwnedEntity> {
    let start = line.find('{')?;
    let end = line.rfind('}')?;
    if start > end {
        return None;
    }
    let json: Value = serde_json::from_str(&line[start..=end]).ok()?;
    let claims = json.get("claims")?.as_object()?;

    let mut quals: Vec<(String, String)> = Vec::new();
    let mut edges: Vec<(String, String)> = Vec::new();

    for pred in QUALITY_PREDICATES.iter().chain(EDGE_PREDICATES.iter()) {
        if let Some(stmts) = claims.get(*pred).and_then(|v| v.as_array()) {
            for stmt in stmts {
                if let Some(id) = stmt
                    .get("mainsnak")
                    .and_then(|m| m.get("datavalue"))
                    .and_then(|d| d.get("value"))
                    .and_then(|v| v.get("id"))
                    .and_then(|i| i.as_str())
                {
                    let pair = (pred.to_string(), id.to_string());
                    if QUALITY_PREDICATES.contains(pred) {
                        quals.push(pair);
                    } else {
                        edges.push(pair);
                    }
                }
            }
        }
    }

    Some(OwnedEntity { quals, edges })
}

/// Locate a parallel bzip2 decompressor, falling back to plain `bzip2`.
fn decompressor() -> (String, bool) {
    for name in ["lbzip2", "bzip2"] {
        if Command::new(name)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
            return (name.to_string(), name == "lbzip2");
        }
    }
    ("bzip2".to_string(), false)
}

fn ingest(path: &Path, limit: Option<usize>, io_procs: usize, procs: usize) -> ModelBuilder {
    let (dec, parallel) = decompressor();

    let mut cmd = Command::new(&dec);
    if parallel {
        cmd.arg("-dc").arg("-n").arg(io_procs.to_string()).arg(path);
    } else {
        cmd.arg("-dc").arg(path);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = cmd.spawn().expect("spawn decompressor");
    let stdout = child.stdout.take().unwrap();

    // Reader thread: decompress + chunk lines (decompression overlaps with parse).
    const CHUNK: usize = 8192;
    let (tx, rx) = sync_channel::<Vec<String>>(procs * 2 + 1);
    let reader = std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut buf: Vec<String> = Vec::with_capacity(CHUNK);
        let mut raw = Vec::new();
        loop {
            raw.clear();
            let n = reader.read_until(b'\n', &mut raw).unwrap_or(0);
            if n == 0 {
                break;
            }
            buf.push(String::from_utf8_lossy(&raw).into_owned());
            if buf.len() >= CHUNK {
                let chunk = std::mem::take(&mut buf);
                buf.reserve(CHUNK);
                if tx.send(chunk).is_err() {
                    break;
                }
            }
        }
        if !buf.is_empty() {
            let _ = tx.send(buf);
        }
    });

    let mut builder = ModelBuilder::new();
    let mut count = 0usize;
    'outer: for chunk in rx {
        let parsed: Vec<OwnedEntity> = chunk
            .into_par_iter()
            .filter_map(|line| parse_line(&line))
            .collect();

        for re in parsed {
            let qs: Vec<(&str, &str)> = re
                .quals
                .iter()
                .map(|(a, b)| (a.as_str(), b.as_str()))
                .collect();
            let es: Vec<(&str, &str)> = re
                .edges
                .iter()
                .map(|(a, b)| (a.as_str(), b.as_str()))
                .collect();
            builder.add_entity(&qs, &es);
            count += 1;
            if let Some(lim) = limit {
                if count >= lim {
                    break 'outer;
                }
            }
        }
    }

    let _ = reader.join();
    let _ = child.wait();
    builder
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<String>>();
    let mut limit: Option<usize> = None;
    let mut procs = 12usize;
    let mut io_procs = 4usize;
    let mut cfg = Config::default();
    let mut positional: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        let a = args[i].clone();
        macro_rules! next_val {
            () => {{
                i += 1;
                args[i].clone()
            }};
        }
        match a.as_str() {
            "--limit" => limit = Some(next_val!().parse().expect("--limit")),
            "--procs" => procs = next_val!().parse().expect("--procs"),
            "--io-procs" => io_procs = next_val!().parse().expect("--io-procs"),
            "--min-size" => cfg.min_size = next_val!().parse().expect("--min-size"),
            "--max-depth" => cfg.max_depth = next_val!().parse().expect("--max-depth"),
            "--fanout" => cfg.fanout = next_val!().parse().expect("--fanout"),
            "--coverage-target" => {
                cfg.coverage_target = next_val!().parse().expect("--coverage-target")
            }
            "--min-lift" => cfg.min_lift = next_val!().parse().expect("--min-lift"),
            "--min-fraction" => cfg.min_fraction = next_val!().parse().expect("--min-fraction"),
            "--max-df-frac" => cfg.max_df_frac = next_val!().parse().expect("--max-df-frac"),
            "--max-picks" => cfg.max_picks = next_val!().parse().expect("--max-picks"),
            "--max-steps-cluster" => {
                cfg.max_steps_cluster = next_val!().parse().expect("--max-steps-cluster")
            }
            "--edges-only" => cfg.edges_only = true,
            other => positional.push(other.to_string()),
        }
        i += 1;
    }

    let input = &positional[0];
    let output = &positional[1];

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(procs)
        .build_global();

    eprintln!("Ingesting & extracting qualities/edges...");
    let t0 = Instant::now();
    let builder = ingest(Path::new(input), limit, io_procs, procs);
    let model = builder.finish();
    let extract_s = t0.elapsed().as_secs_f64();
    eprintln!("  {} entities ({:.1}s)", model.num_entities, extract_s);

    eprintln!("Recursive semantic clustering (coherence early-stop)...");
    let t0 = Instant::now();
    let result = cluster(&model, &cfg);
    let cluster_s = t0.elapsed().as_secs_f64();
    eprintln!("  {} clusters ({:.3}s)", result.num_clusters, cluster_s);

    let out_file = File::create(output).expect("create output");
    let mut writer = std::io::BufWriter::new(out_file);
    serde_json::to_writer(&mut writer, &result).expect("serialize");
    writer.flush().expect("flush");
    eprintln!("Wrote {output}");

    print_summary(&result);
}

fn print_summary(result: &SemanticResult) {
    eprintln!();
    for c in result.clusters.iter().take(20) {
        let mut qs = String::new();
        for r in c.dominant_qualities.iter().take(3) {
            if !qs.is_empty() {
                qs.push_str("; ");
            }
            qs.push_str(&format!(
                "{}:{}@{:.0}%",
                r.predicate,
                r.value,
                r.fraction * 100.0
            ));
        }
        let mut es = String::new();
        for r in c.dominant_edges.iter().take(3) {
            if !es.is_empty() {
                es.push_str("; ");
            }
            es.push_str(&format!("{}->{}({})", r.predicate, r.value, r.count));
        }
        eprintln!(
            "cluster ({} members, {}):",
            c.size,
            if c.truncated { "TRUNC" } else { "ok" }
        );
        eprintln!(
            "  qualities({},cov={:.0}%): {}",
            c.num_qualities,
            c.quality_coverage * 100.0,
            qs
        );
        eprintln!(
            "  edges({},cov={:.0}%): {}",
            c.num_edges,
            c.edge_coverage * 100.0,
            es
        );
    }
}
