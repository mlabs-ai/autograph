# Milestone 2 — Assign Entities & Predicates to Reference Frames

## Milestone Description ##

This milestone will probabilistically assign entities and predicates associated with those entities to the frames of reference determined in the first milestone. A specific entity can be assigned to more than one frame of reference - the degree of belonging is constrained to be 1 or less but there is no constraint on the sum of frame of reference assignment values as appropriate. Assignment of entities between more than one frame of reference provides the link between the different elements of the KG. Assignment of entities will be demonstrated on the same WikiData KG used earlier. 

## Milestone Deliverables ##

* Software for entity assignment of factorized knowledge graphs
* Tooling for ingestion of factorized graphs, and
* Details of the results of assignment for the WikiData knowledge graph

## Milestone Summary ##

This document is the consolidated results report for Milestone 2. It brings together
*every* clustering / assignment approach evaluated on the WikiData career
graph, each run **in the same way** — the same edge/quality role split, and the
same greedy dominant-quality / dominant-edge evaluation — so that the numbers are
directly comparable.

Synthetic ground-truth accuracy and timing are analysed in [`ACCURACY.md`](./ACCURACY.md) and
[`TIME.md`](./TIME.md). This report is the single entry point that collates all
of it.

---

## 1. Data & evaluation protocol

### 1.1 The WikiData career graph

It became clear that running previous graph framing on the entire WikiData semantic graph
would not be feasible. While we believe that the semantic variant of the Autograph algorithm would be capable of scaling
to the full WikiData graph, without the ability to make direct comparisons with existing techniques we felt that
a smaller (though still large) digest of the full graph would be more appropriate.

We therefore selected the WikiData career graph as the basis of our experiments for this milestone. The WikiData career graph
is a large sub-graph (over 11 million nodes and nearly 40 million edges) constructed around people and their occupations.

The filtered dump (`data/wikidata/person_career.json.bz2`, 3.7GB compressed) is
produced by `python/filter_wikidata.py`. Its headline size is as follows:

| Quantity | Value |
|---|---|
| Source entities ("people") | 10,802,457 |
| Typed edges | 39,381,905 |
| Distinct nodes (people + target institutions/organisations/teams) | ~11.44M |
| Full source dump | 96 GB (`latest-all.json.bz2`) |

### 1.2 The property-split (fixed for every experiment below)

When assigning entities and predicates to frames we do not have an external ground truth, only the information contained within
the graph itself. We therefore decided to define two roles for predicates, one for graph partitioning and frame assignment, and one as a surrogate for ground truth. The project's core design decision was: predicates are split into the following two *roles*.

| Role | Predicates | Purpose |
|---|---|---|
| **Qualities** | `P31` (instance of), `P106` (occupation) | *describe / evaluate* clusters only |
| **Edges** | `P39` (position held), `P69` (educated at), `P108` (employer), `P463` (member of), `P1344` (participant in), `P54` (member of sports team), `P102` (member of political party), `P101` (field of work) | *link & split* the graph |

The optional `P2868` (subject has role) experiment adds one further *edge*
predicate (see §4.3).

### 1.3 The common evaluation metric

Every clustering, regardless of algorithm, is scored with the **same** fast
greedy routine (from `evaluate_clusters.evaluate_cluster`). We were mindful
that coverage alone was not a good evaluation metric owing to the variability
in edge frequency. We did not want frequent entity targets or edges dominating
the scoring process. We borrowed a scoring metric from document processing which
we felt addressed this issue and scored according to the product of coverage and
Inverse Document Frequency (IDF):

* For each cluster, rank candidate **qualities** `(predicate, value)` and
  **edges** `(predicate, shared target)` by `coverage × IDF`
  (`IDF = log((N+1)/(df+1))`).
* Keep a tag only if `lift ≥ 2.0` and `fraction ≥ 0.02` (i.e. it is genuinely
  *more concentrated* here than in the population, not just frequent).
* Report the union coverage of the kept tags, plus each tag's count / fraction /
  lift (and, for edges, mean hop-distance "steps" on small clusters).

This is the same greedy dominant-quality / dominant-edge characterisation used
throughout the milestone; using it everywhere is what makes §3–§4 comparable.

---

## 2. Approaches compared

We compared our algorithms against the standard methods implemented in igraph. The full panoply of
approaches is given in the table below:

| Abbrev. | Method | Backend |
|---|---|---|
| **semantic** | Recursive divisive clustering with coherence early-stop (Milestone-2 reference) | `semantic_clustering.py` (pure Python) |
| **m1** | Autograph's Szemerédi-style block factorisation | `autograph.FrameGraph.cluster` (Rust) |
| **louvain** | Modularity (multilevel) | igraph `community_multilevel` |
| **leiden** | Modularity (Leiden) | igraph `community_leiden` |
| **infomap** | Map equation | igraph `community_infomap` |
| **fastgreedy** | Fast greedy modularity | igraph `community_fastgreedy` |
| **walktrap** | Random-walk trap | igraph `community_walktrap` |
| **em** | Expectation-Maximisation soft assignment | `assignment.rs::em_assign` |

The **graph methods** (`m1`, `louvain`, `leiden`, `infomap`, `fastgreedy`,
`walktrap`) see the bipartite person↔target topology built from the *edge*
predicates. The **semantic** method sees the shared-target keys directly and
also uses *qualities* to split. Each graph partition is mapped back to its
**people** and scored with the §1.3 metric, dropping singletons, so the reported
cluster counts / sizes / coverages are apples-to-apples.

---

## 3. Cross-algorithm comparison on a common sample

The full 11.4M graph is tractable only for `semantic` (§4). The graph methods
— and in particular Milestone-1 `cluster()`, whose recursion is
`O(n·(E + V log V))` in the *number of clusters* — do **not** scale to the full career graph
(M1 exhausts RAM single-threaded; walktrap's distance matrix is `O(V²)`). The
comparison below is therefore run on shared samples where every method
completes, via `python/evaluate_algorithms_career.py`. The same samples were used across methods.

### 3.1 Large Sample - 200k sample (walktrap omitted: `293k` vertices > its `120k` limit)

| Algorithm | Clusters | Coverage | Mean size | Max size | Mean qual. cov | Mean edge cov | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| semantic | 816 | **100%** | 245.0 | 15,877 | 0.688 | 0.328 | 0.38 |
| m1 | 889 | 68% | 152.8 | 37,999 | 0.904 | 0.952 | 72.7 |
| louvain | 1,678 | 65% | 77.5 | 22,023 | 0.946 | 0.980 | 4.0 |
| leiden | 1,575 | 65% | 82.5 | 14,248 | 0.945 | 0.979 | 3.0 |
| infomap | 8,433 | 64% | 15.3 | 2,070 | 0.925 | 0.992 | 92.0 |
| fastgreedy | 2,225 | 65% | 58.4 | 23,194 | 0.952 | 0.988 | 209.1 |

### 3.2 Small Sample - 30k sample (walktrap included)

| Algorithm | Clusters | Coverage | Mean size | Max size | Mean qual. cov | Mean edge cov | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| semantic | 425 | **100%** | 70.4 | 2,327 | 0.405 | 0.202 | 0.09 |
| m1 | 152 | 79% | 156.4 | 13,883 | 0.916 | 0.940 | 2.9 |
| louvain | 415 | 73% | 53.1 | 2,938 | 0.949 | 0.956 | 0.6 |
| leiden | 400 | 73% | 55.1 | 2,555 | 0.949 | 0.952 | 0.5 |
| infomap | 2,434 | 71% | 8.8 | 492 | 0.964 | 0.993 | 17.4 |
| fastgreedy | 624 | 73% | 34.9 | 5,121 | 0.964 | 0.973 | 5.1 |
| walktrap | 1,352 | 65% | 14.3 | 5,943 | 0.979 | 0.993 | 42.1 |

*Coverage = fraction of people assigned to a cluster of size ≥ 2.*

> The `semantic` row's *time* is the **Rust** implementation
> (`semantic_cluster`, §8), measured single-threaded (`--procs 1`) to match the
> single-threaded igraph baselines; the cluster/coverage columns are unchanged
> (output is identical to the Python version). Precise clustering times:
> 0.378s at 200k, 0.086s at 30k. Python (`Clusterer`) was 2.5s / 0.4s,
> i.e. Rust is ≈6.6× / ≈4.7× faster at these scales.

### 3.3 Interpretation of these results

1. **Only `semantic` covers everyone.** It is a *partition*, so ~100% of people
   land in a size ≥ 2 cluster. Every link-based method leaves ~29–36% of people
   as singletons: a person whose only link is to a unique employer or niche team
   forms a community of "me + my target", which maps to a 1-person people-cluster
   and is dropped.
2. **The link-based methods cluster *topology*, not *semantics*.** Their dominant
   "*edge*" (a shared target) explains 94–99% of every cluster's members
   (`mean edge cov` ≈ 0.95–0.99), but that shared target is usually a **hub
   institution** — a national political party, an Olympic games, a learned
   society. This produces either very small, tight cliques (infomap: 8.4k
   clusters of mean size 15) or a few giant frames glued by a universal target
   (`max size` of 22k–38k for louvain/m1). It is exactly the "universal
   predicate / hub" failure the semantic method was built to avoid.
3. **`semantic` trades raw edge-coverage for discriminating frames.** Its mean
   edge coverage is lower (0.20–0.33) because it *refuses* to stop on an
   undiscriminating universal target; instead it splits on the *specific*
   occupation / shared link that actually distinguishes one reference frame
   from the next (IDF down-weights near-universal tags). Its dominant *quality*
   coverage (0.69 at 200k) is the part that carries the frame's meaning.
4. **Speed.** Louvain/Leiden are the fastest link-based methods (single-digit
   seconds to 200k); infomap and fastgreedy are markedly slower at scale;
   `semantic` is sub-second to a few seconds at these sizes.

---

## 4. The Milestone-2 reference: semantic clustering at full scale

`semantic` is the only method that completes on the full 11.4M graph. Two
variants were produced (`results/semantic_full.json` and
`results/semantic_full_p2868.json`).

### 4.1 Base (edge predicates = the 8 core set)

| Metric | Value |
|---|---|
| Entities | 10,802,457 |
| Clusters | 11,268 |
| Truncated clusters (depth/step guard) | 3 (1,325,133 members) |
| Non-truncated clusters | 11,265 (**9,477,324 members = 87.7%**) |
| Run time | ~179s stream + ~231s cluster |

**Top frames (dominant *occupation* quality; QID labels resolved via the
Wikidata API, cached in `results/qid_labels.json`):**

| Size | Frame (dominant quality) | Dominant binding edge |
|---:|---|---|
| 959,040 | politician (`Q82955`) | `P102` → Democratic Party (`Q29552`), Republican Party (`Q29468`) |
| 430,133 | writer (`Q36180`) | `P101` → literature (`Q8242`), creative & professional writing (`Q113209507`) |
| 388,836 | association football player (`Q937857`) | — |
| 376,007 | actor (`Q33999`) | — |
| 301,321 | university teacher (`Q1622272`) | `P101` → history (`Q309`) |
| 256,653 | basketball player (`Q3665646`) | — |
| 223,492 | painter (`Q1028181`) | `P101` → painting (`Q11629`) |
| 169,628 | physician (`Q39631`) | `P101` → medicine (`Q11190`) |
| 132,246 | journalist (`Q1930187`) | `P101` → journalism (`Q11030`) |
| 121,516 | musician (`Q639669`) | `P101` → music (`Q638`), performing arts (`Q184485`) |

The three truncated clusters are the "researcher" mega-frame (1.19M members,
top quality `Q1650915` at 100%) plus two small residuals — the known
depth-guard tail, not a correctness failure.

### 4.2 Dominant-edge predicate spread (base)

Across all dominant-edge entries, the edge predicates rank (presence):

`P108` (23,361) · `P69` (16,838) · `P101` (5,808) · `P463` (4,255) ·
`P39` (3,277) · `P1344` (2,529) · `P102` (689) · `P54` (258)

`P108`/`P69` (employer / educated-at)
are the most *frame-concentrating* edges, while `P102`/`P54` (party / team
membership) bind large but fewer frames.

### 4.3 Resolved dominant-edge targets (what binds the frames)

The most prominent shared targets that bind the full-graph clusters, resolved
to English labels (`results/qid_labels.json`):

**Academic positions — `P39` (position held).** The single largest cluster of
dominant edge values is a ladder of research/teaching positions:
*professor, associate professor, assistant professor, lecturer, researcher,
research fellow, research associate, research assistant, doctoral student,
postdoctoral researcher.*

**Academic fields — `P101` (field of work).** *history, medicine, literature,
painting, poetry, music, performing arts, journalism, law, botany, architecture,
photography, art history, translation, illustration, creative & professional
writing, history of Japan* — the disciplinary reference frames the milestone is
meant to recover.

**Universities — `P69` / `P108` (educated-at / employer).** A long tail of world
universities, led by Oxford, Cambridge, Harvard, Stanford, MIT, Columbia, Yale,
the University of Tokyo, UCL, Imperial College London, LSE, ETH Zurich, and
U.S./Commonwealth state universities (Michigan, Wisconsin–Madison, Minnesota,
Ohio State, Texas, Toronto, McGill, Melbourne, Sydney, Auckland, and so on…).

**Political parties — `P102` (member of political party).** *Democratic Party,
Republican Party, Communist Party of the Soviet Union, Chinese Communist Party,
etc.*

**Olympic Games — `P1344` (participant in).** The Summer Games from *1988, 1992,
1996, 2000, 2004, 2008 and 2012*.

**Catholic hierarchy — `P39` (position held).** *bishop, diocesan bishop,
auxiliary bishop, titular bishop, Catholic archbishop, Catholic bishop* — a
self-contained ecclesiastical frame.

**Other notable organisations.** *Wagner Group* (`Q36597284`), *Paris Foreign
Missions Society*, *International Astronomical Union*, *Bavarian Football
Association*, *Cameroon Bar Association*, *United States Military Academy*.

### 4.4 `P2868` (subject has role) added as an edge

`P2868` is a *qualifier* (it lives inside other statements' `qualifiers`, not as
a top-level claim), so it required a dedicated extractor. Re-running with it:

| Metric | Value |
|---|---|
| Clusters | 10,677 |
| Truncated clusters | 3 (1,320,643 members) |
| Non-truncated clusters | 10,674 (**9,481,814 members = 87.8%**) |
| Clusters with a `P2868` dominant edge | 703 |

`P2868` is comparatively sparse (~347 k occurrences), so it surfaces as a
dominant edge only in the 703 clusters where it is genuinely discriminating —
the expected behaviour for a niche role qualifier.

### 4.5 Edges-only ablation (the non-circular quality check)

To check whether the strong "dominant occupation ≈ 100%" numbers are merely an
artifact of *using* occupation to split, the qualities were fully held out of the
split/stop logic (`--edges-only`), leaving only the 8 edge predicates to build
the clustering; qualities were then used purely to *evaluate* the result.
Artifact: `results/semantic_edges_only_full.json`.

| Metric | Qualities in loop | Edges only |
|---|---:|---:|
| Clusters | 11,268 | 7,999 |
| Truncated (unresolved) entities | **12.3%** | **60.9%** (one 6.58M member mega-residual) |
| Entities in coherent (non-truncated) clusters | 87.7% | **39.1%** |
| Mean domain. quality coverage (non-truncated) | 0.785 | 0.791 |
| Mean domain. edge coverage (non-trucated) | 0.723 | 1.000 |

Two conclusions:

1. **Qualities do the heavy lifting of *separating* the graph.** Without them,
   ~61% of entities collapse into a single undifferentiated residual that shared
   edges alone cannot tell apart (the universal-hub problem). Qualities are what
   let the clustering extend from ~39% to ~88% coverage.
2. **The quality coverage is *not* artificially inflated by circularity.** Where
   edges *do* manage to split (39% of entities), the resulting communities are
   genuinely occupationally coherent — mean quality coverage 0.791, with 62% of
   them at ≥ 80%. The edge coverage is 1.0 by construction (the edges-only method
   splits *on* shared targets), which is the true mirror of the full version's
   high quality coverage.

In short: occupation is a real, concentrated signal in these edge communities;
the semantic method's value is mostly that it keeps *splitting the long tail*
that edges alone leave unresolved.

## 5. Synthetic ground-truth accuracy & timing

For completeness, the planted-partition benchmark (`evaluate_other_algorithms.py`)
puts Autograph's clustering against the same igraph baselines on graphs of known
structure. (These are *not* the career graph; they measure raw partition
recovery, summarised from `ACCURACY.md` / `TIME.md`.)

### 5.1 Adjusted Rand score (higher is better; ~1.0 = recovered)

| Algorithm | 150 | 500 | 1000 | 1500 | 2000 |
|---|--:|--:|--:|--:|--:|
| Infomap | 0.996 | 0.984 | 0.977 | 0.971 | 0.966 |
| Walktrap | 0.998 | 0.997 | 0.998 | — | — |
| Louvain | 0.879 | 0.656 | 0.531 | 0.469 | 0.428 |
| Leiden | 0.878 | 0.655 | 0.533 | 0.469 | 0.429 |
| Autograph | 0.731 | 0.700 | 0.715 | 0.695 | 0.683 |
| Fast Greedy | 0.226 | 0.110 | 0.083 | 0.074 | 0.070 |

### 5.2 Wall-clock (seconds)

| Algorithm | 150 | 1000 | 2000 |
|---|--:|--:|--:|
| Fast Greedy | 0.15 | 1.00 | 1.81 |
| Leiden | 0.55 | 6.15 | 10.8 |
| Louvain | 0.83 | 14.4 | 24.6 |
| Autograph | 0.39 | 20.3 | 54.3 |
| Infomap | 2.41 | 23.5 | 41.1 |
| Walktrap | 2.86 | 144 | — |

On clean planted graphs, Infomap/Walktrap recover the partition nearly perfectly but have limited ability to scale
to the size of semantic graph encountered in practice;
Autograph holds ~0.68–0.73 across a 13× size growth, overtaking the tuned
modularity methods (which degrade sharply) around 500 clusters.

---

## 6. Findings & recommendations

1. **For frame-of-reference discovery on the career graph, `semantic` is the
   only method that both completes at 11.4M and yields *semantically coherent*
   frames.** The link-based methods (Szemerédi, Louvain, Leiden, Infomap, Walktrap,
   Fast Greedy) are dominated by hub institutions and split into either
   mega-frames or micro-cliques, leaving ~30% of people unrecovered.
2. **The greedy dominant-quality/edge evaluation is the right lens.** It is fast,
   deterministic, and reproduces the "frame-specific vs universal
   predicate" distinction on real data.
3. **`P2868` is a viable additional signal** (qualifier handling is now in
   place) but sparse; it helps a minority of clusters.
4. **Scale reality.** The milestone-1 `cluster()` recursion does not scale to
   the full graph; `semantic_clustering.py` (recursive, interleaved) is the
   replacement and is the recommended path forward.

---

## 8. Rust reimplementation & performance

The semantic recursive clustering (`python/semantic_clustering.py::Clusterer`)
was reimplemented in Rust for performance:

* `autograph_core/src/semantic.rs` — interned data model + the recursive
  `cluster()` (generation-tagged counting arrays reused across the recursion),
  plus the same greedy dominant-quality/edge evaluation, including the BFS
  "steps" metric.
* `autograph_core/src/bin/semantic_cluster.rs` — end-to-end binary: parallel
  `lbzip2` decompression → rayon JSON parse → interning → clustering → JSON.

### 8.1 Output parity

At full scale the Rust binary reproduces the Python result exactly: **11,268
clusters over 10,802,457 entities**, with an **identical cluster-size multiset**
and an **identical top dominant quality per cluster**. A minority of clusters
(≈6.6%) differ only in the *presentation* of secondary dominant tags — the
ordering of equal `coverage × idf` scores (Python's stable sort ties by
dict-insertion order; Rust ties by first-seen order) and in `lift` rounded to
2 dp (Python round-half-to-even vs Rust round-half-away-from-zero). These do not
affect cluster membership.

Artifacts: `results/semantic_rust_full.json` (structurally identical to
`semantic_full.json`).

### 8.2 Speed (full 11.4M graph, 12 parse threads + 4-way lbzip2)

| Phase | Python | Rust | Speedup |
|---|---:|---:|---:|
| Extract (decompress + parse + intern) | 179.2s | 162.8s | 1.1× |
| Cluster (recursive semantic) | 231.1s | 7.1s | **32×** |
| **Total** | **410.3s** | **~170s** | **2.4×** |

The clustering algorithm itself is ~32× faster in Rust. Extraction is I/O-bound
(dominated by `lbzip2` decompression + serde JSON parsing) and is already
parallel in both implementations, so the two are comparable there.

The Rust clustering is parallelised further: the recursion's split groups are
mutually independent, so they run across the rayon worker pool (per-thread
generation-tagged counting scratch, no locks). Single-threaded Rust clustering
is 16.5s; parallelising the recursion brings it to ~7.1s.

> Progression: (1) single-threaded ingest + single-threaded cluster — extract
> 1123 s, cluster 20.6 s; (2) parallel ingest (`lbzip2` + rayon) — extract
> 162 s, cluster 16.5 s; (3) + parallel recursion — cluster 7.1 s. The 32×
> clustering gain over Python is independent of ingest parallelism.

---

## 9. Deliverables

| Deliverable | Where |
|---|---|
| Entity assignment software | `assignment.rs::em_assign` (soft EM) + `semantic_clustering.py` (greedy, scalable) |
| Tooling for ingestion of factorized graphs | `FrameGraph::frames_from_json` / `seed_from_frames`; `assign_wiki.py --frames-file` |
| Details of results for the WikiData KG | this report + `ACCURACY.md`, `TIME.md`, `results/` |

## 10. Reproducing these results

The commands below reproduce the main results of this report. All assume the
full Wikidata dump is available at `data/wikidata/latest-all.json.bz2` and are run from the repo root.

```sh
# 1. Filter the full dump -> career graph (person_career.json.bz2)
#    Keeps only entities matching the P106/P39/P69/P108/P463/P1344/P54/P102/P101 property set.
python3 python/filter_wikidata.py data/wikidata/latest-all.json.bz2 data/wikidata/person_career.json.bz2

# 2. Graph size stats (source entities / typed edges / distinct nodes)
#    Produces the numbers quoted in §1.1 and §4.
python3 python/graph_stats.py data/wikidata/person_career.json.bz2

# 3. Semantic clustering, full graph (Python reference)
#    12 parse threads + 4-way lbzip2 decompression; writes results/semantic_full.json plus a .log.
python3 python/semantic_clustering.py data/wikidata/person_career.json.bz2 results/semantic_full.json --procs 12 --io-procs 4

# 4. P2868 variant (adds "subject has role" as an extra edge) — §4.4
python3 python/semantic_clustering.py data/wikidata/person_career.json.bz2 results/semantic_full_p2868.json --procs 12 --io-procs 4

# 5. Cross-algorithm comparison on the common sample — §3
#    200k sample (walktrap omitted: too many vertices); then the 30k sample with walktrap included.
python3 python/evaluate_algorithms_career.py data/wikidata/person_career.json.bz2 results/career_algorithms_200k.json --limit 200000
python3 python/evaluate_algorithms_career.py data/wikidata/person_career.json.bz2 results/career_algorithms_30k.json --limit 30000 --walktrap

# 6. Rust reimplementation (build + run full graph) — §8
cargo build --release -p autograph_core --bin semantic_cluster
./target/release/semantic_cluster data/wikidata/person_career.json.bz2 results/semantic_rust_full.json --procs 12 --io-procs 4
./target/release/semantic_cluster data/wikidata/person_career.json.bz2 results/semantic_edges_only_full.json --procs 12 --io-procs 4 --edges-only

# 7. Resolve QIDs -> labels (reads a QID list via stdin; needs a Wikidata API User-Agent)
python3 python/resolve_qids.py
```

Notes:

- The §8.2 speed table maps the Python timings (179.2s extract / 231.1s cluster, from
  `results/semantic_full.log`) against the Rust timings (162.8s / 7.1s) — same 12-parse-thread / 4-way
  lbzip2 configuration. On small samples the `.log` timings are sub-second and printed with 3 decimal places,
  so recompute cross-algorithm tables on the same CPU count to keep comparisons fair.
- The semantic row in the §3 tables was timed single-threaded (`--procs 1`) to match the single-threaded
  igraph baselines; the M1 row uses all cores via rayon.

## 11. Milestone Conclusions ##

We have demonstrated that the semantic variant of the Autograph algorithm scales well to sizeable semantic graphs
with over 11M nodes and nearly 40M edges. We believe that the technique can accommodate even larger graphs
and provides a substantive breakthrough in producing a tractable graph factorisation method for real-world semantic knowledge
graphs. Moreover, the experimental results show a useful semantic factorisation rather than a purely topological partition:
a significant advantage over previous methods.

This report details how Autograph can indeed be employed to assign entities and predicates to a large factorised knowledge
graph.
