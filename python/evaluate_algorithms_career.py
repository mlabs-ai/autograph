#!/usr/bin/env python3
"""Compare clustering algorithms on the Wikidata career graph.

Run every clustering backend *in the same way* as `semantic_clustering.py` so
the milestone-2 results are directly comparable:

* Same edge / quality role split:
    - **Edges**     (build + split the graph):
        P39, P69, P108, P463, P1344, P54, P102, P101.
    - **Qualities** (evaluate the clusters):
        P31 (instance of), P106 (occupation).
* Same greedy dominant-quality / dominant-edge evaluation
  (`evaluate_clusters.evaluate_cluster`).

Algorithms compared on the **same** sample of the dump:

* `semantic`    — recursive divisive clustering with coherence early-stop
                  (`semantic_clustering.Clusterer`), the Milestone-2 reference.
* `m1`          — Autograph's Szemeredi-style block factorisation
                  (`autograph.FrameGraph.cluster`).
* `louvain`     — igraph `community_multilevel`.
* `leiden`      — igraph `community_leiden` (modularity objective).
* `infomap`     — igraph `community_infomap`.
* `fastgreedy`  — igraph `community_fastgreedy` (cut at max modularity).
* `walktrap`    — igraph `community_walktrap` (optional; O(V^2) memory).

The graph methods see the bipartite person<->target topology built from the
same typed edges; `semantic` sees the shared-target keys directly. Each partition
is mapped back to its *people* and scored with the identical evaluation, so the
reported cluster counts, sizes and dominant quality/edge coverages are
comparable across methods.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

from autograph import autograph
import igraph

from evaluate_clusters import (
    QUALITY_PREDICATES,
    EDGE_PREDICATES,
    _extract_batch,
    evaluate_cluster,
)
from semantic_clustering import Clusterer


# Auto-skip walktrap above this many total vertices (its distance matrix is
# quadratic in V and exhausts RAM; see docs/ACCURACY.md).
WALKTRAP_VERTEX_LIMIT = 120_000


def _collect(path, limit, procs, io_procs, chunk_lines):
    """Stream the dump and collect interned attributes + global df (same as
    semantic_clustering). Returns attrs/df dicts plus the people & target sets."""
    lb = shutil.which("lbzip2") or shutil.which("bzip2")
    decomp = subprocess.Popen(
        [lb, "-dc", "-n", str(io_procs), path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    attr_qualities = {}
    attr_edges = {}
    quality_df = {}
    edge_df = {}
    pool = ProcessPoolExecutor(max_workers=procs)
    pending = deque()
    buf = []
    seen = 0
    t0 = time.time()

    def drain():
        for rec in pending.popleft().result():
            qid = sys.intern(rec["qid"])
            qs = frozenset((sys.intern(p), sys.intern(v))
                           for p, v in rec["qualities"])
            for k in qs:
                quality_df[k] = quality_df.get(k, 0) + 1
            attr_qualities[qid] = qs
            if rec["edges"]:
                e2 = {}
                for pred, tgts in rec["edges"].items():
                    ip = sys.intern(pred)
                    it = [sys.intern(t) for t in sorted(set(tgts))]
                    e2[ip] = it
                    for t in it:
                        edge_df[(ip, t)] = edge_df.get((ip, t), 0) + 1
                attr_edges[qid] = e2

    try:
        for raw in decomp.stdout:
            buf.append(raw.decode("utf-8", "replace").rstrip("\n"))
            seen += 1
            if len(buf) >= chunk_lines:
                pending.append(pool.submit(_extract_batch, buf))
                buf = []
            while len(pending) >= procs * 2:
                drain()
            if limit and seen >= limit:
                break
        if buf:
            pending.append(pool.submit(_extract_batch, buf))
        while pending:
            drain()
    finally:
        pool.shutdown(cancel_futures=True)
        decomp.stdout.close()
        decomp.terminate()
        decomp.wait()

    people = list(attr_qualities.keys())
    targets = set()
    for e in attr_edges.values():
        for tgts in e.values():
            targets.update(tgts)
    print(f"  {len(people):,} people, {len(targets):,} targets "
          f"({time.time() - t0:.1f}s)", flush=True)
    return attr_qualities, attr_edges, quality_df, edge_df, people, targets


def _m1_frames(attr_edges, factor, steps, threshold, min_cluster_size):
    """Autograph Milestone-1 cluster() over the person<->target union graph."""
    graph = autograph.FrameGraph()
    for qid, edges in attr_edges.items():
        for pred, tgts in edges.items():
            for t in tgts:
                graph.add_edge(qid, pred, t)
    return graph.cluster(factor, steps, threshold, min_cluster_size)


def _igraph_person_communities(attr_edges, people, method):
    """Build the bipartite person<->target igraph and return a dict
    community_id -> list[person_qid] using the given community method."""
    targets = set()
    for e in attr_edges.values():
        for tgts in e.values():
            targets.update(tgts)
    labels = list(people) + list(targets)
    idx = {lab: i for i, lab in enumerate(labels)}
    g = igraph.Graph(directed=False)
    g.add_vertices(len(labels))
    # A person may reach the same target via several predicates; collapse to a
    # simple (multi-edge-free) graph so every igraph method (incl. fast-greedy,
    # which refuses multi-edges) sees the identical topology.
    edges = set()
    for p, e in attr_edges.items():
        pi = idx[p]
        for tgts in e.values():
            for t in tgts:
                edges.add((pi, idx[t]))
    g.add_edges(edges)

    memb = method(g)

    comm = {}
    for p in people:
        c = int(memb[idx[p]])
        comm.setdefault(c, []).append(p)
    return comm


def _louvain(g):
    return g.community_multilevel().membership


def _leiden(g):
    return g.community_leiden(objective_function="modularity").membership


def _infomap(g):
    return g.community_infomap().membership


def _fastgreedy(g):
    return g.community_fastgreedy().as_clustering().membership


def _walktrap(g):
    return g.community_walktrap().as_clustering().membership


def _evaluate_people_sets(people_sets, attr_qualities, attr_edges, n_total,
                          quality_df, edge_df, max_picks, min_lift,
                          min_fraction, max_steps_cluster, min_members):
    evals = []
    for members in people_sets:
        members = [m for m in members if m in attr_qualities or m in attr_edges]
        if len(members) < min_members:
            continue
        ev = evaluate_cluster(members, attr_qualities, attr_edges, n_total,
                              quality_df, edge_df, max_picks, min_lift,
                              min_fraction, max_steps_cluster)
        # Record size explicitly; evaluate_cluster already returns 'size' == n.
        evals.append(ev)
    evals.sort(key=lambda c: -c["size"])
    return evals


def _summarize(name, evals, n_total, time_s):
    evals = [e for e in evals if e["size"] >= 2]
    sizes = [e["size"] for e in evals]
    coverage = round(sum(sizes) / n_total, 4) if n_total else 0.0
    qc = [e["quality_coverage"] for e in evals]
    ec = [e["edge_coverage"] for e in evals]
    nq = [e["num_qualities"] for e in evals]
    ne = [e["num_edges"] for e in evals]

    summary = {
        "algorithm": name,
        "n_clusters": len(evals),
        "n_entities": n_total,
        "coverage": coverage,
        "mean_size": round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
        "max_size": max(sizes) if sizes else 0,
        "mean_quality_coverage": round(sum(qc) / len(qc), 4) if qc else 0.0,
        "mean_edge_coverage": round(sum(ec) / len(ec), 4) if ec else 0.0,
        "mean_num_qualities": round(sum(nq) / len(nq), 2) if nq else 0.0,
        "mean_num_edges": round(sum(ne) / len(ne), 2) if ne else 0.0,
        "time_s": round(time_s, 1),
        "top": [],
    }
    for e in evals[:5]:
        dq = e["dominant_qualities"]
        de = e["dominant_edges"]
        summary["top"].append({
            "size": e["size"],
            "quality": f"{dq[0]['predicate']}:{dq[0]['value']}" if dq else None,
            "quality_frac": round(dq[0]["fraction"], 3) if dq else None,
            "quality_lift": dq[0]["lift"] if dq else None,
            "edge": f"{de[0]['predicate']}->{de[0]['value']}" if de else None,
            "edge_frac": round(de[0]["fraction"], 3) if de else None,
        })
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--limit", type=int, default=200_000)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--io-procs", type=int, default=4)
    ap.add_argument("--chunk-lines", type=int, default=1024)
    ap.add_argument("--factor", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--boundary-threshold", type=float, default=0.1)
    ap.add_argument("--min-cluster-size", type=int, default=10)
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--max-depth", type=int, default=40)
    ap.add_argument("--fanout", type=int, default=200)
    ap.add_argument("--coverage-target", type=float, default=0.6)
    ap.add_argument("--min-lift", type=float, default=2.0)
    ap.add_argument("--min-fraction", type=float, default=0.02)
    ap.add_argument("--max-df-frac", type=float, default=0.1)
    ap.add_argument("--max-picks", type=int, default=10)
    ap.add_argument("--max-steps-cluster", type=int, default=100_000)
    ap.add_argument("--walktrap", action="store_true")
    ap.add_argument("--no-m1", action="store_true")
    ap.add_argument("--no-semantic", action="store_true")
    args = ap.parse_args()

    (attr_qualities, attr_edges, quality_df, edge_df,
     people, targets) = _collect(args.input, args.limit, args.procs,
                                 args.io_procs, args.chunk_lines)
    n_total = len(people)
    n_vertices = n_total + len(targets)

    semi_args = argparse.Namespace(
        min_size=args.min_size, max_depth=args.max_depth, fanout=args.fanout,
        coverage_target=args.coverage_target, min_lift=args.min_lift,
        min_fraction=args.min_fraction, max_df_frac=args.max_df_frac,
        max_picks=args.max_picks, max_steps_cluster=args.max_steps_cluster,
    )

    results = []

    # --- semantic (reference) ------------------------------------------------
    if not args.no_semantic:
        print("Running semantic (recursive) ...", flush=True)
        t0 = time.time()
        cl = Clusterer(attr_qualities, attr_edges, quality_df, edge_df,
                       semi_args)
        cl.solve(list(people))
        # cl.out are already-evaluated dicts (same shape as evaluate_cluster).
        results.append(_summarize("semantic", cl.out, n_total,
                                  time.time() - t0))
        print(f"  {len(cl.out):,} clusters", flush=True)

    # --- M1 Szemeredi --------------------------------------------------------
    if not args.no_m1:
        print("Running M1 (Szemeredi) ...", flush=True)
        t0 = time.time()
        frames = _m1_frames(attr_edges, args.factor, args.steps,
                            args.boundary_threshold, args.min_cluster_size)
        people_sets = [[m for m in f if m in attr_qualities] for f in frames]
        evals = _evaluate_people_sets(people_sets, attr_qualities, attr_edges,
                                      n_total, quality_df, edge_df,
                                      args.max_picks, args.min_lift,
                                      args.min_fraction, args.max_steps_cluster,
                                      min_members=2)
        results.append(_summarize("m1", evals, n_total, time.time() - t0))
        print(f"  {len(frames):,} frames -> {len(evals):,} people-clusters",
              flush=True)

    # --- igraph baselines ----------------------------------------------------
    igraph_methods = [
        ("louvain", _louvain),
        ("leiden", _leiden),
        ("infomap", _infomap),
        ("fastgreedy", _fastgreedy),
    ]
    if args.walktrap or n_vertices <= WALKTRAP_VERTEX_LIMIT:
        igraph_methods.append(("walktrap", _walktrap))
    else:
        print(f"Skipping walktrap ({n_vertices} vertices > "
              f"{WALKTRAP_VERTEX_LIMIT})", flush=True)

    for name, method in igraph_methods:
        print(f"Running {name} ...", flush=True)
        t0 = time.time()
        comm = _igraph_person_communities(attr_edges, people, method)
        evals = _evaluate_people_sets(list(comm.values()), attr_qualities,
                                      attr_edges, n_total, quality_df, edge_df,
                                      args.max_picks, args.min_lift,
                                      args.min_fraction, args.max_steps_cluster,
                                      min_members=2)
        results.append(_summarize(name, evals, n_total, time.time() - t0))
        print(f"  {len(comm):,} communities -> {len(evals):,} people-clusters",
              flush=True)

    out = {"num_entities": n_total, "num_vertices": n_vertices,
           "edge_predicates": list(EDGE_PREDICATES),
           "quality_predicates": list(QUALITY_PREDICATES),
           "algorithms": results}
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.output}\n", flush=True)

    hdr = ("algorithm", "n_clusters", "coverage", "mean_size", "max_size",
           "mean_qual_cov", "mean_edge_cov", "time_s")
    print(" | ".join(hdr))
    for s in results:
        print(" | ".join(str(s[k]) for k in (
            "algorithm", "n_clusters", "coverage", "mean_size", "max_size",
            "mean_quality_coverage", "mean_edge_coverage", "time_s")))


if __name__ == "__main__":
    main()
