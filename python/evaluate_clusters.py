#!/usr/bin/env python3
"""Cluster a filtered Wikidata graph and greedily characterise each cluster.

Properties split into two roles:

* **Edges**     -- build the graph and cluster it:
    P39, P69, P108, P463, P1344, P54, P102, P101.
* **Qualities** -- evaluate the resulting clusters:
    P31 (instance of), P106 (occupation).

Pipeline:

1. Stream the dump (parallel, lbzip2), recording each entity's qualities and
   edge links, plus the global frequency (df) of every quality value and edge
   target.  Strings are interned so the full dump fits in bounded memory.
2. Build a `FrameGraph` from the edges only and cluster it (Milestone 1).
3. For every cluster, two fast greedy summaries.  Each candidate tag/link is
   scored by `coverage x idf`, where idf = log((N+1)/(df+1)); the top `--max-picks`
   are reported with their count, member fraction, and lift (cluster-frequency /
   global-frequency).  A tag/link is only reported if its lift >= `--min-lift`
   (i.e. it is genuinely more concentrated here than in the population).  Edges
   are additionally ranked by "steps" (mean hop-distance from the shared target
   to the cluster members), computed only for clusters up to `--max-steps-cluster`
   to stay fast on the full graph.

This is the fast greedy replacement for EM (`assign_career.py` / `assign_wiki.py`
keep the EM path).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

from autograph import autograph

QUALITY_PREDICATES = ("P31", "P106")
EDGE_PREDICATES = ("P39", "P69", "P108", "P463", "P1344", "P54", "P102", "P101")
ALL_PREDICATES = tuple(QUALITY_PREDICATES) + tuple(EDGE_PREDICATES)


def _entity_values(claims: dict, pred: str) -> list[str]:
    vals = []
    for stmt in claims.get(pred, []) or []:
        if not isinstance(stmt, dict):
            continue
        snak = stmt.get("mainsnak", {}) or {}
        if snak.get("snaktype") != "value":
            continue
        val = (snak.get("datavalue", {}) or {}).get("value", {}) or {}
        eid = val.get("id") if isinstance(val, dict) else None
        if eid:
            vals.append(eid)
    return vals


def _extract_batch(lines: list[str]) -> list[dict]:
    out = []
    for line in lines:
        line = line.strip()
        if not line or line == "[" or line == "]":
            continue
        line = line.rstrip(",")
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        qid = obj.get("id")
        claims = obj.get("claims")
        if not qid or not isinstance(claims, dict):
            continue

        qualities = []
        for pred in QUALITY_PREDICATES:
            for v in _entity_values(claims, pred):
                qualities.append((pred, v))

        edges = {}
        for pred in EDGE_PREDICATES:
            vals = _entity_values(claims, pred)
            if vals:
                edges[pred] = vals

        out.append({"qid": qid, "qualities": qualities, "edges": edges})
    return out


def _top_dominant(counts, df, n_total, n, max_picks, min_lift, min_fraction,
                  member_keys):
    """Rank candidates by `coverage x idf`, keep those that are both distinctive
    (lift >= min_lift) and cover a meaningful fraction (fraction >= min_fraction).

    Returns (rows, coverage) where rows = [(key, count, fraction, lift)] and
    coverage is the union fraction of members covered by the kept keys.
    """
    if n == 0:
        return [], 0.0

    scored = []
    for k, c in counts.items():
        frac = c / n
        if frac < min_fraction:
            continue
        dfv = df.get(k, 0)
        idf = math.log((n_total + 1.0) / (dfv + 1.0))
        lift = (frac / (dfv / n_total)) if dfv else float("inf")
        scored.append((c * idf, k, c, lift))
    scored.sort(key=lambda t: -t[0])

    kept_keys = []
    rows = []
    for score, k, c, lift in scored:
        if len(rows) >= max_picks:
            break
        if min_lift is not None and lift < min_lift:
            continue
        rows.append((k, c, c / n, lift))
        kept_keys.append(k)

    ks = set(kept_keys)
    covered = sum(1 for (_m, keys) in member_keys if any(k in ks for k in keys))
    return rows, (covered / n)


def _union_adjacency(cluster_members, attr_edges):
    adj = {}
    for m in cluster_members:
        adj.setdefault(m, [])
        for tgts in attr_edges.get(m, {}).values():
            for t in tgts:
                adj.setdefault(t, [])
                adj[m].append(t)
                adj[t].append(m)
    return adj


def mean_steps(target, cluster_members, adj):
    if target not in adj:
        return float("inf")
    dist = {target: 0}
    dq = deque([target])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    ds = [dist[m] for m in cluster_members if m in dist]
    return (sum(ds) / len(ds)) if ds else float("inf")


def evaluate_cluster(member_qids, attr_qualities, attr_edges,
                     n_total, quality_df, edge_df,
                     max_picks, min_lift, min_fraction, max_steps_cluster):
    members = [m for m in member_qids if m in attr_qualities or m in attr_edges]
    n = len(members)

    quality_counts = {}
    edge_counts = {}
    member_q = []   # (member, [quality keys])
    member_e = []   # (member, [edge keys])
    for m in members:
        qkeys = list(attr_qualities.get(m, ()))
        for k in qkeys:
            quality_counts[k] = quality_counts.get(k, 0) + 1
        member_q.append((m, qkeys))

        ekeys = []
        for pred, tgts in attr_edges.get(m, {}).items():
            for t in tgts:
                k = (pred, t)
                ekeys.append(k)
                edge_counts[k] = edge_counts.get(k, 0) + 1
        member_e.append((m, ekeys))

    qual_rows, qual_cov = _top_dominant(quality_counts, quality_df, n_total, n,
                                        max_picks, min_lift, min_fraction,
                                        member_q)
    edge_rows, edge_cov = _top_dominant(edge_counts, edge_df, n_total, n,
                                        max_picks, min_lift, min_fraction,
                                        member_e)

    quality_out = [
        {"predicate": p, "value": v, "count": c, "fraction": round(f, 4),
         "lift": None if math.isinf(l) else round(l, 2)}
        for (p, v), c, f, l in qual_rows
    ]
    edge_out = []
    if edge_rows and n <= max_steps_cluster:
        adj = _union_adjacency(members, attr_edges)
        for (p, t), c, f, l in edge_rows:
            edge_out.append({
                "predicate": p, "value": t, "count": c,
                "fraction": round(f, 4),
                "lift": None if math.isinf(l) else round(l, 2),
                "steps": round(mean_steps(t, members, adj), 2),
            })
        edge_out.sort(key=lambda r: (-r["count"], r["steps"]))
    else:
        for (p, t), c, f, l in edge_rows:
            edge_out.append({
                "predicate": p, "value": t, "count": c,
                "fraction": round(f, 4),
                "lift": None if math.isinf(l) else round(l, 2),
                "steps": None,
            })
        edge_out.sort(key=lambda r: -r["count"])

    return {
        "size": n,
        "num_qualities": len(quality_out),
        "quality_coverage": round(qual_cov, 4),
        "dominant_qualities": quality_out,
        "num_edges": len(edge_out),
        "edge_coverage": round(edge_cov, 4),
        "dominant_edges": edge_out,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--io-procs", type=int, default=4)
    ap.add_argument("--chunk-lines", type=int, default=1024)
    ap.add_argument("--factor", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--boundary-threshold", type=float, default=0.1)
    ap.add_argument("--min-cluster-size", type=int, default=10)
    ap.add_argument("--max-picks", type=int, default=10)
    ap.add_argument("--min-lift", type=float, default=2.0,
                    help="minimum lift to report a tag/link as dominant")
    ap.add_argument("--min-fraction", type=float, default=0.02,
                    help="minimum fraction of members a tag/link must cover")
    ap.add_argument("--max-steps-cluster", type=int, default=100_000,
                    help="skip BFS 'steps' for clusters larger than this")
    ap.add_argument("--min-cluster-members", type=int, default=2)
    args = ap.parse_args()

    # ---- Stage 1: stream, collect attributes (interned) + global df --------
    print("Streaming & extracting qualities/edges...", flush=True)
    lb = shutil.which("lbzip2") or shutil.which("bzip2")
    decomp = subprocess.Popen(
        [lb, "-dc", "-n", str(args.io_procs), args.input],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    attr_qualities = {}   # qid -> frozenset of (pred, value)
    attr_edges = {}       # qid -> {pred: [target, ...]}
    quality_df = {}       # (pred, value) -> global entity count
    edge_df = {}          # (pred, target) -> global entity count
    pool = ProcessPoolExecutor(max_workers=args.procs)
    pending = deque()
    buf = []
    seen = 0
    t0 = time.time()

    def drain():
        for rec in pending.popleft().result():
            qid = sys.intern(rec["qid"])
            qs = frozenset((sys.intern(pred), sys.intern(val))
                           for pred, val in rec["qualities"])
            for k in qs:
                quality_df[k] = quality_df.get(k, 0) + 1
            attr_qualities[qid] = qs

            edges = rec["edges"]
            if edges:
                e2 = {}
                for pred, tgts in edges.items():
                    ip = sys.intern(pred)
                    it = [sys.intern(t) for t in sorted(set(tgts))]
                    e2[ip] = it
                    for t in it:
                        k = (ip, t)
                        edge_df[k] = edge_df.get(k, 0) + 1
                attr_edges[qid] = e2

    try:
        for raw in decomp.stdout:
            buf.append(raw.decode("utf-8", "replace").rstrip("\n"))
            seen += 1
            if len(buf) >= args.chunk_lines:
                pending.append(pool.submit(_extract_batch, buf))
                buf = []
            while len(pending) >= args.procs * 2:
                drain()
            if args.limit and seen >= args.limit:
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

    n_total = len(attr_qualities)
    print(f"  {n_total:,} entities ({time.time() - t0:.1f}s)", flush=True)

    # ---- Stage 2: build graph (edges only) and cluster --------------------
    print("Building graph & clustering (edges only)...", flush=True)
    t0 = time.time()
    graph = autograph.FrameGraph()
    for qid, edges in attr_edges.items():
        for pred, tgts in edges.items():
            for t in tgts:
                graph.add_edge(qid, pred, t)
    print(f"  {graph.num_edges():,} edges ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    frames = graph.cluster(args.factor, args.steps, args.boundary_threshold,
                           args.min_cluster_size)
    print(f"  {len(frames):,} frames ({time.time() - t0:.1f}s)", flush=True)

    # ---- Stage 3: evaluate ------------------------------------------------
    print("Evaluating clusters (greedy dominant qualities & edges)...", flush=True)
    t0 = time.time()
    clusters = []
    for ci, frame in enumerate(frames):
        members = [m for m in frame if m in attr_qualities or m in attr_edges]
        if len(members) < args.min_cluster_members:
            continue
        ev = evaluate_cluster(members, attr_qualities, attr_edges,
                              n_total, quality_df, edge_df,
                              args.max_picks, args.min_lift, args.min_fraction,
                              args.max_steps_cluster)
        ev["cluster_id"] = ci
        clusters.append(ev)
    clusters.sort(key=lambda c: -c["size"])
    print(f"  {len(clusters):,} clusters evaluated ({time.time() - t0:.1f}s)",
          flush=True)

    result = {"num_clusters": len(clusters), "num_entities": n_total,
              "clusters": clusters}
    with open(args.output, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"Wrote {args.output}", flush=True)

    print("\nTop clusters (largest first):", flush=True)
    for c in clusters[:15]:
        qs = "; ".join(f"{r['predicate']}:{r['value']}@{r['fraction']:.0%}"
                       for r in c["dominant_qualities"][:3])
        es = "; ".join(f"{r['predicate']}->{r['value']}({r['count']},d={r['steps']})"
                       for r in c["dominant_edges"][:3])
        print(f"\ncluster {c['cluster_id']}: {c['size']} members", flush=True)
        print(f"  qualities({c['num_qualities']},cov={c['quality_coverage']:.0%}): {qs}",
              flush=True)
        print(f"  edges({c['num_edges']},cov={c['edge_coverage']:.0%}): {es}", flush=True)


if __name__ == "__main__":
    main()
