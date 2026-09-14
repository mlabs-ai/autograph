#!/usr/bin/env python3
"""Semantic divisive clustering with coherence-based early termination.

Instead of a blind `min_cluster_size`, the clustering recursion stops as soon as
a region is already explained by a dominant quality / dominant edge.  This is
the "interleaved" formulation: the dominant-qualities/edges evaluation is the
*split driver* and the *stop criterion* simultaneously.

Roles (same as `evaluate_clusters.py`):

* **Edges**     (P39, P69, P108, P463, P1344, P54, P102, P101) -- used to link
  and (by shared target) to split clusters.
* **Qualities** (P31, P106) -- used to *describe* a cluster and, when they
  partition cleanly, to split it too.

Algorithm (top-down, binary):

    solve(cluster):
        1. if |cluster| <= min_size:  emit leaf.
        2. compute dominant qualities and dominant edges (greedy, idf-weighted).
        3. if either explains >= coverage_target of members (with lift >= min_lift):
              emit this cluster with those explanations -- STOP, do not split.
        4. else: pick the single most informative key (quality tag or shared
              edge target) by `coverage x idf`, split members by "has that key",
              and recurse on both sides.

This produces disjoint clusters; each leaf is a set of members sharing a small,
discriminating explanation.  It is O(V x attrs x depth) overall (depth bounded by
`max_depth`) and does not need the quadratic-in-cluster-count M1 recursion.
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

QUALITY_PREDICATES = ("P31", "P106")
EDGE_PREDICATES = ("P39", "P69", "P108", "P463", "P1344", "P54", "P102", "P101", "P2868")


def _entity_values(claims, pred):
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


def _role_values(claims):
    """P2868 (subject has role) is a *qualifier*: it appears inside other
    statements' `qualifiers`, not as a top-level claim. Collect those role ids."""
    vals = []
    for _pid, stmts in claims.items():
        for stmt in stmts:
            if not isinstance(stmt, dict):
                continue
            quals = stmt.get("qualifiers", {}) or {}
            for snak in quals.get("P2868", []) or []:
                if not isinstance(snak, dict):
                    continue
                if snak.get("snaktype") != "value":
                    continue
                val = (snak.get("datavalue", {}) or {}).get("value", {}) or {}
                eid = val.get("id") if isinstance(val, dict) else None
                if eid:
                    vals.append(eid)
    return vals


def _extract_batch(lines):
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
            vals = _role_values(claims) if pred == "P2868" else _entity_values(claims, pred)
            if vals:
                edges[pred] = vals
        out.append({"qid": qid, "qualities": qualities, "edges": edges})
    return out


def _idf(dfv, n_total):
    return math.log((n_total + 1.0) / (dfv + 1.0))


def _top_keys(counts, df, n_total, n, max_picks, min_lift, min_fraction):
    """Rank (key, count) by coverage x idf; return kept keys with coverage."""
    rows = []
    for k, c in counts.items():
        frac = c / n
        if frac < min_fraction:
            continue
        dfv = df.get(k, 0)
        idf = _idf(dfv, n_total)
        lift = (frac / (dfv / n_total)) if dfv else float("inf")
        rows.append((c * idf, k, c, frac, lift))
    rows.sort(key=lambda t: -t[0])
    kept = []
    for score, k, c, frac, lift in rows:
        if len(kept) >= max_picks:
            break
        if lift < min_lift:
            continue
        kept.append((k, c, frac, lift))
    return kept


def _union_coverage(member_keys, kept_keys, n):
    ks = set(k for (k, _c, _f, _l) in kept_keys)
    if not ks:
        return 0.0
    covered = sum(1 for (_m, keys) in member_keys if any(k in ks for k in keys))
    return covered / n





def _mean_steps(keys, members, attr_edges, n_cap):
    """Mean hop-distance to each dominant edge target, as a dict {(pred,t): steps}."""
    if len(members) > n_cap:
        return None
    adj = {}
    for m in members:
        adj.setdefault(m, [])
        for tgts in attr_edges.get(m, {}).values():
            for t in tgts:
                adj.setdefault(t, [])
                adj[m].append(t)
                adj[t].append(m)
    res = {}
    for (pred, t) in keys:
        steps = float("inf")
        if t in adj:
            dist = {t: 0}
            dq = deque([t])
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        dq.append(v)
            ds = [dist[m] for m in members if m in dist]
            if ds:
                steps = sum(ds) / len(ds)
        res[(pred, t)] = None if steps == float("inf") else round(steps, 2)
    return res


class Clusterer:
    def __init__(self, attr_qualities, attr_edges, quality_df, edge_df, args):
        self.aq = attr_qualities
        self.ae = attr_edges
        self.qdf = quality_df
        self.edf = edge_df
        self.n_total = len(attr_qualities)
        self.a = args
        self.out = []
        self.depth = 0

    def _build_counts(self, members):
        quality_counts = {}
        edge_counts = {}
        member_q = []
        member_e = []
        for m in members:
            qkeys = list(self.aq.get(m, ()))
            for k in qkeys:
                quality_counts[k] = quality_counts.get(k, 0) + 1
            member_q.append((m, qkeys))
            ekeys = []
            for pred, tgts in self.ae.get(m, {}).items():
                for t in tgts:
                    k = (pred, t)
                    ekeys.append(k)
                    edge_counts[k] = edge_counts.get(k, 0) + 1
            member_e.append((m, ekeys))
        return quality_counts, edge_counts, member_q, member_e

    def _specific(self, rows, df):
        """Top dominant key must not be globally near-universal (e.g. 'human')."""
        if not rows:
            return False
        key = rows[0][0]
        return df.get(key, 0) / self.n_total <= self.a.max_df_frac

    def _pick_split_keys(self, quality_counts, edge_counts, n):
        """Top `fanout` quality/edge keys by coverage x idf, with non-trivial splits
        (a key must leave at least `min_size` members on both sides)."""
        cands = []
        for counts, df in ((quality_counts, self.qdf), (edge_counts, self.edf)):
            for k, c in counts.items():
                if c < self.a.min_size or c > n - self.a.min_size:
                    continue
                cands.append((c * _idf(df.get(k, 0), self.n_total), k))
        cands.sort(key=lambda t: -t[0])
        return cands[:self.a.fanout]

    def solve(self, members):
        n = len(members)
        if n <= self.a.min_size:
            self._emit(members, [], [], 0.0, 0.0, truncated=False)
            return

        quality_counts, edge_counts, member_q, member_e = self._build_counts(members)

        qual_rows = _top_keys(quality_counts, self.qdf, self.n_total, n,
                              self.a.max_picks, self.a.min_lift, self.a.min_fraction)
        edge_rows = _top_keys(edge_counts, self.edf, self.n_total, n,
                              self.a.max_picks, self.a.min_lift, self.a.min_fraction)
        qual_cov = _union_coverage(member_q, qual_rows, n)
        edge_cov = _union_coverage(member_e, edge_rows, n)

        q_ok = qual_cov >= self.a.coverage_target and self._specific(qual_rows, self.qdf)
        e_ok = edge_cov >= self.a.coverage_target and self._specific(edge_rows, self.edf)

        if q_ok or e_ok:
            self._emit(members, qual_rows, edge_rows, qual_cov, edge_cov,
                       truncated=False)
            return
        if self.depth >= self.a.max_depth:
            self._emit(members, qual_rows, edge_rows, qual_cov, edge_cov,
                       truncated=True)
            return

        split_keys = self._pick_split_keys(quality_counts, edge_counts, n)
        if not split_keys:
            self._emit(members, qual_rows, edge_rows, qual_cov, edge_cov,
                       truncated=True)
            return

        key_score = {k: s for s, k in split_keys}
        groups = {}
        remainder = []
        for (m, qkeys), (_m, ekeys) in zip(member_q, member_e):
            best_key = None
            best_score = -1.0
            for k in qkeys:
                s = key_score.get(k)
                if s is not None and s > best_score:
                    best_key, best_score = k, s
            for k in ekeys:
                s = key_score.get(k)
                if s is not None and s > best_score:
                    best_key, best_score = k, s
            if best_key is None:
                remainder.append(m)
            else:
                groups.setdefault(best_key, []).append(m)

        self.depth += 1
        try:
            for g in groups.values():
                self.solve(g)
            if remainder:
                self.solve(remainder)
        finally:
            self.depth -= 1

    def _emit(self, members, qual_rows, edge_rows, qual_cov, edge_cov, truncated):
        q = [
            {"predicate": p, "value": v, "count": c, "fraction": round(f, 4),
             "lift": None if math.isinf(l) else round(l, 2)}
            for (p, v), c, f, l in qual_rows
        ]
        steps = _mean_steps([(p, v) for (p, v), _c, _f, _l in edge_rows],
                            members, self.ae, self.a.max_steps_cluster)
        e = []
        for (p, v), c, f, l in edge_rows:
            e.append({"predicate": p, "value": v, "count": c,
                      "fraction": round(f, 4),
                      "lift": None if math.isinf(l) else round(l, 2),
                      "steps": (steps.get((p, v)) if steps else None)})
        self.out.append({
            "size": len(members),
            "num_qualities": len(q),
            "quality_coverage": round(qual_cov, 4),
            "dominant_qualities": q,
            "num_edges": len(e),
            "edge_coverage": round(edge_cov, 4),
            "dominant_edges": e,
            "truncated": truncated,
        })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--io-procs", type=int, default=4)
    ap.add_argument("--chunk-lines", type=int, default=1024)
    ap.add_argument("--min-size", type=int, default=20,
                    help="stop subdividing clusters below this many members")
    ap.add_argument("--max-depth", type=int, default=40)
    ap.add_argument("--fanout", type=int, default=200,
                    help="how many keys a node may split into at once")
    ap.add_argument("--coverage-target", type=float, default=0.6,
                    help="union coverage a cluster must reach to be 'coherent'")
    ap.add_argument("--min-lift", type=float, default=2.0)
    ap.add_argument("--min-fraction", type=float, default=0.02)
    ap.add_argument("--max-df-frac", type=float, default=0.1,
                    help="a cluster's top dominant tag cannot exceed this global "
                         "frequency to count as a meaningful explanation")
    ap.add_argument("--min-split-frac", type=float, default=0.05)
    ap.add_argument("--max-split-frac", type=float, default=0.95)
    ap.add_argument("--max-picks", type=int, default=10)
    ap.add_argument("--max-steps-cluster", type=int, default=100_000)
    args = ap.parse_args()

    # ---- stream + collect attributes (interned) + global df ---------------
    print("Streaming & extracting qualities/edges...", flush=True)
    lb = shutil.which("lbzip2") or shutil.which("bzip2")
    decomp = subprocess.Popen([lb, "-dc", "-n", str(args.io_procs), args.input],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    attr_qualities = {}
    attr_edges = {}
    quality_df = {}
    edge_df = {}
    pool = ProcessPoolExecutor(max_workers=args.procs)
    pending = deque()
    buf = []
    seen = 0
    t0 = time.time()

    def drain():
        for rec in pending.popleft().result():
            qid = sys.intern(rec["qid"])
            qs = frozenset((sys.intern(p), sys.intern(v)) for p, v in rec["qualities"])
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

    # ---- recursive semantic clustering ------------------------------------
    print("Recursive semantic clustering (coherence early-stop)...", flush=True)
    t0 = time.time()
    members = list(attr_qualities.keys())
    cl = Clusterer(attr_qualities, attr_edges, quality_df, edge_df, args)
    cl.solve(members)
    clusters = cl.out
    clusters.sort(key=lambda c: -c["size"])
    print(f"  {len(clusters):,} clusters ({time.time() - t0:.1f}s)", flush=True)

    result = {"num_clusters": len(clusters), "num_entities": n_total,
              "clusters": clusters}
    with open(args.output, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"Wrote {args.output}", flush=True)

    print("\nTop clusters (largest first):", flush=True)
    for c in clusters[:20]:
        qs = "; ".join(f"{r['predicate']}:{r['value']}@{r['fraction']:.0%}"
                       for r in c["dominant_qualities"][:3])
        es = "; ".join(f"{r['predicate']}->{r['value']}({r['count']},d={r['steps']})"
                       for r in c["dominant_edges"][:3])
        print(f"\ncluster ({c['size']} members, {'TRUNC' if c['truncated'] else 'ok'}):", flush=True)
        print(f"  qualities({c['num_qualities']},cov={c['quality_coverage']:.0%}): {qs}", flush=True)
        print(f"  edges({c['num_edges']},cov={c['edge_coverage']:.0%}): {es}", flush=True)


if __name__ == "__main__":
    main()
