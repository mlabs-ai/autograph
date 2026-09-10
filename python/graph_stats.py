#!/usr/bin/env python3
"""Report node/edge statistics for a filtered Wikidata dump.

Counts, for a graph built from a `person_career.json.bz2`-style file:

* source entities (objects present in the array),
* typed edges (retained statements whose value is an entity QID),
* distinct target entities referenced by those edges,
* total distinct nodes (source QIDs union target QIDs),
* per-predicate edge counts.

Streams with lbzip2 and fans JSON parsing across a process pool.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor

ALLOW = ("P106", "P39", "P69", "P108", "P463", "P1344", "P54", "P102", "P101", "P31")


def _count_batch(lines):
    n_ent = 0
    n_stmt = 0      # retained statements (across target props + P31)
    n_edge = 0      # statements with an entity-typed value (a typed edge)
    n_nonentity = 0 # retained statements whose value is not an entity id
    src = set()
    tgt = set()
    pred = Counter()
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
        n_ent += 1
        src.add(qid)
        for pid, stmts in claims.items():
            if not isinstance(pid, str) or not isinstance(stmts, list):
                continue
            for stmt in stmts:
                n_stmt += 1
                snak = stmt.get("mainsnak", {}) if isinstance(stmt, dict) else {}
                if snak.get("snaktype") != "value":
                    continue
                val = snak.get("datavalue", {}).get("value", {})
                eid = val.get("id") if isinstance(val, dict) else None
                if eid:
                    n_edge += 1
                    tgt.add(eid)
                    pred[pid] += 1
                else:
                    n_nonentity += 1
    return n_ent, n_stmt, n_edge, n_nonentity, src, tgt, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--io-procs", type=int, default=4)
    ap.add_argument("--chunk-lines", type=int, default=1024)
    args = ap.parse_args()

    lb = shutil.which("lbzip2") or shutil.which("bzip2")
    decomp = subprocess.Popen(
        [lb, "-dc", "-n", str(args.io_procs), args.input],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    pool = ProcessPoolExecutor(max_workers=args.procs)
    pending = deque()
    max_inflight = args.procs * 2

    n_ent = n_stmt = n_edge = n_nonentity = 0
    nodes = set()      # source ∪ target
    targets = set()    # distinct targets only
    pred = Counter()

    buf = []
    try:
        for raw in decomp.stdout:
            buf.append(raw.decode("utf-8", "replace").rstrip("\n"))
            if len(buf) >= args.chunk_lines:
                pending.append(pool.submit(_count_batch, buf))
                buf = []
            while len(pending) >= max_inflight:
                e, s, g, nn, src, tgt, p = pending.popleft().result()
                n_ent += e; n_stmt += s; n_edge += g; n_nonentity += nn
                nodes.update(src); nodes.update(tgt); targets.update(tgt)
                pred.update(p)
        if buf:
            pending.append(pool.submit(_count_batch, buf))
        while pending:
            e, s, g, nn, src, tgt, p = pending.popleft().result()
            n_ent += e; n_stmt += s; n_edge += g; n_nonentity += nn
            nodes.update(src); nodes.update(tgt); targets.update(tgt)
            pred.update(p)
    finally:
        pool.shutdown(cancel_futures=True)
        decomp.stdout.close()
        decomp.terminate()
        decomp.wait()

    print("Graph statistics")
    print("=" * 40)
    print(f"source entities (nodes present):  {n_ent:,}")
    print(f"typed edges (entity -> entity):   {n_edge:,}")
    print(f"distinct target entities:         {len(targets):,}")
    print(f"total distinct nodes (union):     {len(nodes):,}")
    print(f"retained statements (all):        {n_stmt:,}")
    print(f"  non-entity-valued statements:   {n_nonentity:,}")
    print("edges per predicate:")
    for pid, c in pred.most_common():
        print(f"  {pid:8s} {c:,}")


if __name__ == "__main__":
    main()
