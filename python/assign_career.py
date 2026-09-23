#!/usr/bin/env python3
"""Run frame-of-reference assignment (Milestone 2) on the filtered career graph.

Ingests a `person_career.json.bz2`-style dump (entities + the career/membership
predicates + instance_of), discovers reference frames by clustering the union
graph, then assigns entities and predicates to those frames with
Expectation-Maximisation.

Memory is bounded: if the EM `theta` matrix (`num_frames x num_entities`, with a
3x peak during an update) would exceed the budget, the graph is re-clustered
with a coarser `min_cluster_size` to reduce the frame count.
"""

from autograph import autograph
import fire
import json
import time

# The predicates present in the filtered graph.
DEFAULT_PREDICATES = [
    "P106",  # occupation
    "P39",   # position held
    "P69",   # educated at
    "P108",  # employer
    "P463",  # member of
    "P1344", # participant in
    "P54",   # member of sports team
    "P102",  # member of political party
    "P101",  # field of work
    "P31",   # instance of
]

# Peak EM memory ~= 3 * num_frames * num_entities * 8 bytes (old theta + raw
# accumulator + new theta). Default budget 20 GiB.
DEFAULT_EM_BUDGET_GIB = 20.0


def _normalize_predicates(predicates):
    if predicates is None:
        return list(DEFAULT_PREDICATES)
    if isinstance(predicates, str):
        return [p.strip() for p in predicates.split(",") if p.strip()]
    return [str(p).strip() for p in predicates if str(p).strip()]


def main(
    graph_file: str,
    output_file: str,
    predicates=None,
    limit: int = None,
    factor: float = 0.01,
    steps: int = 5,
    boundary_threshold: float = 0.1,
    min_cluster_size: int = 10,
    epsilon: float = 1e-3,
    tol: float = 1e-6,
    max_iters: int = 100,
    em_budget_gib: float = DEFAULT_EM_BUDGET_GIB,
):
    preds = _normalize_predicates(predicates)

    print("Ingesting graph...", flush=True)
    start = time.time()
    graph = autograph.FrameGraph.from_wikidata_bz2(graph_file, preds, limit)
    print(
        f"  {graph.num_entities():,} entities, {graph.num_predicates()} predicates, "
        f"{graph.num_edges():,} edges ({time.time() - start:.1f}s)",
        flush=True,
    )

    num_entities = graph.num_entities()
    budget = em_budget_gib * 1024 ** 3

    print("Clustering...", flush=True)
    start = time.time()
    frames = graph.cluster(factor, steps, boundary_threshold, min_cluster_size)
    while 3 * len(frames) * num_entities * 8 > budget:
        min_cluster_size *= 2
        frames = graph.cluster(factor, steps, boundary_threshold, min_cluster_size)
        print(
            f"  re-cluster min_cluster_size={min_cluster_size} -> "
            f"{len(frames):,} frames (bound EM memory)",
            flush=True,
        )
    print(
        f"  {len(frames):,} frames "
        f"(theta ~ {3 * len(frames) * num_entities * 8 / 1024**3:.1f} GiB) "
        f"({time.time() - start:.1f}s)",
        flush=True,
    )

    print("Assigning entities & predicates (EM)...", flush=True)
    start = time.time()
    m = graph.em_assign(frames, epsilon, tol, max_iters)
    print(f"  done ({time.time() - start:.1f}s)", flush=True)

    predicate_labels = graph.predicate_labels()
    argmax_entities = m.argmax_entities()
    argmax_predicates = m.argmax_predicates()

    frame_sizes = [0] * m.num_frames()
    for f in argmax_entities:
        frame_sizes[f] += 1

    print("\nPredicates -> frames:", flush=True)
    for pid, name in enumerate(predicate_labels):
        print(f"  {name:6s} -> frame {argmax_predicates[pid]}", flush=True)
    print("Frames (size desc, top 20):", flush=True)
    for f, size in sorted(enumerate(frame_sizes), key=lambda t: -t[1])[:20]:
        print(f"  frame {f:4d}: {size:9,d} entities", flush=True)

    result = {
        "num_frames": m.num_frames(),
        "num_entities": m.num_entities(),
        "num_predicates": m.num_predicates(),
        "predicate_labels": predicate_labels,
        "predicate_frames": argmax_predicates,
        "frame_sizes": frame_sizes,
        "phi": m.phi,
    }
    with open(output_file, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"\nWrote memberships to {output_file}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
