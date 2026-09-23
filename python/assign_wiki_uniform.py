from autograph import autograph
import bz2
import fire
import json
import sys
import time

DEFAULT_PREDICATES = ["P106", "P108", "P69", "P185", "P101", "P50", "P921"]

# Peak EM memory is ~2 * num_frames * num_entities * 8 bytes (theta + its raw
# accumulator). Bound it so the run cannot OOM.
MAX_EM_BYTES = 12 * 1024 ** 3


def _normalize_predicates(predicates):
    if predicates is None:
        return list(DEFAULT_PREDICATES)
    if isinstance(predicates, str):
        return [p.strip() for p in predicates.split(",") if p.strip()]
    return [str(p).strip() for p in predicates if str(p).strip()]


def _target_ids(claims, pid):
    out = []
    for stmt in claims.get(pid, []):
        dst = (
            stmt.get("mainsnak", {})
            .get("datavalue", {})
            .get("value", {})
            .get("id")
        )
        if dst:
            out.append(dst)
    return out


def main(
    wikidata_json_file: str,
    output_file: str,
    head: int = 200_000,
    tail_stride: int = 100,
    predicates=None,
    max_sample: int = None,
    factor: float = 0.01,
    steps: int = 5,
    boundary_threshold: float = 0.1,
    min_cluster_size: int = 10,
    epsilon: float = 1e-3,
    tol: float = 1e-6,
    max_iters: int = 100,
):
    """
    Sample a Wikidata dump as: the first `head` entities in full, plus every
    `tail_stride`-th entity thereafter. Then run frame-of-reference assignment.

    These reads the entire dump (a full ~hour+ pass) but bounds memory so it
    cannot OOM: if the EM's theta matrix would exceed the memory budget, it
    re-clusters with a coarser `min_cluster_size`.
    """
    preds = _normalize_predicates(predicates)
    graph = autograph.FrameGraph()
    labels = {}  # QID -> English label

    seen = 0
    sampled = 0
    print("Streaming dump (head + every-Nth)...", flush=True)
    start = time.time()
    with bz2.open(wikidata_json_file, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            seen += 1
            # Keep: the first `head` entities, then every `tail_stride`-th.
            if not (seen <= head or seen % tail_stride == 0):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("id")
            if not qid:
                continue
            sampled += 1

            en = obj.get("labels", {}).get("en", {}).get("value")
            if en:
                labels[qid] = en
            claims = obj.get("claims", {})
            for pid in preds:
                for dst in _target_ids(claims, pid):
                    graph.add_edge(qid, pid, dst)

            if sampled % 10000 == 0:
                print(
                    f"  [sampled] {sampled:,} entities (read {seen:,} lines, "
                    f"{time.time() - start:.0f}s)",
                    file=sys.stderr,
                    flush=True,
                )
            if max_sample and sampled >= max_sample:
                break

    print(
        f"Sampled {sampled:,} entities (read {seen:,} lines, "
        f"{time.time() - start:.0f}s)",
        flush=True,
    )
    print(
        f"Graph: {graph.num_entities():,} entities, "
        f"{graph.num_predicates()} predicates, {graph.num_edges():,} edges",
        flush=True,
    )

    # Cluster, re-clustering with a coarser min_cluster_size if the EM would
    # overflow the memory budget.
    print("Clustering...", flush=True)
    frames = graph.cluster(factor, steps, boundary_threshold, min_cluster_size)
    num_entities = graph.num_entities()
    while 2 * len(frames) * num_entities * 8 > MAX_EM_BYTES:
        min_cluster_size *= 2
        frames = graph.cluster(
            factor, steps, boundary_threshold, min_cluster_size
        )
        print(
            f"  Re-clustering with min_cluster_size={min_cluster_size} "
            f"-> {len(frames):,} frames (to bound EM memory)",
            flush=True,
        )
    print(
        f"  {len(frames):,} frames (theta ~ "
        f"{2 * len(frames) * num_entities * 8 / 1024**3:.1f} GiB)",
        flush=True,
    )

    print("Assigning (EM)...", flush=True)
    start = time.time()
    m = graph.em_assign(frames, epsilon, tol, max_iters)
    print(f"  done ({time.time() - start:.1f}s)", flush=True)

    predicate_labels = graph.predicate_labels()
    entity_labels = graph.entity_labels()
    argmax_entities = m.argmax_entities()
    argmax_predicates = m.argmax_predicates()

    frame_sizes = [0] * m.num_frames()
    for f in argmax_entities:
        frame_sizes[f] += 1

    print("\nPredicates -> frames:", flush=True)
    for pid, name in enumerate(predicate_labels):
        print(f"  {name:6s} -> frame {argmax_predicates[pid]}", flush=True)
    print("Frames (size desc, top 10):", flush=True)
    for f, size in sorted(enumerate(frame_sizes), key=lambda t: -t[1])[:10]:
        print(f"  frame {f:4d}: {size:8d} entities", flush=True)

    result = {
        "num_frames": m.num_frames(),
        "num_entities": m.num_entities(),
        "num_predicates": m.num_predicates(),
        "predicate_labels": predicate_labels,
        "predicate_frames": argmax_predicates,
        "entity_frames": argmax_entities,
        "entity_labels": entity_labels,
        "entity_names": labels,
        "frame_sizes": frame_sizes,
        "phi": m.phi,
    }
    with open(output_file, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"\nWrote memberships to {output_file}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
