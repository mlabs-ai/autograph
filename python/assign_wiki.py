from autograph import autograph
import fire
import json
import time

# The Milestone-2 "knowledge production + careers" predicate ontology.
DEFAULT_PREDICATES = ["P106", "P108", "P69", "P185", "P101", "P50", "P921"]


def _normalize_predicates(predicates):
    """fire may pass a comma-string, a tuple, or a list; normalize to list[str]."""
    if predicates is None:
        return list(DEFAULT_PREDICATES)
    if isinstance(predicates, str):
        return [p.strip() for p in predicates.split(",") if p.strip()]
    return [str(p).strip() for p in predicates if str(p).strip()]


def main(
    wikidata_json_file: str,
    output_file: str,
    predicates=None,
    frames_file: str = None,
    factor: float = 0.01,
    steps: int = 5,
    boundary_threshold: float = 0.1,
    min_cluster_size: int = 10,
    limit: int = None,
    epsilon: float = 1e-3,
    tol: float = 1e-6,
    max_iters: int = 100,
):
    """
    Ingests a multi-predicate Wikidata dump, discovers reference frames, and
    assigns entities and predicates to those frames via Expectation-Maximisation.

    Args:
        wikidata_json_file: Path to a (possibly bzip2-compressed) Wikidata dump.
        output_file: Where to write the memberships JSON.
        predicates: Comma-separated Wikidata property IDs to ingest.
        frames_file: Optional JSON of pre-computed frames (list of lists of
            entity QIDs, e.g. the output of cluster_wiki.py). If given, this is
            used instead of re-clustering.
        limit: Cap the number of dump entities ingested (use on the full dump).
    """
    pred_list = _normalize_predicates(predicates)

    print("Ingesting graph...", flush=True)
    start = time.time()
    graph = autograph.FrameGraph.from_wikidata_bz2(
        wikidata_json_file, pred_list, limit
    )
    print(
        f"  {graph.num_entities()} entities, {graph.num_predicates()} predicates, "
        f"{graph.num_edges()} edges ({time.time() - start:.1f}s)",
        flush=True,
    )

    if frames_file:
        print(f"Loading pre-computed frames from {frames_file}", flush=True)
        with open(frames_file) as f:
            frames = json.load(f)
        print(f"  {len(frames)} frames", flush=True)
    else:
        print("Clustering...", flush=True)
        start = time.time()
        frames = graph.cluster(factor, steps, boundary_threshold, min_cluster_size)
        print(f"  {len(frames)} frames ({time.time() - start:.1f}s)", flush=True)

    print("Assigning entities & predicates (EM)...", flush=True)
    start = time.time()
    m = graph.em_assign(frames, epsilon, tol, max_iters)
    print(f"  done ({time.time() - start:.1f}s)", flush=True)

    predicate_labels = graph.predicate_labels()
    entity_labels = graph.entity_labels()
    argmax_entities = m.argmax_entities()
    argmax_predicates = m.argmax_predicates()

    # Frame sizes from the argmax entity assignment.
    frame_sizes = [0] * m.num_frames()
    for f in argmax_entities:
        frame_sizes[f] += 1

    # Human-readable summary.
    print("\nPredicates -> frames:", flush=True)
    for pid, name in enumerate(predicate_labels):
        print(f"  {name:6s} -> frame {argmax_predicates[pid]}", flush=True)
    print("\nFrames (size desc):", flush=True)
    for f, size in sorted(enumerate(frame_sizes), key=lambda t: -t[1]):
        print(f"  frame {f:4d}: {size:6d} entities", flush=True)

    result = {
        "num_frames": m.num_frames(),
        "num_entities": m.num_entities(),
        "num_predicates": m.num_predicates(),
        "predicate_labels": predicate_labels,
        "predicate_frames": argmax_predicates,
        "entity_frames": argmax_entities,
        "entity_labels": entity_labels,
        "frame_sizes": frame_sizes,
        "phi": m.phi,
    }

    with open(output_file, "w") as f:
        json.dump(result, f)
    print(f"\nWrote memberships to {output_file}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
