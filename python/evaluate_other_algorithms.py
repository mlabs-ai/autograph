import argparse
import json
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from autograph import autograph
import igraph
from sklearn.metrics import adjusted_rand_score


# Resolution grid swept by the modularity-based methods (Leiden, Louvain).
# Each method reports the best ARI found over this grid, so every method is
# given the same opportunity to recover the planted partition.
RESOLUTION_GRID = [0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0]


def ari(true_ids, membership):
    return adjusted_rand_score(true_ids, membership)


def experiment_step(
    num_clusters: int,
    min_nodes: int,
    max_nodes: int,
    pepper_percent: float,
    salt_percent: float,
    random_seed: int,
    autograph_factor: float = 0.01,
    autograph_steps: int = 2,
) -> dict:
    """Run one experiment on a single seeded graph and return its scores.

    This is a module-level function (rather than a nested closure) so it can be
    pickled and shipped to worker processes under `ProcessPoolExecutor`.
    """
    # Start by generating a graph whose clustering is known
    random.seed(random_seed)

    clusters = [random.randint(min_nodes, max_nodes) for _ in range(num_clusters)]
    builder = autograph.GraphBuilder(random_seed)

    # Add clusters
    for cluster_size in clusters:
        new_edges_per_node = int(salt_percent * cluster_size)
        builder.add_scale_free_cluster(cluster_size, new_edges_per_node)

    # Add pepper
    for i in range(len(clusters)):
        j = (i + 1) % len(clusters)
        smaller_cluster_size = min(clusters[i], clusters[j])
        num_pepper_edges = int(pepper_percent * smaller_cluster_size)
        for _ in range(num_pepper_edges):
            builder.add_random_link(i, j)

    # Finalize graph
    graph = builder.finalize_graph()
    graph.shuffle_vertex_ids(random_seed)

    # Get the cluster ids for the base truth graph
    true_cluster_ids = []
    for i, cluster_size in enumerate(clusters):
        true_cluster_ids += [i] * cluster_size

    # Convert graph to iGraph format for use in other algorithms
    ig_graph = igraph.Graph()
    nodes = set()
    edges = graph.edge_list()
    for v1, v2 in edges:
        nodes.add(v1)
        nodes.add(v2)
    nodes = list(nodes)
    nodes.sort(key=lambda s: int(s))
    ig_graph.add_vertices(nodes)
    ig_graph.add_edges(edges)

    # Run our clustering algorithm
    starttime = time.time()
    graph.cluster(autograph_factor, autograph_steps, 0.1, 10)
    autograph_clusters = [0] * len(true_cluster_ids)
    for i, cluster in enumerate(graph.get_clusters()):
        for node_id in cluster:
            node_id = int(node_id)
            autograph_clusters[node_id] = i
    autograph_time = time.time() - starttime

    # --- Baselines ---------------------------------------------------------------
    # Louvain (multilevel) and Leiden are both modularity-based, so we sweep
    # the resolution parameter and report the best ARI over the sweep.
    starttime = time.time()
    louvain_best = 0.0
    for resolution in RESOLUTION_GRID:
        membership = ig_graph.community_multilevel(resolution=resolution).membership
        louvain_best = max(louvain_best, ari(true_cluster_ids, membership))
    louvain_time = time.time() - starttime

    starttime = time.time()
    leiden_best = 0.0
    for resolution in RESOLUTION_GRID:
        membership = ig_graph.community_leiden(
            objective_function="modularity", resolution=resolution
        ).membership
        leiden_best = max(leiden_best, ari(true_cluster_ids, membership))
    leiden_time = time.time() - starttime

    starttime = time.time()
    fast_greedy_membership = ig_graph.community_fastgreedy().as_clustering(
        n=num_clusters
    ).membership
    fast_greedy_time = time.time() - starttime

    starttime = time.time()
    walktrap_membership = ig_graph.community_walktrap().as_clustering(
        n=num_clusters
    ).membership
    walktrap_time = time.time() - starttime

    starttime = time.time()
    infomap_membership = ig_graph.community_infomap().membership
    infomap_time = time.time() - starttime

    # Calculate distance scores
    return {
        "seed": random_seed,
        "autograph": ari(true_cluster_ids, autograph_clusters),
        "autograph_time": autograph_time,
        "louvain": louvain_best,
        "louvain_time": louvain_time,
        "leiden": leiden_best,
        "leiden_time": leiden_time,
        "fast_greedy": ari(true_cluster_ids, fast_greedy_membership),
        "fast_greedy_time": fast_greedy_time,
        "walktrap": ari(true_cluster_ids, walktrap_membership),
        "walktrap_time": walktrap_time,
        "infomap": ari(true_cluster_ids, infomap_membership),
        "infomap_time": infomap_time,
        "num_vertices": graph.num_vertices(),
        "num_edges": graph.num_edges(),
    }


def _run_step(args):
    """Wrapper for parallel execution; returns (seed, scores_or_error)."""
    seed, params, rayon_threads = args
    if rayon_threads is not None:
        import os

        os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)
    try:
        return seed, experiment_step(
            params["num_clusters"],
            params["min_nodes"],
            params["max_nodes"],
            params["pepper"],
            params["salt"],
            seed,
            autograph_factor=params["autograph_factor"],
            autograph_steps=params["autograph_steps"],
        )
    except Exception as exc:
        return seed, f"ERROR: {exc}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Autograph against igraph clustering algorithms."
    )
    parser.add_argument(
        "--rayon-threads",
        type=int,
        default=None,
        help="Set RAYON_NUM_THREADS for Autograph's internal parallelism. "
        "Default leaves rayon to auto-detect CPU count.",
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of seeded runs (default 10)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel processes (default: CPU count)",
    )
    parser.add_argument("--clusters", type=int, default=150, help="Planted clusters")
    parser.add_argument("--min-nodes", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=200)
    parser.add_argument("--pepper", type=float, default=0.7)
    parser.add_argument("--salt", type=float, default=0.1)
    parser.add_argument(
        "--autograph-factor",
        type=float,
        default=0.01,
        help="Autograph cluster factor (default 0.01)",
    )
    parser.add_argument(
        "--autograph-steps",
        type=int,
        default=2,
        help="Autograph steps_before_subdivide (default 2, was 5)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run in a single process (default: parallel)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to write full JSON results"
    )
    args = parser.parse_args()

    params = {
        "num_clusters": args.clusters,
        "min_nodes": args.min_nodes,
        "max_nodes": args.max_nodes,
        "pepper": args.pepper,
        "salt": args.salt,
        "autograph_factor": args.autograph_factor,
        "autograph_steps": args.autograph_steps,
    }
    seeds = list(range(args.iterations))

    results = {}
    failures = []

    if args.sequential or args.workers == 1:
        for seed in seeds:
            print(f"Running seed {seed} ({seed + 1}/{args.iterations})", flush=True)
            seed, score = _run_step((seed, params, args.rayon_threads))
            if isinstance(score, str):
                print(score, flush=True)
                failures.append(seed)
            else:
                results[seed] = score
    else:
        workers = args.workers
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_step, (seed, params, args.rayon_threads)): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    seed, score = future.result()
                except Exception as exc:
                    print(f"Seed {seed} crashed unexpectedly: {exc}")
                    failures.append(seed)
                    continue

                if isinstance(score, str):
                    print(f"Seed {seed} failed:\n{score}", flush=True)
                    failures.append(seed)
                else:
                    results[seed] = score
                    print(f"Seed {seed} done", flush=True)

    # Aggregate
    all_scores = list(results.values())
    if not all_scores:
        print("No successful runs.")
        return

    metric_keys = [k for k in all_scores[0] if k != "seed"]
    average_scores = {
        k: sum(s[k] for s in all_scores) / len(all_scores) for k in metric_keys
    }

    summary = {
        "n_runs": len(all_scores),
        "failures": failures,
        "averages": average_scores,
    }
    print(json.dumps(summary, indent=4))

    if args.output:
        # Preserve raw field order for readability
        ordered = {k: average_scores[k] for k in metric_keys}
        with open(args.output, "w") as f:
            json.dump(ordered, f, indent=4)
        print(f"Wrote averages to {args.output}")


if __name__ == "__main__":
    main()
