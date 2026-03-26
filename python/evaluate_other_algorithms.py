from autograph import autograph
import igraph
import json
import random
from sklearn.metrics import adjusted_rand_score
import time
import matplotlib.pyplot as plt

def experiment_step(
    num_clusters: int,
    min_nodes: int,
    max_nodes: int,
    pepper_percent: float,
    salt_percent: float,
    random_seed: int
) -> dict[str, float]:
    # Start by generating a graph whose clustering is known
    print("Building graph...")
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
        true_cluster_ids += ([i] * cluster_size)

    # Convert graph to iGraph format for use in other algorithms
    print("Converting to iGraph format")
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
    print("Calculating clusters using Autograph")
    starttime = time.time()
    graph.cluster(0.01, 5, 0.1, 10)
    autograph_clusters = [0] * len(true_cluster_ids)
    for i, cluster in enumerate(graph.get_clusters()):
        for node_id in cluster:
            node_id = int(node_id)
            autograph_clusters[node_id] = i
    autograph_time = time.time() - starttime

    # Calculate communities using different methods
    print("Calculating clusters using eigenvector method")
    starttime = time.time()
    eigenvector_clusters = ig_graph.community_leading_eigenvector().membership
    eigenvector_time = time.time() - starttime
    
    print("Calculating clusters using Louvain method")
    starttime = time.time()
    louvain_clusters = ig_graph.community_multilevel().membership
    louvain_time = time.time() - starttime

    print("Calculating clusters using Leiden method")
    starttime = time.time()
    leiden_clusters = ig_graph.community_leiden().membership
    leiden_time = time.time() - starttime

    print("Calculating clusters using random walk method")
    starttime = time.time()
    random_walk_clusters = ig_graph.community_walktrap().as_clustering().membership
    random_walk_time = time.time() - starttime

    print("Calculating clusters using fast greedy method")
    starttime = time.time()
    fast_greedy_clusters = ig_graph.community_fastgreedy().as_clustering().membership
    fast_greedy_time = time.time() - starttime

    print("Calculating clusters using walktrap method")
    starttime = time.time()
    walktrap_clusters = ig_graph.community_walktrap().as_clustering().membership
    walktrap_time = time.time() - starttime

    print("Calculating clusters using infomap method")
    starttime = time.time()
    infomap_clusters = ig_graph.community_walktrap().as_clustering().membership
    infomap_time = time.time() - starttime

    # Calculate distance scores
    scores = {
        "autograph": adjusted_rand_score(true_cluster_ids, autograph_clusters),
        "autograph_time": autograph_time,
        "louvain": adjusted_rand_score(true_cluster_ids, louvain_clusters),
        "louvain_time": louvain_time,
        "leiden": adjusted_rand_score(true_cluster_ids, leiden_clusters),
        "leiden_time": leiden_time,
        "random_walk": adjusted_rand_score(true_cluster_ids, random_walk_clusters),
        "random_walk_time": random_walk_time,
        "eigenvector": adjusted_rand_score(true_cluster_ids, eigenvector_clusters),
        "eigenvector_time": eigenvector_time,
        "fast_greedy": adjusted_rand_score(true_cluster_ids, fast_greedy_clusters),
        "fast_greedy_time": fast_greedy_time,
        "walktrap": adjusted_rand_score(true_cluster_ids, walktrap_clusters),
        "walktrap_time": walktrap_time,
        "infomap_greedy": adjusted_rand_score(true_cluster_ids, infomap_clusters),
        "infomap_time": infomap_time,
        "num_vertices": graph.num_vertices(),
        "num_edges": graph.num_edges()
    }
    return scores

if __name__ == "__main__":
    total_scores = {}
    offset = 0
    i = 0
    while i < 25:
        print(f"Iteration {i + 1}/25")
        try:
            run_scores = experiment_step(1000, 20, 200, 0.7, 0.1, i + offset)
            for k, v in run_scores.items():
                total_scores.setdefault(k, [])
                total_scores[k].append(v)
            i += 1
        except:
            offset += 1

    average_scores = {k: sum(v) / len(v) for k, v in total_scores.items()}
    print(json.dumps(average_scores, indent=4))
