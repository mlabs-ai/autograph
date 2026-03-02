from autograph import autograph
import igraph
import random
from sklearn.metrics import adjusted_rand_score

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
    ig_graph.add_vertices(nodes)
    ig_graph.add_edges(edges)

    # Calculate communities using different methods
    print("Calculating clusters using Louvain method")
    louvain_clusters = ig_graph.community_multilevel().membership
    print("Calculating clusters using Newman Girvan method")
    newman_girvan_clusters = ig_graph.community_edge_betweenness().as_clustering().membership
    print("Calculating clusters using random walk method")
    random_walk_clusters = ig_graph.community_walktrap().as_clustering().membership
    print("Calculating clusters using eigenvector method")
    eigenvector_clusters = ig_graph.community_leading_eigenvector().membership

    # Run our clustering algorithm
    print("Calculating clusters using Autograph")
    graph.cluster(0.01, 5, 0.1, 50)
    autograph_clusters = [0] * len(true_cluster_ids)
    for i, cluster in enumerate(graph.get_clusters()):
        for node_id in cluster:
            node_id = int(node_id)
            autograph_clusters[node_id] = i

    # Calculate distance scores
    scores = {
        "louvain": adjusted_rand_score(true_cluster_ids, louvain_clusters),
        "newman_girvan": adjusted_rand_score(true_cluster_ids, newman_girvan_clusters),
        "random_walk": adjusted_rand_score(true_cluster_ids, random_walk_clusters),
        "eigenvector": adjusted_rand_score(true_cluster_ids, eigenvector_clusters),
        "autograph": adjusted_rand_score(true_cluster_ids, autograph_clusters)
    }
    return scores

if __name__ == "__main__":
    total_scores = {}

    for i in range(25):
        run_scores = experiment_step(10, 1000, 10_000, 0.01, 0.1, i)
        for k, v in run_scores.items():
            total_scores.setdefault(k, [])
            total_scores[k].append(v)

    average_scores = {k: sum(v) / len(v) for k, v in total_scores.items()}
    print(average_scores)