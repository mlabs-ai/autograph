from autograph import autograph
import fire
import json
import time

def main(
    dot_file: str,
    output_file: str,
    factor: float = 0.0001,
    steps_before_subdivide: int = 5,
    boundary_threshold: float = 0.25,
    min_cluster_size: int = 10
):
    """
    This script performs the Autograph clustering algorithm on the graph contained
    within a given DOT file and outputs the result to a given JSON output file. 
    Additional parameters for the algorithm may be given, but have defaults set.
    """

    # Load the graph
    print("Loading graph")
    graph = autograph.KnowledgeGraph.from_dot_file(dot_file)

    # Perform the clustering algorithm
    print("Clustering")
    start_time = time.time()
    graph.cluster(factor, steps_before_subdivide, boundary_threshold, min_cluster_size)
    cluster_time = time.time() - start_time

    # Write to file
    print(f"Cluster time: {cluster_time:.3f}s", flush=True)
    clusters = graph.get_clusters()
    with open(output_file, 'w') as file:
        json.dump(clusters, file)

if __name__ == "__main__":
    fire.Fire(main)