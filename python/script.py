from autograph import autograph
import json
import time

RELATIONSHIP = "P31"

print("Loading graph", flush=True)
graph = autograph.KnowledgeGraph.from_dot_file(f"../data/wikidata/{RELATIONSHIP}.dot")

print("Clustering", flush=True)
start_time = time.time()
graph.cluster(0.001, 10, 0.25, 10)
cluster_time = time.time() - start_time

print(f"Cluster time: {cluster_time:.3f}s", flush=True)
clusters = graph.get_clusters()
with open(f"../data/clusters/{RELATIONSHIP}.json", 'w') as file:
    json.dump(clusters, file)