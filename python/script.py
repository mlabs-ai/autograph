from autograph import autograph
import matplotlib.pyplot as plt
import time

CLUSTER_SIZE = 10_000
NUM_CLUSTERS = 40
PEPPER_AMOUNT = 1_000

builder = autograph.GraphBuilder(0)
for i in range(NUM_CLUSTERS):
    builder.add_scale_free_cluster(CLUSTER_SIZE, 100)

print("Added scale free clusters", flush=True)

for _ in range(PEPPER_AMOUNT):
    for i in range(NUM_CLUSTERS):
        builder.add_random_link(i, (i + 1) % NUM_CLUSTERS)

print("Added links", flush=True)

graph = builder.finalize_graph()
graph.shuffle_vertex_ids(0)

print("Clustering", flush=True)
start_time = time.time()
graph.cluster(0.01, 5, 0.05, 250)
cluster_time = time.time() - start_time

print(f"Cluster time: {cluster_time:.3f}s", flush=True)
print(graph.get_clusters())