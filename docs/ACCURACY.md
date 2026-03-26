# Accuracy
In this document, we explore the "accuracy" of our algorithm and compare it to other graph clustering algorithms. We begin by detailing the difficulties with evaluating clustering algorithms, then we explain how we circumvented those difficulties, before finally detailing the accuracy comparison and interpreting those results.

## Evaluating Graph Clustering Algorithms
Evaluating the accuracy of a graph clustering algorithm is not a trivial task, like it often is with classification algorithms. To begin with, there are very few existing datasets whose clusters are definitively known. Most graphs have "fuzzy" edges around their clusters; i.e., the clusters do not have absolute borders. One individual might cluster a graph one way, while another might cluster it another way. Thus, real world graphs are impractical to use for evaluation.

To combat this, we evaluated the accuracy of our algorithm on graphs that we generated algorithmically. The graphs were generated in the following way:
1. Each graph had 1000 clusters.
2. Each cluster had a number of nodes chosen randomly from the range 20 to 200.
3. Each cluster was a [scale-free network](https://en.wikipedia.org/wiki/Scale-free_network). We chose this kind of cluster because it more closely reflects real world data. During the generation of the cluster, each new node added was connected to 10% of the other nodes in the cluster (We call this parameter "salt").
4. 70% of the nodes in each cluster had a connection to a node in another cluster (We call this parameter "pepper").

The code for this evaluation can be found in `python/evaluate_other_algorithms.py`. Instructions for running that script can be found in the README file.

This gave us a known, indisputable ground truth against which we could compare. To do this comparison, we used the [`adjusted_rand_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html) metric provided by scikit-learn. This metric, which can range from -0.5 to 1.0, gives a score of 0.0 when the clustering is random, and a score of 1.0 when the clustering is identical (even factoring in different labels for the clusters).

Thus, we have our method for evaluating graph clusters.

## Results
In this section, we present the accuracy results for Autograph vs a selection of some of the most common graph clustering algorithms. We repeated the experiment outlined in the above section 25 times (each iteration seeded with a value corresponding to the number of the iteration we are in) and collected the average results for each algorithm, which is what is presented here.

| Algorithm      | Score |
|----------------|-------|
| Autograph      | 0.712 |
| Louvain        | 0.316 |
| Leiden         | 0.0   |
| Random Walk    | 0.332 |
| Eigenvector    | 0.343 |
| Fast Greedy    | 0.085 |
| Walktrap       | 0.332 |
| Infomap Greedy | 0.332 |

This shows a clear advantage for Autograph. We will note, however, that the potential number of graphs is infinite, and we could only feasibly test our algorithm on a small number of graphs. Therefore, it is difficult to generalize these results. Nonetheless, we believe that these kinds of graphs are similar to the ones that will be used by AGI, which makes us excited.