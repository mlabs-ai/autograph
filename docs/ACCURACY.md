# Accuracy

In this document, we explore the "accuracy" of the Autograph algorithm and
compare it to other graph clustering algorithms. We begin by detailing the
difficulties with evaluating clustering algorithms, then we explain how we
circumvented those difficulties, before finally detailing the accuracy
comparison and interpreting those results.

## Evaluating Graph Clustering Algorithms

Evaluating the accuracy of a graph clustering algorithm is not a trivial task,
like it often is with classification algorithms. To begin with, there are very
few existing datasets whose clusters are definitively known. Most graphs have
"fuzzy" edges around their clusters; i.e., the clusters do not have absolute
borders. One individual might cluster a graph one way, while another might
cluster it another way. Thus, real world graphs are impractical to use for
evaluation.

To combat this, we evaluated the accuracy of our algorithm on graphs that we
generated algorithmically. The graphs were generated in the following way:

1. Each graph had a configurable number of clusters. We report results across a
   range of scales — 150, 500, 1000, 1500, and 2000 planted clusters — to
   observe how each algorithm scales.
2. Each cluster had a number of nodes chosen randomly from the range 20 to 200.
3. Each cluster was a
   [scale-free network](https://en.wikipedia.org/wiki/Scale-free_network). We
   chose this kind of cluster because it more closely reflects real world data.
   During the generation of the cluster, each new node added was connected to
   10% of the other nodes in the cluster (we call this parameter "salt").
4. 70% of the nodes in each cluster had a connection to a node in another
   cluster (we call this parameter "pepper").

The code for this evaluation can be found in `python/evaluate_other_algorithms.py`.

This gave us a known, indisputable ground truth against which we could compare.
To do this comparison, we used the
[`adjusted_rand_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
metric provided by scikit-learn. This metric, which can range from -0.5 to 1.0,
gives a score of 0.0 when the clustering is random, and a score of 1.0 when the
clustering is identical (even factoring in different labels for the clusters).

### Making the comparison fair

A naive benchmark hands each baseline its library defaults. This is not a fair
comparison, because the community-detection methods in `igraph` are controlled
by parameter choices (notably the resolution of the quality function and the
number of communities the result is cut into) that fundamentally determine how
many clusters they find. To give every algorithm an equal opportunity to recover
the planted partition, we applied two corrections:

1. **Tuned resolution.** Louvain (`community_multilevel`) and Leiden
   (`community_leiden`) are both modularity-based, and we swept their
   `resolution` parameter over a grid, reporting each method's best
   `adjusted_rand_score` over the grid.
2. **Cut dendrograms at the known cluster count.** The hierarchical methods
   (fast greedy and walktrap) return a dendrogram rather than a flat partition;
   we cut them at the known number of planted clusters via `.as_clustering(n=k)`
   so they are not evaluated at an arbitrary level.

Without these corrections, Leiden in particular scored 0.0: its default
configuration (Constant Potts Model objective with `resolution=1.0`) fragments
the graph into one community per node, which trivially scores zero against the
planted partition. That 0.0 was an artifact of misconfiguration, not a
reflection of the method.

## Results

In this section, we present the accuracy results for Autograph vs a selection of
some of the most common graph clustering algorithms. We repeated the experiment
10 times per scale (each iteration seeded with its iteration index) and
collected the average `adjusted_rand_score` for each algorithm.

### Accuracy across scales

The table below reports the average `adjusted_rand_score` at each planted
cluster count. The graphs range from ~17,000 vertices (150 clusters) to
~219,000 vertices (2000 clusters).

| Algorithm   | 150    | 500    | 1000   | 1500   | 2000   |
|-------------|--------|--------|--------|--------|--------|
| Infomap     | 0.996  | 0.984  | 0.977  | 0.971  | 0.966  |
| Autograph   | 0.731  | 0.700  | 0.715  | 0.695  | 0.683  |
| Louvain     | 0.879  | 0.656  | 0.531  | 0.469  | 0.428  |
| Leiden      | 0.878  | 0.655  | 0.533  | 0.469  | 0.429  |
| Fast Greedy | 0.226  | 0.110  | 0.083  | 0.074  | 0.070  |
| Walktrap    | 0.998  | 0.997  | 0.998  | —      | —      |

**Walktrap** recovers the planted partition almost perfectly up to 1000
clusters, but its distance-matrix memory scales quadratically in the vertex
count and exhausts available RAM at 1500+ clusters, so it is omitted there.

**Infomap** is the strongest generally-applicable baseline, degrading only
slightly (0.996 → 0.966) across the full scale range.

**Autograph holds its accuracy steady** (~0.68–0.73) across a 13× growth in
graph size. The tuned modularity methods — Louvain and Leiden — degrade sharply
(from ~0.88 to ~0.43) as the number of clusters grows. As a result, Autograph
overtakes both around 500 clusters, and by 2000 clusters it leads them by a
margin of roughly 0.26 in adjusted Rand score.

This is consistent with the project's design goal: Autograph is intended for
large, frame-of-reference-structured knowledge graphs, where its accuracy is
relatively insensitive to scale, whereas modularity-based methods struggle once
the community structure becomes finer-grained.

### Caveats

The potential number of graphs is infinite, and we could only feasibly test on
a small number of graphs. Therefore it is difficult to generalize these
results. In particular, these graphs were generated to have clean, well-separated
clusters; on fuzzier or more realistic graphs the ordering between methods may
differ. The comparison above uses a single accuracy metric on planted graphs;
it does not measure other qualities Autograph is designed for (e.g. robust
frame-of-reference assignment, or the ability of the same algorithm to serve as
a recommender).

### How this differs from an earlier version of this document

An earlier version of this document reported Autograph with a "clear advantage"
(0.712 vs 0.0–0.34 for the baselines). That conclusion was the product of two
bugs in the evaluation harness, both since fixed:

1. **Leiden's 0.0** came from its default `CPM`/`resolution=1.0` configuration,
   not from the method's actual capability.
2. **Infomap and "Random Walk"** were both implemented as calls to walktrap
   (a copy-paste error), so three table rows reported the same number.

After correcting these and tuning the baselines fairly, the comparison is
methodologically sound, and the results above reflect it honestly.