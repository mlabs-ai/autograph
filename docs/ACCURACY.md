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

1. Each graph had a configurable number of clusters (150 in the runs reported
   below).
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
10 times (each iteration seeded with its iteration index) and collected the
average results for each algorithm, which is what is presented here.

| Algorithm    | Score |
|--------------|-------|
| Walktrap     | 0.998 |
| Infomap      | 0.996 |
| Louvain      | 0.879 |
| Leiden       | 0.878 |
| Autograph    | 0.731 |
| Fast Greedy  | 0.226 |

On the planted scale-free graphs we tested, **Walktrap and Infomap recover the
planted partition almost perfectly**, followed closely by the tuned modularity
methods (Louvain and Leiden). Autograph scores in the middle of the pack, ahead
of fast greedy.

We will note, however, that the potential number of graphs is infinite, and we
could only feasibly test our algorithm on a small number of graphs. Therefore,
it is difficult to generalize these results. In particular, these graphs were
generated to have clean, well-separated clusters; on fuzzier or more realistic
graphs the ordering between methods may differ, and Autograph's
context-driven (frame-of-reference) partitioning may offer advantages that a
raw accuracy score on planted graphs does not capture.

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