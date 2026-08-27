# Time Complexity

In this document, we analyze the time complexity of the Autograph algorithm and
compare the running time of our algorithm against the running time of other
algorithms.

We contend that the worst case runtime of Autograph is _O(n · (E + V log V))_,
where _n_ is the number of clusters, _E_ is the number of edges, and _V_ is the
number of vertices. In the main step of the algorithm, we iterate through each
edge (giving the _E_ term in the runtime complexity) and update scores given to
the vertices at either end of that edge. Then, we sort the vertices by
descending score (giving us the _V log V_ term). This gives us a score for each
vertex. We then split the graph at a point where the scores change drastically
(indicating a cluster boundary), and repeat the algorithm on either side of the
split (giving us the _n_ term).

## Performance work

The theoretical bound above assumes a reasonably efficient implementation, but
for a time the implementation did not meet it, because the inner `cluster_step`
performed two avoidable per-edge operations:

1. **A transcendental `exp()` call per edge endpoint.** The weighting step
   computed `exp(-factor * local_index)` on the fly for every edge, on every
   step. Since the exponent only ever depends on the (small, integer) local
   index, this was replaced with a single precomputed lookup table built once
   per `cluster()` call. This is an exact, bit-for-bit equivalent optimization
   that removes a dominant cost.
2. **A `HashMap`-based vertex remap per step.** Each step rebuilt a `HashMap`
   permutation and hashed over every vertex and edge. Because the permutation
   produced by a step is a contiguous-range permutation, a flat array index
   lookup was substituted for the hash map.

Together these reduced the wall-clock time of a single `cluster_step` by roughly
two orders of magnitude on the benchmark graphs (see below), with no change to
the produced clusters. Small sub-ranges additionally use a serial sort instead
of a fork to the parallel sort, avoiding overhead on the many small ranges seen
during recursion.

A further optimization removed the remaining hot cost: the recursion originally
re-scanned the *full* edge list on every one of the `steps_before_subdivide`
iterations at every recursion node. The recursion now extracts the in-range
edges (in local coordinates) once per node and drives each step against that
local list, applying the net vertex permutation to global state only once at the
end of the node. This amortizes the edge scan away from the step-count and
recursion-depth multipliers while preserving identical cluster output.

## Wall clock comparison

When we evaluated Autograph's performance against other algorithms (see
`ACCURACY.md` for details — 150 clusters, 20–200 nodes each, ~16.6k vertices and
~206k edges per graph, 10 iterations), we also recorded the average wall clock
time of each algorithm. The following table outlines the time complexity and
average wall clock time for each algorithm:

| Algorithm   | Big O Complexity     | Wall Clock Time (s) |
|-------------|----------------------|---------------------|
| Fast Greedy | O(E log V)           | 0.15                |
| Autograph   | O(n · (E + V log V)) | 0.39                |
| Leiden      | O(E)                 | 0.55                |
| Louvain     | O(E)                 | 0.83                |
| Infomap     | O(E)                 | 2.41                |
| Walktrap    | O((V²) log V)        | 2.86                |

There are a few caveats to mention. Wall clock evaluation of algorithms is
rarely a truly reliable measurement; different computers have different
architectures, different hardware, and different operating conditions. All of
these can affect the wall clock time of an algorithm. Additionally, the Louvain
and Leiden figures above include a resolution sweep (see `ACCURACY.md`), so they
reflect the cost of tuning those methods rather than a single run.

Autograph's time is now in the same order of magnitude as the modularity-based
methods on these graphs, rather than orders of magnitude slower as an earlier
version of this document reported. The earlier figure (347s per run) reflected
the un-optimized `exp`-per-edge implementation described above.