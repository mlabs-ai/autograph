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

We recorded the average wall-clock time per algorithm on the planted
scale-free graphs described in `ACCURACY.md` (20–200 nodes per cluster, 10
iterations per scale). Times are average seconds per run.

| Algorithm   | Big O Complexity     | 150    | 1000   | 2000   |
|-------------|----------------------|--------|--------|--------|
| Fast Greedy | O(E log V)           | 0.15   | 1.00   | 1.81   |
| Leiden      | O(E)                 | 0.55   | 6.15   | 10.8   |
| Louvain     | O(E)                 | 0.83   | 14.4   | 24.6   |
| Autograph   | O(n · (E + V log V)) | 0.39   | 20.3   | 54.3   |
| Infomap     | O(E)                 | 2.41   | 23.5   | 41.1   |
| Walktrap    | O((V²) log V)        | 2.86   | 144    | —      |

(Dashes denote scales at which the method could not complete: walktrap's
distance-matrix memory is quadratic in the vertex count and exceeds available
RAM at 1500 clusters and above. See `ACCURACY.md`.)

There are a few caveats to mention. Wall clock evaluation of algorithms is
rarely a truly reliable measurement; different computers have different
architectures, different hardware, and different operating conditions. All of
these can affect the wall clock time of an algorithm. Additionally, the Louvain
and Leiden figures include a resolution sweep (see `ACCURACY.md`), so they
reflect the cost of tuning those methods rather than a single run.

Autograph's time grows more steeply than the pure-modularity methods, consistent
with its `O(n · (E + V log V))` bound and the extra `n` term from recursive
subdivision. It remains within the same order of magnitude as Louvain and
Infomap across the scales tested, and is far faster than the un-optimized
implementation (which took 347s at the 150-cluster scale; see above).