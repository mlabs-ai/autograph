# Time Complexity
In this document, we analyze the time complexity of the Autograph algorithm and compare the running time of our algorithm against the running time of other algorithms.

We contend that the worst case runtime of Autograph is _O(n * (E + V log V))_, where _n_ is the number of clusters, _E_ is the number of edges, and _V_ is the number of vertices. In the main step of the algorithm, we iterate through each edge (giving the _E_ term in the runtime complexity) and update scores given to the vertices at either end of that edge. Then, we sort the vertices by descending score (giving us the _V log V_ term). This gives us a score for each vertex. We then split the graph at a point where the scores change drastically (indicating a cluster boundary), and repeat the algorithm on either side of the split (giving us the _n_ term).

When we evaluated Autograph's performance against other algorithms (see `ACCURACY.md` for more details), we also evaluated the wall clock time of each algorithm. The following table outlines the time complexity and average wall clock time for each algorithm:

| Algorithm      | Big O Complexity     | Wall Clock Time (s) |
|----------------|----------------------|---------------------|
| Autograph      | O(n * (E + V log V)) | 347.150             |
| Louvain        | O(E)                 | 0.912               |
| Leiden         | O(E)                 | 0.204               |
| Random Walk    | O(E * V^2)           | 51.306              |
| Eigenvector    | O(V ^ 3)             | 838.383             |
| Fast Greedy    | O(E log V)           | 0.889               |
| Walktrap       | O((V^2) log V)       | 51.531              |
| Infomap Greedy | O(E)                 | 51.821              |

There are a few caveats to mention. Wall clock evaluation of algorithms is rarely a truly reliable measurement; different computers have different architectures, different hardware, and different operating conditions. All of these can affect the wall clock time of an algorithm. Additionally, Autograph beat all the other algorithms in terms of accuracy by quite a wide margin (again, see `ACCURACY.md` for more details), giving it an extra edge over the other algorithms.