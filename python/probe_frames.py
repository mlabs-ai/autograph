#!/usr/bin/env python3
"""Quick probe: ingest a bounded subset and report the cluster frame structure."""
from autograph import autograph
import sys
import time


def main(path, limit, preds_csv, factor=0.01, steps=5, bt=0.1, mcs=10):
    preds = preds_csv.split(",")
    t = time.time()
    g = autograph.FrameGraph.from_wikidata_bz2(path, preds, limit)
    print(f"ingest: {g.num_entities()} entities, {g.num_edges()} edges "
          f"({time.time() - t:.1f}s)", flush=True)
    t = time.time()
    frames = g.cluster(factor, steps, bt, mcs)
    print(f"cluster: {len(frames)} frames ({time.time() - t:.1f}s)", flush=True)
    sizes = sorted((len(f) for f in frames), reverse=True)
    print("frame sizes (desc):", sizes, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
