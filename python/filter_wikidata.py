#!/usr/bin/env python3
"""Filter a Wikidata JSON dump to a curated set of entity properties.

Reads a bzip2-compressed Wikidata dump (``latest-all.json.bz2`` layout: a JSON
array with one entity object per line) and keeps every entity that carries at
least one of a fixed set of career/membership properties. For each kept entity,
only those properties -- plus ``instance_of`` (P31) when present -- are kept in
``claims``. The result is written as a bzip2-compressed JSON array in the same
layout as the source dump.

Parallelism & resource bounds
-----------------------------
* Decompression and re-compression are handled by ``lbzip2`` (a multi-threaded
  bzip2), so those stages are parallelised at the C level.
* JSON line filtering is fanned out across a ``ProcessPoolExecutor`` of
  ``--procs`` worker processes (default 12).
* Memory is bounded by the pipeline depth (``2 * procs`` in-flight chunks) times
  ``--chunk-lines``; the defaults keep peak RSS well under a few hundred MB,
  comfortably below the 24 GB budget.

Progress is reported with ``tqdm`` (entities scanned, matches, rate).

Usage
-----
    python3 python/filter_wikidata.py data/wikidata/latest-all.json.bz2 \
        data/wikidata/person_career.json.bz2

Run a quick smoke test on the first N decompressed lines with ``--limit-lines``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

# Properties we select entities on.
TARGET_PROPERTIES: tuple[str, ...] = (
    "P106",  # occupation
    "P39",   # position held
    "P69",   # educated at
    "P108",  # employer
    "P463",  # member of
    "P1344", # participant in
    "P54",   # member of sports team
    "P102",  # member of political party
    "P101",  # field of work
)

# Additionally retained for matching entities, when present.
INSTANCE_OF = "P31"


def _filter_batch(lines: list[str]) -> list[str]:
    """Filter one batch of raw entity lines -> compact JSON for kept entities.

    Pure function, run in worker processes (must stay importable / picklable).
    """
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line == "[" or line == "]":
            continue
        line = line.rstrip(",")  # array separator in the dump layout
        try:
            obj = json.loads(line)
        except ValueError:
            continue

        claims = obj.get("claims")
        if not isinstance(claims, dict) or not claims:
            continue

        kept: dict[str, object] = {}
        for pid in TARGET_PROPERTIES:
            stmts = claims.get(pid)
            if isinstance(stmts, list) and stmts:
                kept[pid] = stmts

        if not kept:
            continue

        p31 = claims.get(INSTANCE_OF)
        if isinstance(p31, list) and p31:
            kept[INSTANCE_OF] = p31

        obj["claims"] = kept
        out.append(json.dumps(obj, ensure_ascii=True, separators=(",", ":")))
    return out


def _require_lbzip2() -> str:
    """Return the lbzip2 executable path, or fall back to the plain bz2 chain."""
    lb = shutil.which("lbzip2")
    if lb:
        return lb
    bz2 = shutil.which("bzip2")
    if bz2:
        return bz2
    raise SystemExit("neither lbzip2 nor bzip2 found on PATH")


def _decode_chunks(decomp: subprocess.Popen, chunk_lines: int, limit: int | None):
    """Yield lists of raw entity lines (list[str]) read from `decomp` stdout."""
    buf: list[str] = []
    seen = 0
    for raw in decomp.stdout:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        buf.append(line)
        seen += 1
        if len(buf) >= chunk_lines:
            yield buf
            buf = []
        if limit is not None and seen >= limit:
            break
    if buf:
        yield buf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="path to the bzip2-compressed Wikidata dump")
    ap.add_argument("output", help="output .bz2 path")
    ap.add_argument("--procs", type=int, default=12,
                    help="number of filter worker processes (default 12)")
    ap.add_argument("--io-procs", type=int, default=4,
                    help="threads for lbzip2 decompress/compress (default 4)")
    ap.add_argument("--chunk-lines", type=int, default=1024,
                    help="entities batched per task (default 1024)")
    ap.add_argument("--limit-lines", type=int, default=None,
                    help="stop after this many decompressed lines (testing)")
    ap.add_argument("--total", type=int, default=121_000_000,
                    help="total entities for the %% progress / ETA (an estimate; "
                         "the full dump is ~121M lines). Set 0 to disable.")
    args = ap.parse_args()

    total = args.total if args.total and args.total > 0 else None

    lbzip2 = _require_lbzip2()
    max_inflight = max(2, args.procs * 2)

    decomp_cmd = [lbzip2, "-dc", "-n", str(args.io_procs), args.input]
    decomp = subprocess.Popen(
        decomp_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    comp_cmd = [lbzip2, "-zc", "-n", str(args.io_procs)]
    with open(args.output, "wb") as out_fh:
        comp = subprocess.Popen(
            comp_cmd, stdin=subprocess.PIPE, stdout=out_fh, stderr=subprocess.DEVNULL
        )

        scanned = 0
        matched = 0
        wrote_first = False

        # Streaming array header/footer, byte-wise, through the compressor.
        def write_raw(b: bytes) -> None:
            comp.stdin.write(b)

        def write_entity(s: str) -> None:
            nonlocal wrote_first
            prefix = b"[\n" if not wrote_first else b",\n"
            wrote_first = True
            comp.stdin.write(prefix + s.encode("utf-8"))

        pending: deque["object"] = deque()
        pool = ProcessPoolExecutor(max_workers=args.procs)

        def drain_one() -> None:
            """Write the oldest finished batch's matches to the output, in order."""
            nonlocal matched
            for s in pending.popleft().result():
                write_entity(s)
                matched += 1

        try:
            with tqdm(total=total, unit="ent", desc="scan/filter",
                      dynamic_ncols=True, smoothing=0.0) as bar:
                for batch in _decode_chunks(decomp, args.chunk_lines,
                                            args.limit_lines):
                    scanned += len(batch)
                    bar.update(len(batch))

                    pending.append(pool.submit(_filter_batch, batch))

                    # Bound in-flight work: drain results (in order) as needed.
                    while len(pending) >= max_inflight:
                        drain_one()

                # Drain the rest.
                while pending:
                    drain_one()

            bar.set_postfix(matches=matched)
            if wrote_first:
                write_raw(b"\n]\n")
            else:
                write_raw(b"[\n]\n")
        finally:
            pool.shutdown(cancel_futures=True)
            comp.stdin.close()
            comp.wait()
            decomp.stdout.close()
            decomp.terminate()
            decomp.wait()

    print(f"\ndone: scanned={scanned:,} matched={matched:,} -> {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
