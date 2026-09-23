import argparse
import bz2
import json
import sys

# Properties of interest.
P101 = "P101"  # field of work        (person -> field)
P106 = "P106"  # occupation           (person -> occupation)
P50 = "P50"    # author               (work   -> person)
P921 = "P921"  # main subject         (work   -> subject)


def target_ids(claims, pid):
    """Entity-id targets of a property (only 'wikibase-entityid' values)."""
    out = []
    for stmt in claims.get(pid, []):
        snak = stmt.get("mainsnak", {})
        if snak.get("snaktype") != "value":
            continue
        val = snak.get("datavalue", {}).get("value", {})
        eid = val.get("id")
        if eid:
            out.append(eid)
    return out


def english_label(obj):
    return obj.get("labels", {}).get("en", {}).get("value")


def stream_entities(path, stride=1, max_sample=None, progress_every=500_000):
    """Yield every `stride`-th entity, up to `max_sample` yielded values.

    `stride > 1` implements systematic (uniform-by-position) sampling across the
    full dump. Every line is still decompressed, so total wall-clock time is a
    full pass regardless of stride; `stride` only controls how many entities are
    *kept* (and therefore memory).
    """
    line_index = 0
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line or line == "[" or line == "]":
                continue
            line_index += 1
            if line_index % stride != 0:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("id")
            if not qid:
                continue
            yield {
                "qid": qid,
                "label": english_label(obj),
                P101: target_ids(obj.get("claims", {}), P101),
                P106: target_ids(obj.get("claims", {}), P106),
                P50: target_ids(obj.get("claims", {}), P50),
                P921: target_ids(obj.get("claims", {}), P921),
            }
            if line_index % progress_every == 0:
                print(f"[progress] kept ~{line_index // stride:,} entities",
                      file=sys.stderr, flush=True)
            if max_sample is not None and line_index // stride >= max_sample:
                return


def compute(entities):
    """Seed list, connected list, and overlap over a sampled set."""
    seed = [e for e in entities if e[P101]]
    seed_qids = {e["qid"] for e in seed}

    occ_targets = set()
    for e in seed:
        occ_targets.update(e[P106])

    works_by_author = []
    works_by_subject = []
    for e in entities:
        if any(t in seed_qids for t in e[P50]):
            works_by_author.append(e)
        if any(t in seed_qids for t in e[P921]):
            works_by_subject.append(e)

    connected_qids = set(occ_targets)
    connected_qids.update(e["qid"] for e in works_by_author)
    connected_qids.update(e["qid"] for e in works_by_subject)

    overlap = seed_qids & connected_qids
    seed_with_occupation = [e for e in seed if e[P106]]

    return {
        "n_sampled": len(entities),
        "seed_size": len(seed),
        "seed_fraction": len(seed) / len(entities) if entities else 0.0,
        "seed_with_occupation": len(seed_with_occupation),
        "occ_targets_size": len(occ_targets),
        "works_by_author_size": len(works_by_author),
        "works_by_subject_size": len(works_by_subject),
        "connected_size": len(connected_qids),
        "overlap_size": len(overlap),
    }


def report(stats):
    s = stats
    print(
        f"sampled={s['n_sampled']:,} "
        f"seed={s['seed_size']:,} ({100 * s['seed_fraction']:.1f}%) "
        f"seed∩occupation={s['seed_with_occupation']:,} "
        f"occupations={s['occ_targets_size']:,} "
        f"authored={s['works_by_author_size']:,} "
        f"subject={s['works_by_subject_size']:,} "
        f"connected={s['connected_size']:,} "
        f"overlap={s['overlap_size']:,}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--stride", type=int, default=1,
                    help="sample every Nth entity (default 1 = every entity)")
    ap.add_argument("--max-sample", type=int, default=2_000_000,
                    help="stop after this many sampled entities")
    ap.add_argument("--checkpoint", type=int, default=250_000,
                    help="print a summary every this many sampled entities")
    args = ap.parse_args()

    buf = []
    next_checkpoint = args.checkpoint
    for e in stream_entities(args.path, stride=args.stride,
                             max_sample=args.max_sample):
        buf.append(e)
        if len(buf) >= next_checkpoint:
            report(compute(buf))
            next_checkpoint += args.checkpoint


if __name__ == "__main__":
    main()
