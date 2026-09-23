#!/usr/bin/env python3
"""Resolve a batch of Wikidata QIDs to their English labels.

Distributed on the command line (or stdin) as a comma/newline-separated list.
Uses the public Wikidata MediaWiki API; emits `QID \t label` pairs to stdout.
Skips QIDs it cannot resolve.
"""
import json
import sys
import urllib.request
import urllib.parse


def resolve(qids):
    out = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        url = (
            "https://www.wikidata.org/w/api.php?"
            + urllib.parse.urlencode(
                {
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "props": "labels",
                    "languages": "en",
                    "format": "json",
                }
            )
        )
        req = urllib.request.Request(url, headers={
            "User-Agent": "MLabs autograph milestone-2 report/0.1 (https://example.org; dev tool)"
        })
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        for qid, ent in data.get("entities", {}).items():
            label = ent.get("labels", {}).get("en", {}).get("value")
            if label:
                out[qid] = label
    return out


def main():
    text = sys.stdin.read()
    qids = [q.strip() for q in text.replace(",", "\n").splitlines() if q.strip()]
    labels = resolve(qids)
    for qid in qids:
        if qid in labels:
            print(f"{qid}\t{labels[qid]}")


if __name__ == "__main__":
    main()
