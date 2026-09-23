import bz2
import json
import sys
from collections import Counter

# id -> human description. Includes the "obvious" P31/P279 as a baseline.
PROPERTIES = {
    "P106": "occupation",
    "P108": "employer",
    "P69": "educated at",
    "P185": "doctoral student",
    "P184": "doctoral advisor",
    "P101": "field of work",
    "P27": "country of citizenship",
    "P31": "instance of",
}

def main(path, limit):
    counts = Counter()
    seen = 0
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line or line == "[" or line == "]":
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            claims = obj.get("claims", {})
            for pid in PROPERTIES:
                if pid in claims:
                    counts[pid] += 1
            seen += 1
            if seen >= limit:
                break

    print(f"sampled {seen} entities (head of dump)")
    for pid, desc in PROPERTIES.items():
        n = counts[pid]
        pct = 100.0 * n / seen if seen else 0.0
        print(f"{pid:6s} {desc:55s} {n:>8d}  {pct:5.1f}%")

if __name__ == "__main__":
    path = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
    main(path, limit)
