import json
import random
import re

random.seed(123)

with open("outputs/ccneg/ccneg_pairs.json") as f:
    d = json.load(f)

pairs = d["pairs"]

patterns = {
    "not X": [],
    "without X": [],
    "no X": [],
    "n't": [],
    "other": [],
}

for pos, neg in pairs:
    if ", not " in neg or neg.startswith("not "):
        patterns["not X"].append((pos, neg))
    elif "without" in neg:
        patterns["without X"].append((pos, neg))
    elif re.search(r"\bno \w", neg):
        patterns["no X"].append((pos, neg))
    elif "n't" in neg:
        patterns["n't"].append((pos, neg))
    else:
        patterns["other"].append((pos, neg))

print("=== 부정어 패턴별 개수 ===")
for k, v in patterns.items():
    print(f"  {k}: {len(v)}")

print("\n=== 패턴별 샘플 5개씩 ===\n")
for k, v in patterns.items():
    print(f"--- {k} ---")
    for pos, neg in random.sample(v, min(5, len(v))):
        print(f"  POS: {pos}")
        print(f"  NEG: {neg}")
    print()
