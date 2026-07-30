import json
import random

random.seed(42)

with open("outputs/ccneg/ccneg_pairs.json") as f:
    d = json.load(f)

pairs = d["pairs"]
print(f"전체 쌍 개수: {len(pairs)}\n")

print("=== 무작위 샘플 15개 (긍정 / 부정) ===\n")
for pos, neg in random.sample(pairs, 15):
    print(f"POS: {pos}")
    print(f"NEG: {neg}")
    print()
