"""Sanity check: every reference solution must pass all of its own tests, and a trivially wrong solution must fail."""
import time, json, collections
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.common import *
probs = load_problems()
print("problems", len(probs), collections.Counter(p["src"] for p in probs))
t = time.time()
res = run_many([(p["reference"], p) for p in probs], workers=10)
print("ref run secs", round(time.time() - t, 1))
bad = []
ntests = []
for p, r in zip(probs, res):
    if r["bits"] is None or not r["bits"] or min(r["bits"]) == 0 or (r["vis"] and min(r["vis"]) == 0):
        bad.append((p["pid"], r["status"], None if r["bits"] is None else (sum(r["bits"]), len(r["bits"])), r["vis"]))
    else:
        ntests.append(len(r["bits"]))
print("reference failures:", len(bad))
for b in bad[:40]: print("  ", b)
print("tests per problem: min", min(ntests), "median", sorted(ntests)[len(ntests)//2], "max", max(ntests))
wrong = run_many([(f"def {p['entry']}(*a, **k):\n    return None\n", p) for p in probs[:50]], workers=10)
print("trivial-wrong all-pass count (should be ~0):", sum(1 for r in wrong if r["bits"] and min(r["bits"]) == 1))
save_json({"bad_reference": [b[0] for b in bad]}, os.path.join(DATA, "harness_check.json"))
