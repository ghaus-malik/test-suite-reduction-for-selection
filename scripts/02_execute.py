"""Run every unique candidate against the hidden EvalPlus tests (one bit per test) and the visible MBPP asserts."""
import sys, time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.common import *
gen = load_json(os.path.join(DATA, "generations_n16.json"))
probs = {p["pid"]: p for p in load_problems()}
out_path = os.path.join(DATA, "exec_n16.json")
done = load_json(out_path) if os.path.exists(out_path) else {}
jobs, keys = [], []
for pid, cands in gen.items():
    for c in cands:
        k = pid + "::" + h(c["code"])
        if k in done or k in keys:
            continue
        keys.append(k); jobs.append((c["code"], probs[pid]))
print("unique candidates to run:", len(jobs), flush=True)
t = time.time()
B = 200
for s in range(0, len(jobs), B):
    res = run_many(jobs[s:s + B], workers=4)
    for k, r in zip(keys[s:s + B], res):
        done[k] = r
    save_json(done, out_path)
    print(f"{min(s + B, len(jobs))}/{len(jobs)} {time.time() - t:.0f}s", flush=True)
