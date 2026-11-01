"""Score unique candidates with one judge. Only problems where the policy's 16 samples disagree on the
hidden tests ("mixed" problems) are judged, because selection cannot change the outcome elsewhere."""
import sys, time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.common import *
which = sys.argv[1]  # A or B
repo = {"A": JUDGE_A, "B": JUDGE_B}[which]
gen = load_json(os.path.join(DATA, "generations_n16.json"))
ex = load_json(os.path.join(DATA, "exec_n16.json"))
probs = {p["pid"]: p for p in load_problems()}

def gold(r):
    return int(bool(r["bits"]) and min(r["bits"]) == 1)

mixed = []
for pid, cands in gen.items():
    g = [gold(ex[pid + "::" + h(c["code"])]) for c in cands]
    if 0 < sum(g) < len(g):
        mixed.append(pid)
print("mixed problems:", len(mixed), "of", len(gen), flush=True)
# GPU prefill runs at about 250 tokens/s for a 7B model here, so judge a fixed random subset of 150 mixed problems
sub_path = os.path.join(DATA, "judged_problems.json")
if not os.path.exists(sub_path):
    import random
    rng = random.Random(0)
    mb = [m for m in mixed if m.startswith("MBPP")]; he = [m for m in mixed if not m.startswith("MBPP")]
    k_mb = round(150 * len(mb) / len(mixed))
    save_json(sorted(rng.sample(mb, k_mb)) + sorted(rng.sample(he, 150 - k_mb)), sub_path)
mixed = load_json(sub_path)
if which == "B":
    # judge B (slower, about 3.3 s per sample) scores a fixed random 100 of the 150. The two-judge analyses use these 100
    import random
    mixed = sorted(random.Random(1).sample(mixed, 100))
print("judging subset:", len(mixed), flush=True)
out_path = os.path.join(DATA, f"judge_{which}.json")
done = load_json(out_path) if os.path.exists(out_path) else {}
keys, msgs = [], []
for pid in mixed:
    for c in gen[pid]:
        k = pid + "::" + h(c["code"])
        if k in done or k in keys:
            continue
        keys.append(k); msgs.append(judge_messages(probs[pid]["task"], c["code"]))
print("to score:", len(keys), flush=True)
J = LogitJudge(repo)
t = time.time()
B = 100
for s in range(0, len(keys), B):
    r = J.score(msgs[s:s + B])
    for k, v in zip(keys[s:s + B], r):
        done[k] = v
    save_json(done, out_path)
    print(f"{min(s+B, len(keys))}/{len(keys)} {time.time()-t:.0f}s", flush=True)
