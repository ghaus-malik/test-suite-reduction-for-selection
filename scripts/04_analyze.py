"""How few tests can a verifier use before best-of-n selection overfits to them?

Proxy = number of the k chosen tests a sample passes. Gold = passes every hidden EvalPlus test.
Samples 0-7 of each problem are used only to choose tests (for the cross-fit strategy).
Samples 8-15 are the selection pool, so no strategy sees the pool it is scored on."""
import sys, os, ast, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.common import *
from harness.bon import bon, boot_ci
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
gen = load_json(os.path.join(DATA, "generations_n16.json"))
ex = load_json(os.path.join(DATA, "exec_n16.json"))
JA = load_json(os.path.join(DATA, "judge_A.json"))
probs = {p["pid"]: p for p in load_problems()}
KS = [1, 2, 3, 5, 10, 25]
NS = [1, 2, 4, 8]


def input_sizes(test):
    tree = ast.parse(test)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "inputs" for t in node.targets):
            if isinstance(node.value, ast.List):
                return [len(ast.get_source_segment(test, e) or "") for e in node.value.elts]
    return None


def choose(strategy, T, sizes, fit_bits, fit_gold, k, rng):
    k = min(k, T)
    if strategy == "first k (benchmark order)":
        return list(range(k))
    if strategy == "random k":
        return list(rng.choice(T, k, replace=False))
    if strategy == "extremes (shortest and longest inputs)":
        order = list(np.argsort(sizes, kind="stable"))
        out = []
        while len(out) < k:
            out.append(order.pop(0))
            if order and len(out) < k:
                out.append(order.pop(-1))
        return out
    if strategy == "cross-fit greedy (catch fit-half failures)":
        # greedy: pick the test failed by the most still-uncaught wrong samples in the fit half
        wrong = [i for i in range(len(fit_gold)) if not fit_gold[i]]
        chosen, caught = [], set()
        for _ in range(k):
            best, gain = None, -1
            for t in range(T):
                if t in chosen:
                    continue
                g = sum(1 for i in wrong if i not in caught and fit_bits[i][t] == 0)
                if g > gain:
                    best, gain = t, g
            chosen.append(best)
            caught |= {i for i in wrong if fit_bits[i][best] == 0}
        return chosen
    raise ValueError(strategy)


STRATS = ["first k (benchmark order)", "random k", "extremes (shortest and longest inputs)",
          "cross-fit greedy (catch fit-half failures)"]
res = {s: {k: {n: [] for n in NS} for k in KS} for s in STRATS}
overfit = {s: {k: [] for k in KS} for s in STRATS}
judge_rows = {"judge A only": {n: [] for n in NS}, "random": {n: [] for n in NS}}
judge_plus = {k: {n: [] for n in NS} for k in [1, 3]}
used = 0
baseline = []
for pi, (pid, cands) in enumerate(gen.items()):
    keys = [pid + "::" + h(c["code"]) for c in cands]
    rs = [ex[k] for k in keys]
    T = max(len(r["bits"]) for r in rs if r["bits"]) if any(r["bits"] for r in rs) else 0
    if T == 0:
        continue
    bits = np.array([r["bits"] if r["bits"] and len(r["bits"]) == T else [0] * T for r in rs])
    g = bits.min(axis=1) == 1
    fit, pool = slice(0, 8), slice(8, 16)
    if g[pool].all() or (~g[pool]).all():
        continue  # selection cannot matter in this pool
    sizes = input_sizes(probs[pid]["test"]) or [0] * T
    if len(sizes) != T:
        sizes = (sizes + [0] * T)[:T]
    used += 1
    baseline.append(float(g[pool].mean()))
    rng = np.random.default_rng(pi)
    pool_bits, pool_gold = bits[pool], g[pool].astype(float)
    for s in STRATS:
        for k in KS:
            reps = 10 if s == "random k" else 1
            accg = {n: [] for n in NS}
            accof = []
            for _ in range(reps):
                tests = choose(s, T, sizes, bits[fit], g[fit], k, rng)
                proxy = pool_bits[:, tests].sum(axis=1).astype(float)
                passes_all_k = (proxy == len(tests)).astype(float)
                out = bon(proxy, {"gold": pool_gold, "of": passes_all_k * (1 - pool_gold)}, NS, draws=1000, seed=pi)
                for n in NS:
                    accg[n].append(out[n]["gold"])
                accof.append(out[8]["of"])
            for n in NS:
                res[s][k][n].append(float(np.mean(accg[n])))
            overfit[s][k].append(float(np.mean(accof)))
    if all(k in JA for k in keys[8:16]):
        A = np.array([JA[k]["p_yes"] for k in keys[8:16]])
        out = bon(A, {"gold": pool_gold}, NS, draws=1000, seed=pi)
        out0 = bon(np.zeros(8), {"gold": pool_gold}, NS, draws=1000, seed=pi)
        for n in NS:
            judge_rows["judge A only"][n].append(out[n]["gold"])
            judge_rows["random"][n].append(out0[n]["gold"])
        for k in [1, 3]:
            tests = choose("random k", T, sizes, bits[fit], g[fit], k, np.random.default_rng(pi))
            proxy = pool_bits[:, tests].sum(axis=1) + A * 0.5
            out = bon(proxy, {"gold": pool_gold}, NS, draws=1000, seed=pi)
            for n in NS:
                judge_plus[k][n].append(out[n]["gold"])

print("problems used (pool of 8 is mixed):", used)
print(f"no selection: a random sample from the pool passes {np.mean(baseline):.3f} of the time")
summary = {"problems": used, "no_selection": float(np.mean(baseline)), "strategies": {}}
lines = ["| strategy | " + " | ".join(f"k={k}" for k in KS) + " |", "|---" * (len(KS) + 1) + "|"]
lines2 = list(lines)
for s in STRATS:
    summary["strategies"][s] = {k: {"gold_n8": float(np.mean(res[s][k][8])), "ci": boot_ci(res[s][k][8]),
                                    "overfit_n8": float(np.mean(overfit[s][k])),
                                    "gold_by_n": {n: float(np.mean(res[s][k][n])) for n in NS}} for k in KS}
    lines.append(f"| {s} | " + " | ".join(f"{summary['strategies'][s][k]['gold_n8']:.3f}" for k in KS) + " |")
    lines2.append(f"| {s} | " + " | ".join(f"{summary['strategies'][s][k]['overfit_n8']:.3f}" for k in KS) + " |")
t1 = "\n".join(lines); t2 = "\n".join(lines2)
print("Gold pass rate of best-of-8 pick by k tests:\n" + t1)
print("Overfit rate (pick passes all k tests but fails hidden tests):\n" + t2)
nj = len(judge_rows["judge A only"][8])
jl = [f"Problems with judge scores for the pool: {nj}", "", "| selector | n=1 | n=2 | n=4 | n=8 |", "|---|---|---|---|---|"]
for name, d in [("random", judge_rows["random"]), ("judge A only", judge_rows["judge A only"]),
                ("1 random test + judge A tie-break", judge_plus[1]), ("3 random tests + judge A tie-break", judge_plus[3])]:
    jl.append(f"| {name} | " + " | ".join(f"{np.mean(d[n]):.3f}" for n in NS) + " |")
t3 = "\n".join(jl)
print(t3)
summary["judge_vs_tests"] = {name: {n: float(np.mean(d[n])) for n in NS} for name, d in
                             [("random", judge_rows["random"]), ("judge A only", judge_rows["judge A only"]),
                              ("1 test + A", judge_plus[1]), ("3 tests + A", judge_plus[3])]}
summary["judge_vs_tests"]["n_problems"] = nj
with open(os.path.join(RESULTS, "results.json"), "w") as f:
    json.dump(summary, f, indent=1, default=float)
with open(os.path.join(RESULTS, "tables.md"), "w") as f:
    f.write(f"Problems used: {used} (pool of 8 held-out samples has both passing and failing samples)\n\n"
            f"No selection: a random sample from the pool passes {np.mean(baseline):.3f} of the time.\n\n"
            "## Hidden-test pass rate of the best-of-8 pick, selecting by k tests\n\n" + t1 +
            "\n\n## Overfit rate at n=8: pick passes all k chosen tests but fails the hidden suite\n\n" + t2 +
            "\n\n## A few tests vs an LLM judge (problems where the pool was judged)\n\n" + t3 + "\n")
fig, ax = plt.subplots(figsize=(6, 4))
for s in STRATS:
    ax.plot(KS, [summary["strategies"][s][k]["gold_n8"] for k in KS], marker="o", label=s)
ax.axhline(np.mean(judge_rows["judge A only"][8]), ls="--", c="gray", label="judge A only (subset)")
ax.set_xscale("log"); ax.set_xlabel("k tests used to select"); ax.set_ylabel("hidden-test pass rate of best-of-8 pick")
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "k_curve.png"), dpi=150)
print("saved")
