"""Best-of-n simulation from a fixed pool of K samples per problem.

For each problem we draw many random size-n subsets of the K samples (the same subsets for every selector,
so selector differences are not sampling noise), pick the subset member with the highest selector score
(ties broken at random), and average the metrics of the picked sample."""
import numpy as np


def subsets(K, n, draws, rng):
    perm = np.argsort(rng.random((draws, K)), axis=1)
    return perm[:, :n]


def bon(scores, metrics, ns, draws=2000, seed=0):
    """scores: (K,) selector scores. metrics: dict name -> (K,) values.
    Returns dict n -> dict metric -> mean value of the selected sample."""
    rng = np.random.default_rng(seed)
    K = len(scores)
    out = {}
    tie = np.random.default_rng(seed + 1).random((draws, K)) * 1e-9
    for n in ns:
        idx = subsets(K, n, draws, rng)
        s = scores[idx] + np.take_along_axis(tie, idx, axis=1)
        pick = idx[np.arange(draws), s.argmax(axis=1)]
        out[n] = {m: float(np.mean(v[pick])) for m, v in metrics.items()}
        out[n]["_picks"] = pick
    return out


def auroc(score, label):
    """P(score of a random positive > score of a random negative), ties count half."""
    score = np.asarray(score, float)
    label = np.asarray(label, int)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float)
    # average ranks for ties
    order = np.argsort(allv)
    sv = allv[order]
    r = np.empty(len(allv))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def boot_ci(values, stat=np.mean, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    bs = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
