Problems used: 309 (pool of 8 held-out samples has both passing and failing samples)

No selection: a random sample from the pool passes 0.523 of the time.

## Hidden-test pass rate of the best-of-8 pick, selecting by k tests

| strategy | k=1 | k=2 | k=3 | k=5 | k=10 | k=25 |
|---|---|---|---|---|---|---|
| first k (benchmark order) | 0.811 | 0.852 | 0.881 | 0.929 | 0.956 | 0.988 |
| random k | 0.819 | 0.862 | 0.875 | 0.899 | 0.922 | 0.950 |
| extremes (shortest and longest inputs) | 0.712 | 0.874 | 0.889 | 0.908 | 0.940 | 0.970 |
| cross-fit greedy (catch fit-half failures) | 0.905 | 0.954 | 0.970 | 0.978 | 0.984 | 0.995 |

## Overfit rate at n=8: pick passes all k chosen tests but fails the hidden suite

| strategy | k=1 | k=2 | k=3 | k=5 | k=10 | k=25 |
|---|---|---|---|---|---|---|
| first k (benchmark order) | 0.189 | 0.148 | 0.119 | 0.071 | 0.044 | 0.012 |
| random k | 0.181 | 0.138 | 0.125 | 0.101 | 0.078 | 0.050 |
| extremes (shortest and longest inputs) | 0.288 | 0.126 | 0.111 | 0.092 | 0.060 | 0.030 |
| cross-fit greedy (catch fit-half failures) | 0.095 | 0.046 | 0.030 | 0.022 | 0.016 | 0.005 |

## A few tests vs an LLM judge (problems where the pool was judged)

Problems with judge scores for the pool: 142

| selector | n=1 | n=2 | n=4 | n=8 |
|---|---|---|---|---|
| random | 0.533 | 0.537 | 0.535 | 0.536 |
| judge A only | 0.533 | 0.680 | 0.769 | 0.824 |
| 1 random test + judge A tie-break | 0.533 | 0.709 | 0.828 | 0.908 |
| 3 random tests + judge A tie-break | 0.533 | 0.713 | 0.837 | 0.923 |
