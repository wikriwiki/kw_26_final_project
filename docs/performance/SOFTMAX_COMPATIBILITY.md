# Indexed Night matching: compatibility and complexity

The production scorer supplies an immutable list of unique agent pairs, string IDs,
finite float scores, integer degree cap, and a standard `random.Random`. The
optimized selector only accepts that contract and at least 256 rows. Other
private-helper inputs use `_softmax_select_scan`, the unchanged original body.

## Preserving the reference draw

Reference steps for the currently eligible rows, in their original order:

1. `s[i] = score[i] / temperature`, `m = max(s)`.
2. `w[i] = exp(s[i] - m)`; `W = sum(w)`.
3. `p[i] = w[i] / W`; `C = list(accumulate(p))`.
4. Draw one `u = rng.random()` and bisect at `u * (C[-1] + 0.0)` with
   upper index `len(C) - 1`.

The implementation recomputes weights whenever the eligible maximum changes.
Thus retained weights have exactly the same float values as step 2, including
underflowed zeros. In particular, retaining weights based on an already-removed
maximum would be incorrect for large score ranges; this implementation does not
do that. Duplicate pairs, self pairs, unusual RNGs, nonfinite scaled scores, and
non-integer caps use the original helper before any random draw.

The segment tree recomputes parents from children after a deletion, instead of
repeatedly subtracting from a total. Every selection uses the original row order.
Adjacent-edge indexes remove pairs when either endpoint reaches its cap. No
sampling/score threshold or candidate pruning is added.

## Floating-point boundary guard

Let n be the original row count and epsilon = `sys.float_info.epsilon` (twice
binary64 unit roundoff). Eligible weights are nonnegative and include a maximum
weight exactly 1; total weight lies in [1,n]. The fast path requires
`n * epsilon <= 1e-6`.

For positive summands, each serial sum has relative error bounded by
`gamma_n = n*u/(1-n*u)` where u=epsilon/2. Pairwise tree and prefix sums have
no greater bound than a serial n-term sum. Reference weight normalization and
cumulative addition add division and accumulation error; the final total/multiply
introduce another normalization error. Bounding these operations separately,
including prefix/total divisions on the indexed side, is comfortably below
`64*n*epsilon` in absolute normalized CDF position under the stated n limit.
Subnormal errors are absolute (at most one minimum-subnormal quantum per
operation); dividing by a total >=1 and summing at most n of these is negligible
within that margin. Exponential-library error is not compared to exact exp: both
algorithms use the same already-rounded exponential outputs.

Only if u is more than this margin inside BOTH boundaries of the tree-proposed
row is the indexed result accepted. Otherwise `_reference_draw` recomputes steps
2–4's sums/divisions/bisect over the active weights using the SAME u. It does not
draw again. Zero-weight and last-leaf boundary cases also take that reference
path when needed. The original per-choice random state advance is preserved.

Tests check exact selected objects/order and RNG state for 300 seeded graphs,
five temperatures including negative and near-underflow values, degree caps
1/2/5, adjacent representable floats around cumulative boundaries, uniform 0
and the greatest float below 1, maximum removal, and helper fallback cases.
These are compatibility checks, not a statistical goodness-of-fit substitute.

## Cost and limits

P = candidate pairs, S = selected pairs, R = number of distinct eligible maximum
phases, B = boundary-fallback draws. Initialization is O(P); the heap, edge
removal, and tree searches cost O((P+S) log P). Rebuilds cost at most O(RP), and
boundary replay at most O(BP). Total:

`O((P+S) log P + RP + BP)` time and O(P) auxiliary memory.

This is NOT an unconditional O(P log P) algorithm. When every selection removes
the maximum or every random draw falls near a boundary, worst-case work remains
O(SP). Retaining the original numerical contract is the reason for that tradeoff.
For equal-score inputs R=1, and ordinary nonboundary draws avoid rescanning the
candidate list. Counters and benchmarks report actual R and B rather than hiding
rebuild cost. The 1,024-agent/8,192-pair benchmark had 9 rebuilds, 0 boundary
fallbacks and 974 identical selections; median time changed 3.849s → 0.114s.

Reproduce with `python scripts/analysis/benchmark_matching.py --output PATH`.
This is a local synthetic CPU benchmark; no GPU or complete simulation speedup
is asserted. The source hashes and runtime are in `matching_benchmark_20260909.json`.
