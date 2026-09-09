"""Indexed softmax matching with reference-compatible floating boundaries.

Weights are recomputed using the original exp(score / T - max_score / T)
whenever the eligible maximum changes. Within a maximum phase they are fixed.
Candidate order and one random() call per choice are retained. Near a cumulative
probability boundary, the original normalization/accumulation is replayed with
that SAME uniform draw. See docs/performance/SOFTMAX_COMPATIBILITY.md.
"""
from __future__ import annotations

import heapq
import math
import random
import sys
from bisect import bisect
from collections import defaultdict
from itertools import accumulate


class _WeightTree:
    def __init__(self, weights):
        self.size = 1 << (len(weights) - 1).bit_length()
        self.values = [0.0] * (2 * self.size)
        self.values[self.size:self.size + len(weights)] = weights
        for i in range(self.size - 1, 0, -1):
            self.values[i] = self.values[2 * i] + self.values[2 * i + 1]

    @property
    def total(self):
        return self.values[1]

    def remove(self, index):
        i = self.size + index
        self.values[i] = 0.0
        i //= 2
        while i:
            # Recompute parents; subtraction updates accumulate cancellation
            # error over the whole run, invalidating the boundary bound.
            self.values[i] = self.values[2 * i] + self.values[2 * i + 1]
            i //= 2

    def choose(self, target):
        i = 1
        while i < self.size:
            left = self.values[2 * i]
            if target < left:
                i *= 2
            else:
                target -= left
                i = 2 * i + 1
        return i - self.size

    def prefix(self, stop):
        left, right = self.size, self.size + stop
        total = 0.0
        while left < right:
            if left & 1:
                total += self.values[left]
                left += 1
            if right & 1:
                right -= 1
                total += self.values[right]
            left //= 2
            right //= 2
        return total


def _reference_draw(active, weights, uniform):
    indices = [i for i, live in enumerate(active) if live]
    exps = [weights[i] for i in indices]
    total_exp = sum(exps)
    probs = [e / total_exp for e in exps]
    cumulative = list(accumulate(probs))
    # Exactly Random.choices(range(n), weights=probs, k=1), without a second
    # random draw. Its upper search bound excludes the final cumulative entry.
    picked = bisect(cumulative, uniform * (cumulative[-1] + 0.0), 0, len(indices) - 1)
    return indices[picked]


def indexed_softmax_select(scored, limit, temperature, rng, reference, *, stats=None):
    """Use the indexed path only for the normal immutable scored-row contract.

    Unusual private-helper inputs keep the original behavior, including custom
    RNG choices, duplicate rows (list.remove equality), and nonfinite values.
    """
    if (len(scored) < 256 or type(limit) is not int or limit <= 0
            or type(temperature) not in (float, int) or not math.isfinite(temperature)
            or temperature == 0 or type(rng) is not random.Random
            or getattr(rng.choices, "__func__", None) is not random.Random.choices):
        return reference(scored, limit, temperature, rng)

    scaled, endpoints, seen = [], [], set()
    for row in scored:
        if type(row) is not dict:
            return reference(scored, limit, temperature, rng)
        a, b, score = row.get("aid_a"), row.get("aid_b"), row.get("score")
        if (type(a) is not str or type(b) is not str or a == b
                or (a, b) in seen or type(score) not in (float, int)):
            return reference(scored, limit, temperature, rng)
        value = score / temperature
        if not math.isfinite(value):
            return reference(scored, limit, temperature, rng)
        seen.add((a, b))
        endpoints.append((a, b))
        scaled.append(value)

    n = len(scored)
    # Positive summands, n*epsilon << 1. This conservative absolute CDF bound
    # covers reference normalization/division, serial accumulation, multiply,
    # and tree/prefix summation. Reject large n instead of extending the proof.
    epsilon = sys.float_info.epsilon
    if n * epsilon > 1e-6:
        return reference(scored, limit, temperature, rng)
    margin = 64.0 * n * epsilon
    counters = stats if stats is not None else {}
    counters.update(rebuilds=0, weight_evaluations=0, tree_draws=0, reference_draws=0, removed=0)
    adjacent = defaultdict(list)
    for i, (a, b) in enumerate(endpoints):
        adjacent[a].append(i)
        adjacent[b].append(i)
    active = bytearray([1]) * n
    maximum_heap = [(-score, i) for i, score in enumerate(scaled)]
    heapq.heapify(maximum_heap)
    counts = defaultdict(int)
    selected = []
    current_max = None
    weights, tree = [], None

    def remove(i):
        if active[i]:
            active[i] = 0
            tree.remove(i)
            counters["removed"] += 1

    while maximum_heap:
        while maximum_heap and not active[maximum_heap[0][1]]:
            heapq.heappop(maximum_heap)
        if not maximum_heap:
            break
        maximum = -maximum_heap[0][0]
        if maximum != current_max:
            weights = [math.exp(score - maximum) if active[i] else 0.0 for i, score in enumerate(scaled)]
            tree = _WeightTree(weights)
            current_max = maximum
            counters["rebuilds"] += 1
            counters["weight_evaluations"] += sum(active)
        uniform = rng.random()
        i = tree.choose(uniform * tree.total)
        if (i < n and active[i]
                and tree.prefix(i) / tree.total + margin < uniform
                and uniform < tree.prefix(i + 1) / tree.total - margin):
            counters["tree_draws"] += 1
        else:
            i = _reference_draw(active, weights, uniform)
            counters["reference_draws"] += 1
        selected.append(scored[i])
        remove(i)
        for aid in endpoints[i]:
            counts[aid] += 1
            if counts[aid] >= limit:
                for edge in adjacent.pop(aid):
                    remove(edge)
    return selected
