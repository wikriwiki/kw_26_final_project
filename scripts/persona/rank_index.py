"""Exact rank selection after UUID exclusion, using integer prefix counts.

Pools keep their original stable order. No float weights, random draws, or
approximate quantiles are introduced. Instances belong to one build only.
"""
from collections import defaultdict


class _ActivePool:
    def __init__(self, records, used, owners):
        self.records = records
        self.tree = [0] * (len(records) + 1)
        self.remaining = 0
        for i, record in enumerate(records, 1):
            uuid = record.get("uuid")
            active = int(uuid not in used)
            self.tree[i] += active
            self.remaining += active
            parent = i + (i & -i)
            if parent < len(self.tree):
                self.tree[parent] += self.tree[i]
            if active and uuid:
                owners[uuid].append((self, i))

    def remove(self, i):
        self.remaining -= 1
        while i < len(self.tree):
            self.tree[i] -= 1
            i += i & -i

    def at_percentile(self, percentile):
        rank = round(percentile * (self.remaining - 1)) if self.remaining > 1 else 0
        if rank < 0:
            rank += self.remaining
        if not 0 <= rank < self.remaining:
            raise IndexError("list index out of range")
        index = 0
        step = 1 << (len(self.records).bit_length() - 1)
        # Find the zero-based record index of the (rank + 1)-th live row.
        while step:
            next_index = index + step
            if next_index < len(self.tree) and self.tree[next_index] <= rank:
                rank -= self.tree[next_index]
                index = next_index
            step >>= 1
        return self.records[index]


class RankMatcher:
    """Lazily index fallback pools and remove a selected UUID from every pool."""

    def __init__(self, pool_index, all_sorted, age_neighbors, score_key):
        self.pool_index = pool_index
        self.all_sorted = all_sorted
        self.age_neighbors = age_neighbors
        self.score_key = score_key
        self.used = set()
        self.pools = {}
        self.owners = defaultdict(list)

    def mark_used(self, uuid):
        # Existing build only excludes truthy UUIDs. Blank/missing UUID rows
        # remain reusable; duplicate nonempty UUIDs are all excluded together.
        if not uuid or uuid in self.used:
            return
        self.used.add(uuid)
        for pool, index in self.owners.pop(uuid, ()):
            pool.remove(index)

    def _pool(self, key, factory):
        if key not in self.pools:
            self.pools[key] = _ActivePool(factory(), self.used, self.owners)
        return self.pools[key]

    def _merged(self, predicate):
        # Same concatenation and stable sort as the original fallback; sorting
        # the global pool instead would change the order of tied SES scores.
        rows = [r for cell, records in self.pool_index.items() if predicate(cell) for r in records]
        rows.sort(key=self.score_key)
        return rows

    def pick(self, cell, percentile):
        _, sex, age = cell
        pool = self._pool(("cell", cell), lambda: self.pool_index.get(cell) or [])
        if pool.remaining:
            return pool.at_percentile(percentile), "gu_sex_age"

        pool = self._pool(("sex_age", sex, age), lambda: self._merged(lambda c: c[1] == sex and c[2] == age))
        if pool.remaining:
            return pool.at_percentile(percentile), "sex_age"

        adjacent = set(self.age_neighbors.get(age, []))
        if adjacent:
            pool = self._pool(("adjacent", sex, age), lambda: self._merged(lambda c: c[1] == sex and c[2] in adjacent))
            if pool.remaining:
                return pool.at_percentile(percentile), "sex_adjacent_age"

        pool = self._pool(("sex", sex), lambda: self._merged(lambda c: c[1] == sex))
        if pool.remaining:
            return pool.at_percentile(percentile), "sex_only"

        pool = self._pool(("all",), lambda: self.all_sorted)
        if pool.remaining:
            return pool.at_percentile(percentile), "any_emergency"
        # Original emergency path deliberately allows reuse after exhaustion.
        rank = round(percentile * (len(self.all_sorted) - 1)) if len(self.all_sorted) > 1 else 0
        return self.all_sorted[rank], "any_emergency"
