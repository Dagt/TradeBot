from __future__ import annotations

from typing import Dict, List, Tuple


class CorrelationGuard:
    """Group highly correlated symbols and compute a global position cap.

    The guard builds clusters of symbols whose absolute correlation exceeds
    a threshold.  The largest cluster determines the reduction factor applied
    to a base limit.  For example, if three symbols are mutually correlated
    over the threshold, the global cap becomes ``base_limit / 3``.
    """

    def __init__(self, base_limit: float) -> None:
        self.base_limit = float(base_limit)

    def group_and_cap(
        self, pairs: Dict[Tuple[str, str], float], threshold: float
    ) -> tuple[List[List[str]], float]:
        """Return correlated groups and the resulting global cap.

        Parameters
        ----------
        pairs:
            Mapping of ``(symbol_a, symbol_b)`` to correlation coefficient.
        threshold:
            Absolute correlation value above which symbols are considered
            linked.

        Returns
        -------
        groups:
            A list of correlated symbol groups (each sorted alphabetically).
        cap:
            The global cap computed as ``base_limit / largest_group_size``.
        """

        graph: Dict[str, set[str]] = {}
        for (a, b), corr in pairs.items():
            if abs(corr) >= threshold:
                graph.setdefault(a, set()).add(b)
                graph.setdefault(b, set()).add(a)

        visited: set[str] = set()
        groups: List[List[str]] = []
        for sym in graph:
            if sym in visited:
                continue
            stack = [sym]
            group: List[str] = []
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                group.append(node)
                for neigh in graph.get(node, []):
                    if neigh not in visited:
                        stack.append(neigh)
            if len(group) > 1:
                groups.append(sorted(group))

        largest = max((len(g) for g in groups), default=1)
        cap = self.base_limit / largest
        return groups, cap
