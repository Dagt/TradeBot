from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

SymbolPair = Tuple[str, str]


def group_correlated(symbol_pairs: Dict[SymbolPair, float], threshold: float) -> List[Set[str]]:
    """Group symbols connected by correlations above ``threshold``.

    Parameters
    ----------
    symbol_pairs:
        Mapping of ``(sym_a, sym_b)`` to correlation value.
    threshold:
        Minimum absolute correlation to consider two symbols linked.

    Returns
    -------
    list[set[str]]
        A list with one ``set`` per correlated group.  Only symbols linked by
        correlations exceeding ``threshold`` are included.  Uncorrelated
        symbols are omitted.
    """
    adjacency: Dict[str, Set[str]] = {}
    for (a, b), corr in symbol_pairs.items():
        if abs(corr) < threshold:
            continue
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    visited: Set[str] = set()
    groups: List[Set[str]] = []
    for node in adjacency:
        if node in visited:
            continue
        stack = [node]
        comp: Set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            stack.extend(adjacency.get(cur, ()))
        if comp:
            groups.append(comp)
    return groups


def global_cap(groups: Iterable[Set[str]], base_cap: float) -> float:
    """Compute global position cap given correlated groups.

    ``base_cap`` is divided by the size of the largest group.  When no groups
    are provided or all have a single element, ``base_cap`` is returned
    unchanged.
    """
    max_group = max((len(g) for g in groups), default=1)
    if max_group <= 1:
        return float(base_cap)
    return float(base_cap) / float(max_group)
