"""Exact-rational leak localization on a tree network.

Model
-----
* Pipes are edges (u, v) with positive integer length.  They form a tree.
* Sensor ``i`` sits on node ``n_i`` and reports integer arrival time ``t_i``.
* A source at continuous position emits at unknown time ``t0`` (no prior).
* Arrival prediction:  t0 + tree_distance(source, n_i).

For a point at coordinate ``x`` measured from endpoint ``a`` of edge
``(a, b)`` (length ``L``), the path to a sensor either leaves through ``a``
(distance ``x + d(a, n_i)``, slope +1 in ``x``) or through ``b``
(distance ``(L - x) + d(b, n_i)``, slope -1 in ``x``).  With

    A_i = t_i - constant_i,   residual_i(x, t0) = A_i - slope_i * x - t0

the best ``t0`` for a fixed ``x`` is the midpoint of the residual range and

    g(x) = max_i residual_i - min_i residual_i

is convex piecewise linear on ``[0, L]``.  Its minimum is the edge optimum;
the global optimum is the minimum over edges.  All arithmetic uses
``fractions.Fraction`` so every optimum point / closed interval, emission
time and residual is exact.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

# Graph is represented internally by compact node indices.
Edge = Tuple[int, int, int]  # (u_index, v_index, length)
Sensor = Tuple[int, int]  # (node_index, observed_time)


def _frac_obj(value: Fraction) -> Dict[str, object]:
    """JSON-serialisable exact fraction plus a readable string."""
    num, den = value.numerator, value.denominator
    return {"num": num, "den": den, "text": str(num) if den == 1 else f"{num}/{den}"}


def _root_tree(
    node_count: int,
    edges: Sequence[Edge],
    root: int,
) -> Tuple[List[List[Tuple[int, int, int]]], List[int], List[int]]:
    """Return (adjacency, parent_node, parent_edge_index) rooted at ``root``."""
    adj: List[List[Tuple[int, int, int]]] = [[] for _ in range(node_count)]
    for k, (u, v, length) in enumerate(edges):
        adj[u].append((v, length, k))
        adj[v].append((u, length, k))

    parent = [-1] * node_count
    parent_edge = [-1] * node_count
    parent[root] = root

    stack = [root]
    order: List[int] = []
    while stack:
        node = stack.pop()
        order.append(node)
        for nxt, _length, edge_k in adj[node]:
            if nxt == parent[node]:
                continue
            parent[nxt] = node
            parent_edge[nxt] = edge_k
            stack.append(nxt)
    return adj, parent, parent_edge


def _subtree_intervals(
    node_count: int,
    adj: Sequence[Sequence[Tuple[int, int, int]]],
    root: int,
) -> Tuple[List[int], List[int]]:
    """Iterative DFS pre-order timer / subtree exit timer."""
    tin = [-1] * node_count
    tout = [-1] * node_count
    timer = 0
    # (node, parent, entered)
    stack: List[Tuple[int, int, bool]] = [(root, -1, False)]
    while stack:
        node, par, entered = stack.pop()
        if entered:
            tout[node] = timer - 1
            continue
        tin[node] = timer
        timer += 1
        stack.append((node, par, True))
        for nxt, _length, _edge_k in reversed(adj[node]):
            if nxt != par:
                stack.append((nxt, node, False))
    return tin, tout


def _distances_from(
    node_count: int,
    adj: Sequence[Sequence[Tuple[int, int, int]]],
    source: int,
) -> List[int]:
    """Exact integer distances from ``source`` to every node (tree DFS)."""
    dist = [-1] * node_count
    dist[source] = 0
    stack = [source]
    while stack:
        node = stack.pop()
        for nxt, length, _edge_k in adj[node]:
            if dist[nxt] == -1:
                dist[nxt] = dist[node] + length
                stack.append(nxt)
    return dist


def _edge_directional_extrema(
    node_count: int,
    adj: Sequence[Sequence[Tuple[int, int, int]]],
    parent: Sequence[int],
    root: int,
    sensors: Sequence[Sensor],
) -> List[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]:
    """Build the residual-line extrema on both sides of every rooted edge."""
    children: List[List[Tuple[int, int, int]]] = [[] for _ in range(node_count)]
    order: List[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        for nxt, length, edge_k in adj[node]:
            if parent[nxt] == node:
                children[node].append((nxt, length, edge_k))
                stack.append(nxt)

    observed_at: List[Optional[int]] = [None] * node_count
    for node, observed in sensors:
        observed_at[node] = observed

    down_max: List[Optional[int]] = [None] * node_count
    down_min: List[Optional[int]] = [None] * node_count
    for node in reversed(order):
        high = observed_at[node]
        low = observed_at[node]
        for child, length, _edge_k in children[node]:
            if down_max[child] is not None:
                value = down_max[child] - length
                high = value if high is None or value > high else high
            if down_min[child] is not None:
                value = down_min[child] - length
                low = value if low is None or value < low else low
        down_max[node] = high
        down_min[node] = low

    up_max: List[Optional[int]] = [None] * node_count
    up_min: List[Optional[int]] = [None] * node_count
    sides: List[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]] = [
        (None, None, None, None) for _ in range(max(0, node_count - 1))
    ]

    for node in order:
        outer_high = up_max[node]
        outer_low = up_min[node]
        if observed_at[node] is not None:
            value = observed_at[node]
            outer_high = value if outer_high is None or value > outer_high else outer_high
            outer_low = value if outer_low is None or value < outer_low else outer_low

        high = outer_high
        low = outer_low
        high_branch: Optional[int] = None
        low_branch: Optional[int] = None
        for child, length, _edge_k in children[node]:
            if down_max[child] is not None:
                value = down_max[child] - length
                if high is None or value > high:
                    high = value
                    high_branch = child
            if down_min[child] is not None:
                value = down_min[child] - length
                if low is None or value < low:
                    low = value
                    low_branch = child

        for child, length, edge_k in children[node]:
            parent_high = high if high_branch != child else outer_high
            parent_low = low if low_branch != child else outer_low
            child_high = None if down_max[child] is None else down_max[child] - length
            child_low = None if down_min[child] is None else down_min[child] - length
            sides[edge_k] = (parent_high, parent_low, child_high, child_low)
            up_max[child] = None if parent_high is None else parent_high - length
            up_min[child] = None if parent_low is None else parent_low - length

    return sides


def _edge_minimum(
    length: int,
    p_max: Optional[int],
    p_min: Optional[int],
    m_max: Optional[int],
    m_min: Optional[int],
) -> Tuple[Fraction, List[Tuple[Fraction, Fraction, List[Tuple[Fraction, Fraction]]]]]:
    """Minimise g(x) on one edge.

    P-lines (sensor reached through the parent endpoint, slope +1 in the
    distance, hence residual line slope -1):  residual = A - x.
    M-lines (sensor reached through the child endpoint): residual = A + x.

    ``upper(x) = max`` of active lines, ``lower(x) = min`` of active lines,
    ``g = upper - lower``.  Returns (g*, [(x_lo, x_hto, t0_pieces)]) where a
    t0 piece is (x_from, x_to, t0_from, t0_to).
    """
    upper: List[Tuple[int, int]] = []
    lower: List[Tuple[int, int]] = []
    if p_max is not None:
        upper.append((-1, p_max))
        lower.append((-1, p_min))
    if m_max is not None:
        upper.append((1, m_max))
        lower.append((1, m_min))

    L = Fraction(length)

    def value(lines: Sequence[Tuple[int, int]], x: Fraction) -> Fraction:
        result = None
        for slope, intercept in lines:
            cand = intercept + slope * x
            if result is None or cand > result:
                result = cand
        return result  # type: ignore[return-value]

    def upper_at(x: Fraction) -> Fraction:
        return value(upper, x)

    def lower_at(x: Fraction) -> Fraction:
        result = None
        for slope, intercept in lower:
            cand = intercept + slope * x
            if result is None or cand < result:
                result = cand
        return result  # type: ignore[return-value]

    # Kinks: intersection of the -1 line and the +1 line, if both exist.
    kinks: List[Fraction] = []
    if len(upper) == 2:
        (s1, c1), (s2, c2) = upper
        kinks.append(Fraction(c1 - c2, s2 - s1))  # (a - b) / 2
    if len(lower) == 2:
        (s1, c1), (s2, c2) = lower
        kinks.append(Fraction(c1 - c2, s2 - s1))

    points = {Fraction(0), L}
    for kink in kinks:
        if 0 < kink < L:
            points.add(kink)
    ordered = sorted(points)

    g_values = [upper_at(x) - lower_at(x) for x in ordered]
    g_star = min(g_values)

    intervals: List[Tuple[Fraction, Fraction, List[Tuple[Fraction, Fraction]]]] = []
    i = 0
    while i < len(ordered):
        if g_values[i] != g_star:
            i += 1
            continue
        j = i
        while j + 1 < len(ordered) and g_values[j + 1] == g_star:
            j += 1
        x_lo, x_hi = ordered[i], ordered[j]

        # t0(x) = (upper(x) + lower(x)) / 2 is piecewise linear; split at any
        # interior kink so the returned emission-time pieces stay linear.
        breaks = [x_lo]
        for kink in kinks:
            if x_lo < kink < x_hi:
                breaks.append(kink)
        breaks.append(x_hi)
        pieces: List[Tuple[Fraction, Fraction]] = []
        for bp, bq in zip(breaks, breaks[1:]):
            t0_p = (upper_at(bp) + lower_at(bp)) / 2
            t0_q = (upper_at(bq) + lower_at(bq)) / 2
            pieces.append((bp, bq, t0_p, t0_q))  # type: ignore[arg-type]
        intervals.append((x_lo, x_hi, pieces))  # type: ignore[arg-type]
        i = j + 1

    return g_star, intervals


def solve(
    node_ids: Sequence[int],
    edges: Sequence[Edge],
    sensors: Sequence[Sensor],
) -> Dict[str, object]:
    """Compute the L-infinity optimum over the whole tree.

    ``edges`` keeps the user-supplied order and orientation
    ``(u_index, v_index, length)``; coordinates in the result are measured
    from that first-listed ("head") endpoint.
    """
    n = len(node_ids)
    root = sensors[0][0]
    adj, parent, parent_edge = _root_tree(n, edges, root)
    tin, tout = _subtree_intervals(n, adj, root)

    sensor_nodes = [node for node, _time in sensors]
    sensor_times = [time for _node, time in sensors]
    edge_sides = _edge_directional_extrema(n, adj, parent, root, sensors)

    best: Optional[Fraction] = None
    per_edge: List[Tuple[Fraction, List[Tuple[Fraction, Fraction, list]]]] = []

    for k, (u, v, length) in enumerate(edges):
        p_max, p_min, m_max, m_min = edge_sides[k]
        g_star, intervals = _edge_minimum(length, p_max, p_min, m_max, m_min)
        per_edge.append((g_star, intervals))
        if best is None or g_star < best:
            best = g_star

    assert best is not None
    # With t0 free, max absolute residual at the optimal midpoint t0 is half
    # the residual span g*; minimizer set is unchanged.
    best_half = best / 2

    # Canonical solution: earliest edge in input order attaining g*, smallest
    # coordinate measured from the user-supplied head endpoint.
    canonical_edge = next(k for k, (g_star, _iv) in enumerate(per_edge) if g_star == best)
    u, v, length = edges[canonical_edge]
    child = v if parent_edge[v] == canonical_edge else u
    head = u if child == v else v
    internal_lo = per_edge[canonical_edge][1][0][0]
    x_canon_internal = internal_lo
    x_canon = x_canon_internal if head == u else Fraction(length) - x_canon_internal

    # Residuals at the canonical point (computed in internal orientation).
    a_node, b_node = head, child
    distance_from_head = _distances_from(n, adj, a_node)
    canonical_distances: List[Fraction] = []
    residuals_raw: List[Fraction] = []
    for snode, stime in zip(sensor_nodes, sensor_times):
        inside = tin[b_node] <= tin[snode] <= tout[b_node]
        distance = Fraction(distance_from_head[snode])
        distance += -x_canon_internal if inside else x_canon_internal
        canonical_distances.append(distance)
        residuals_raw.append(Fraction(stime) - distance)
    t0_canon = (max(residuals_raw) + min(residuals_raw)) / 2

    residual_rows: List[Dict[str, object]] = []
    positive_witnesses: List[int] = []
    negative_witnesses: List[int] = []
    for i, (snode, stime) in enumerate(zip(sensor_nodes, sensor_times)):
        distance = canonical_distances[i]
        predicted = t0_canon + distance
        residual = Fraction(stime) - predicted
        residuals_raw_check = residual
        assert residuals_raw_check == residuals_raw[i] - t0_canon
        role = None
        if residual == best_half:
            role = "positive"
            positive_witnesses.append(i)
        elif residual == -best_half:
            role = "negative"
            negative_witnesses.append(i)
        residual_rows.append(
            {
                "sensor_index": i,
                "node": node_ids[snode],
                "observed": stime,
                "distance": _frac_obj(distance),
                "predicted": _frac_obj(predicted),
                "residual": _frac_obj(residual),
                "abs_residual": _frac_obj(abs(residual)),
                "extremal": role,
            }
        )

    # All co-optimal intervals, converted to user orientation (head = u).
    optima: List[Dict[str, object]] = []
    for k, (g_star, intervals) in enumerate(per_edge):
        if g_star != best:
            continue
        eu, ev, elen = edges[k]
        echild = ev if parent_edge[ev] == k else eu
        ehead = eu if echild == ev else ev
        for lo, hi, pieces in intervals:
            if ehead == eu:
                start, end = lo, hi
                mapped_pieces = [
                    {
                        "x_start": _frac_obj(xp),
                        "x_end": _frac_obj(xq),
                        "t0_start": _frac_obj(tp),
                        "t0_end": _frac_obj(tq),
                    }
                    for xp, xq, tp, tq in pieces
                ]
            else:
                start, end = Fraction(elen) - hi, Fraction(elen) - lo
                mapped_pieces = [
                    {
                        "x_start": _frac_obj(Fraction(elen) - xq),
                        "x_end": _frac_obj(Fraction(elen) - xp),
                        "t0_start": _frac_obj(tq),
                        "t0_end": _frac_obj(tp),
                    }
                    for xp, xq, tp, tq in reversed(pieces)
                ]
            optima.append(
                {
                    "edge_index": k,
                    "from_node": node_ids[eu],
                    "to_node": node_ids[ev],
                    "length": elen,
                    "coordinate_start": _frac_obj(start),
                    "coordinate_end": _frac_obj(end),
                    "point": start == end,
                    "t0_segments": mapped_pieces,
                }
            )

    return {
        "optimal_value": _frac_obj(best_half),
        "canonical": {
            "edge_index": canonical_edge,
            "from_node": node_ids[u],
            "to_node": node_ids[v],
            "length": length,
            "coordinate": _frac_obj(x_canon),
            "emission_time": _frac_obj(t0_canon),
        },
        "optima": optima,
        "residuals": residual_rows,
        "witnesses": {
            "positive": positive_witnesses,
            "negative": negative_witnesses,
        },
        "node_count": n,
        "edge_count": len(edges),
        "sensor_count": len(sensors),
    }
