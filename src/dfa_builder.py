"""
DFA construction for priority-slot symmetry breaking.

Two steps:
  - a topological sort that produces a total order on the operations;
  - a DFA construction from that total order and the non-overlapping graph.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field

# Label used for idle (epsilon) arcs into the absorbing sink state when a DFA is
# built with allow_idle=True.  Chosen so it can never collide with an operation
# name in W (the SBS flow constraints only sum arcs whose label is in W).
IDLE_LABEL = "__idle__"


@dataclass
class DFA:
    """
    Deterministic Finite Automaton for NOPSOS symmetry breaking.

    States are integers: 0 is always the initial state q0,
    and state (i+1) corresponds to operation order[i].

    All states are final (accepting), so any path of length n from
    state 0 is accepted.

    Idle sink (optional, allow_idle=True)
    -------------------------------------
    When built with ``allow_idle=True`` the DFA gains one absorbing state
    ``idle_state`` reached by an ``IDLE_LABEL`` arc from every operation state
    (plus a self-loop).  This lets an accepted word of fixed length n end in a
    run of idle steps, so the number of *real* operations can be fewer than n.
    In the MIP this turns n_slots into an upper bound on the schedule length
    rather than an exact count, without reintroducing slot symmetry (the sink is
    absorbing, so idle steps are forced to the tail of the word).
    """
    states: list          # integer state IDs; 0 = initial (q0)
    initial: int          # always 0
    arcs: list            # list of (from_state, to_state, label); label in W or IDLE_LABEL
    state_of: dict        # operation -> integer state ID
    operation_order: list # operations in total order ≺ (list, first = smallest)
    idle_state: int = None  # absorbing sink state ID, or None if allow_idle=False
    final_states: list = None  # accepting states; None means "all states are final"

    def final(self):
        """The set of accepting states (all states when final_states is None)."""
        return list(self.states) if self.final_states is None else list(self.final_states)

    @property
    def n_states(self):
        return len(self.states)

    @property
    def n_arcs(self):
        return len(self.arcs)

    def arcs_from(self, state):
        return [(f, t, lab) for f, t, lab in self.arcs if f == state]

    def arcs_to(self, state):
        return [(f, t, lab) for f, t, lab in self.arcs if t == state]

    def summary(self):
        r1, r2, r3, r4 = dfa_arc_counts(self)
        n_idle = sum(1 for _, _, lab in self.arcs if lab == IDLE_LABEL)
        idle = f", Idle={n_idle}" if n_idle else ""
        loops = f", Rule4={r4}" if r4 else ""
        return (f"DFA: {self.n_states} states, {self.n_arcs} arcs "
                f"(Rule1={r1}, Rule2={r2}, Rule3={r3}{loops}{idle})")


def topological_sort(W, prec_edges, rng=None):
    """
    Algorithm 1: total order on W extending the precedence partial order.

    Free-start operations (no predecessors) come first. Among operations
    at the same precedence level, a fixed alphabetical order is used for
    determinism. Any valid topological sort suffices for correctness.

    Parameters
    ----------
    W : list[str]
        Set of operations (any hashable, unique elements).
    prec_edges : list[tuple]
        Directed precedence arcs (v, w): v must finish before w starts.
    rng : random.Random, optional
        When given, ties are broken by shuffling instead of alphabetically,
        which yields a different (still valid) topological order per seed.
        Used only to measure how much the arbitrary choice in Algorithm 1
        affects solver performance; the automaton's size is invariant.

    Returns
    -------
    list[str]
        Total order of W from smallest (≺-first) to largest.

    Raises
    ------
    ValueError
        If the precedence graph contains a directed cycle.
    """
    in_degree = {w: 0 for w in W}
    successors = defaultdict(list)
    for v, w in prec_edges:
        in_degree[w] += 1
        successors[v].append(w)

    def _tie_break(items):
        items = sorted(items)          # sorted first, so a seed is reproducible
        if rng is not None:
            rng.shuffle(items)
        return items

    queue = deque(_tie_break(w for w in W if in_degree[w] == 0))
    order = []

    while queue:
        w = queue.popleft()
        order.append(w)
        for v in _tie_break(successors[w]):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != len(W):
        raise ValueError(
            "Precedence graph has a directed cycle; topological sort impossible."
        )
    return order


def build_dfa(W, prec_edges, no_edges, allow_idle=False, repeatable=None,
              rng=None):
    """
    Algorithm 2: DFA for NOPSOS symmetry breaking.

    State 0 = initial state q0.
    State state_of[w] = q_w for each operation w.
    All states are final.

    Arc rules:
      Rule 1  q0  -> q_w     for every w in W          (label: w)
      Rule 2  q_w -> q_v     for every w ≺ v           (label: v)
      Rule 3  q_w -> q_v     for v in NO(w), v ≺ w     (label: v)
      Rule 4  q_w -> q_w     for w with eta_w >= 2        (label: w)

    Parameters
    ----------
    W : list[str]
        Operations.
    prec_edges : list[tuple[str, str]]
        Directed precedence arcs (v, w).
    no_edges : list
        Non-overlapping edges, each either a 2-tuple (v, w) or a
        frozenset/set {v, w}.  Pass each undirected edge once.
    repeatable : iterable or True, optional
        Operations whose execution cap allows re-execution (eta_w >= 2).
        Each gets a self-loop (Rule 4).  Without it the DFA rejects every
        word placing two executions of w in consecutive slots, which
        excludes feasible schedules whenever w is repeatable; the loop adds
        no symmetric duplicate, since it is the only way to put two
        executions of w in adjacent slots.  Pass True for "all of W".
        Default None = single-execution setting, no loops.
    allow_idle : bool, default False
        If True, add an absorbing idle sink (see the DFA docstring): an
        IDLE_LABEL arc from every operation state to a new sink state plus a
        sink self-loop.  This lets a fixed-length-n accepted word end in idle
        steps, so the schedule may use fewer than n operations.  The sink is
        absorbing, so idle steps are confined to the tail and no slot symmetry
        is reintroduced.  Idle arcs deliberately do NOT originate at q0 (every
        schedule uses slot 1 for a real operation).

    Returns
    -------
    DFA
    """
    order = topological_sort(W, prec_edges, rng=rng)
    rank = {w: i for i, w in enumerate(order)}  # 0-indexed rank in ≺

    # Symmetric neighbourhood for non-overlapping graph
    no_nbrs = defaultdict(set)
    for e in no_edges:
        if isinstance(e, (set, frozenset)):
            v, w = tuple(e)
        else:
            v, w = e
        no_nbrs[v].add(w)
        no_nbrs[w].add(v)

    # States: 0 = q0, then 1 .. |W| for operations in order
    state_of = {w: i + 1 for i, w in enumerate(order)}
    states = list(range(len(W) + 1))
    initial = 0

    arcs = []

    # Rule 1: q0 -> q_w  for each w
    for w in order:
        arcs.append((initial, state_of[w], w))

    # Rule 2: q_w -> q_v  whenever w ≺ v
    for i, w in enumerate(order):
        for j in range(i + 1, len(order)):
            v = order[j]
            arcs.append((state_of[w], state_of[v], v))

    # Rule 3: q_w -> q_v  whenever v in NO(w) and v ≺ w
    for w in order:
        for v in sorted(no_nbrs[w]):
            if rank[v] < rank[w]:
                arcs.append((state_of[w], state_of[v], v))

    # Rule 4: q_w -> q_w for every repeatable operation (eta_w >= 2).
    if repeatable:
        rep = set(order) if repeatable is True else set(repeatable)
        for w in order:
            if w in rep:
                arcs.append((state_of[w], state_of[w], w))

    # Optional absorbing idle sink: lets the word end in idle steps so a
    # schedule may use fewer than n operations (n_slots becomes an upper bound).
    idle_state = None
    if allow_idle:
        if IDLE_LABEL in state_of:
            raise ValueError(f"operation name collides with idle label {IDLE_LABEL!r}")
        idle_state = len(W) + 1
        states.append(idle_state)
        for w in order:                                  # idle from each op state
            arcs.append((state_of[w], idle_state, IDLE_LABEL))
        arcs.append((idle_state, idle_state, IDLE_LABEL))  # absorbing self-loop

    return DFA(
        states=states,
        initial=initial,
        arcs=arcs,
        state_of=state_of,
        operation_order=order,
        idle_state=idle_state,
    )


def dfa_arc_counts(dfa):
    """
    Return arc counts split by rule for reporting.
    Rule 1: arcs from state 0.
    Rule 3: backward arcs (destination has smaller rank than source).
    Rule 4: self-loops on repeatable operations.
    Rule 2: the rest.
    """
    op_of = {v: k for k, v in dfa.state_of.items()}
    rank = {w: i for i, w in enumerate(dfa.operation_order)}

    r1, r2, r3, r4 = 0, 0, 0, 0
    for f, t, lab in dfa.arcs:
        if lab == IDLE_LABEL:
            continue                       # idle/sink arcs are not Rule 1/2/3/4
        if f == dfa.initial:
            r1 += 1
        elif f == t:
            r4 += 1
        elif t != dfa.initial and rank[lab] < rank[op_of[f]]:
            r3 += 1
        else:
            r2 += 1
    return r1, r2, r3, r4


def forbidden_pairs(dfa):
    """Ordered pairs (w, v) of operations that may not occupy consecutive slots.

    The DFA of build_dfa has one state per operation, so the transition out of
    q_w depends only on w: its language is 2-local (strictly locally testable of
    order 2) and acceptance is equivalent to forbidding these pairs.  Returned
    as a sorted list so the AMPL data is deterministic.  Idle arcs are ignored;
    the caller handles idle slots separately.
    """
    allowed = {(f, lab) for f, t, lab in dfa.arcs if lab != IDLE_LABEL}
    ops = list(dfa.operation_order)
    return sorted((w, v) for w in ops for v in ops
                  if (dfa.state_of[w], v) not in allowed)


def accepts(dfa, word):
    """
    Return True if the DFA accepts the given word (list of operations).
    Useful for testing and debugging.
    """
    arc_map = defaultdict(dict)
    for f, t, lab in dfa.arcs:
        arc_map[f][lab] = t

    state = dfa.initial
    for sym in word:
        nxt = arc_map[state].get(sym)
        if nxt is None:
            return False
        state = nxt
    return True  # all states are final


def plot_dfa(
    dfa,
    output_path=None,
    title=None,
    show_labels=True,
    figsize=None,
    dpi=160,
):
    """
    Plot a DFA network using matplotlib.

    States are placed left-to-right in the DFA operation order. Forward arcs
    are drawn above the nodes and backward NO arcs below the nodes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    op_of_state = {state: op for op, state in dfa.state_of.items()}
    rank = {op: i for i, op in enumerate(dfa.operation_order)}

    ordered_states = [dfa.initial] + [dfa.state_of[op] for op in dfa.operation_order]
    pos = {state: (i, 0.0) for i, state in enumerate(ordered_states)}

    if figsize is None:
        figsize = (max(10, 1.1 * len(ordered_states)), 6)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title or f"DFA network ({dfa.n_states} states, {dfa.n_arcs} arcs)")
    ax.axis("off")

    def arc_style(from_state, to_state, label):
        if from_state == dfa.initial:
            return 0.22, "#6b7280", 0.9
        source_op = op_of_state.get(from_state)
        is_backward = source_op is not None and rank[label] < rank[source_op]
        if is_backward:
            return -0.34, "#d97706", 1.25
        distance = abs(pos[to_state][0] - pos[from_state][0])
        return min(0.12 + 0.035 * distance, 0.42), "#2563eb", 0.8

    for from_state, to_state, label in dfa.arcs:
        if label == IDLE_LABEL:
            continue                       # idle/sink arcs are not drawn
        rad, color, linewidth = arc_style(from_state, to_state, label)
        arrow = FancyArrowPatch(
            pos[from_state],
            pos[to_state],
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            alpha=0.55,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=18,
            shrinkB=18,
        )
        ax.add_patch(arrow)

        if show_labels:
            x1, y1 = pos[from_state]
            x2, y2 = pos[to_state]
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 + (1.25 * rad)
            ax.text(
                mx,
                my,
                label,
                fontsize=7,
                color=color,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
            )

    for state in ordered_states:
        x, y = pos[state]
        is_initial = state == dfa.initial
        ax.scatter(
            [x],
            [y],
            s=760 if is_initial else 650,
            color="#111827" if is_initial else "#f8fafc",
            edgecolor="#111827",
            linewidth=1.6,
            zorder=5,
        )
        label = "q0" if is_initial else f"q{state}"
        ax.text(
            x,
            y + 0.03,
            label,
            color="white" if is_initial else "#111827",
            fontsize=10,
            ha="center",
            va="center",
            weight="bold",
            zorder=6,
        )
        if not is_initial:
            ax.text(
                x,
                y - 0.38,
                op_of_state[state],
                color="#374151",
                fontsize=9,
                ha="center",
                va="top",
            )

    ax.set_xlim(-0.8, len(ordered_states) - 0.2)
    ax.set_ylim(-2.3, 2.4)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig, ax
