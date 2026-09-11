"""
SOS+RE symmetry breaker: a regular-expression automaton.

This module builds the regular-expression symmetry breaker.  For each
distillation program (DP) l it builds the regular-expression block

    L_l = (w_{i1} ... w_{id})            # the d charges of the DP, in order
          * prod_{w in W_T(l)} (eps + w)  # each compatible transfer, optional
          * prod_{v in W_U} prod_{w in delta_NO(v)} (eps + v + v.w)

where W_T(l) are the transfers compatible with l and, for an unload v,
delta_NO(v) = { w in W_T(l) : the transfer w draws from the storage tank that v
fills }.  The full language is

    RL = sum_{ {l,l'} neighbour DPs } (eps + L_l) (L_l L_{l'})^* (eps + L_{l'})

(neighbour = the DPs differ in exactly one CDU assignment).  This reproduces
the hand-crafted expressions on the reference topologies; e.g. for the
first reference instance it yields  RL = (eps + L_7)(L_7 L_8)^*(eps + L_8)  with
L_7 = 7(eps+4)(eps+6)(eps+1+14)(eps+2+26) and the analogous L_8.

The regular expression is compiled to a *minimal DFA* (Thompson construction ->
subset determinization -> Moore minimisation) so the SBS state count is a fair,
canonical figure to compare against the NOPSOS DFA -- not an inflated NFA.  An
absorbing idle sink is attached to the accept states so the schedule may use
fewer than n_slots operations (matching the NOPSOS idle relaxation); the model's
FlowEnd constraint then requires the slot-n state to be accepting.
"""

from collections import defaultdict, deque
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

from .crude_oil_ampl import CrudeOilFullInstance
from .crude_oil import enumerate_distillation_programs, compatible_transfers
from .dfa_builder import DFA, IDLE_LABEL


# ---------------------------------------------------------------------------
# Thompson NFA with epsilon transitions
# ---------------------------------------------------------------------------

class _Frag:
    """An NFA fragment: a single start state and a single accept state."""
    __slots__ = ("start", "accept")

    def __init__(self, start: int, accept: int):
        self.start = start
        self.accept = accept


class _NFA:
    def __init__(self):
        self.n = 0
        self.eps: Dict[int, Set[int]] = defaultdict(set)         # epsilon moves
        self.delta: Dict[int, Dict[str, int]] = defaultdict(dict)  # state,label->state (Thompson: at most one)

    def _state(self) -> int:
        s = self.n
        self.n += 1
        return s

    def symbol(self, label: str) -> _Frag:
        a, b = self._state(), self._state()
        self.delta[a][label] = b
        return _Frag(a, b)

    def epsilon(self) -> _Frag:
        a, b = self._state(), self._state()
        self.eps[a].add(b)
        return _Frag(a, b)

    def concat(self, f1: _Frag, f2: _Frag) -> _Frag:
        self.eps[f1.accept].add(f2.start)
        return _Frag(f1.start, f2.accept)

    def sequence(self, frags: List[_Frag]) -> _Frag:
        frags = list(frags)
        if not frags:
            return self.epsilon()
        acc = frags[0]
        for f in frags[1:]:
            acc = self.concat(acc, f)
        return acc

    def union(self, frags: List[_Frag]) -> _Frag:
        frags = list(frags)
        if len(frags) == 1:
            return frags[0]
        a, b = self._state(), self._state()
        for f in frags:
            self.eps[a].add(f.start)
            self.eps[f.accept].add(b)
        return _Frag(a, b)

    def star(self, f: _Frag) -> _Frag:
        a, b = self._state(), self._state()
        self.eps[a].add(f.start)
        self.eps[a].add(b)
        self.eps[f.accept].add(f.start)
        self.eps[f.accept].add(b)
        return _Frag(a, b)

    def optional(self, f: _Frag) -> _Frag:
        return self.union([self.epsilon(), f])


# ---------------------------------------------------------------------------
# Regular-expression blocks
# ---------------------------------------------------------------------------

def _order_key(order: Dict[str, int]):
    return lambda op: (order.get(op, 10 ** 9), op)


def _delta_no_unload(inst, v: str, WT_l: List[str]) -> List[str]:
    """Transfers in W_T(l) that draw from the storage tank unload v fills.

    In code terms: op_inlet[transfer] (its source storage) == op_outlet[v]
    (the storage v deposits into).  This is delta_NO(v) restricted to W_T(l).
    """
    storage = inst.op_outlet[v]
    return [w for w in WT_l if inst.op_inlet[w] == storage]


def _build_block(nfa: _NFA, inst, dp: Tuple[str, ...], order: Dict[str, int]) -> _Frag:
    """Construct a fresh NFA fragment for the block L_l of DP ``dp``."""
    key = _order_key(order)
    WT_l = compatible_transfers(
        inst.W_T, dp, outlet_of=inst.op_inlet, inlet_of=inst.op_outlet
    )
    WT_l_sorted = sorted(WT_l, key=key)

    factors: List[_Frag] = []

    # 1. the d charges of the DP, concatenated in DP (per-CDU) order
    for charge in dp:
        factors.append(nfa.symbol(charge))

    # 2. each compatible transfer, optional: (eps + w)
    for w in WT_l_sorted:
        factors.append(nfa.optional(nfa.symbol(w)))

    # 3. per unload v and transfer w in delta_NO(v): (eps + v + v.w)
    for v in sorted(inst.W_U, key=key):
        for w in sorted(_delta_no_unload(inst, v, WT_l), key=key):
            factors.append(nfa.union([
                nfa.epsilon(),
                nfa.symbol(v),
                nfa.sequence([nfa.symbol(v), nfa.symbol(w)]),
            ]))

    return nfa.sequence(factors)


def _neighbour_pairs(dps: List[Tuple[str, ...]]) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """Unordered DP pairs differing in exactly one CDU assignment."""
    pairs = []
    for i in range(len(dps)):
        for j in range(i + 1, len(dps)):
            a, b = set(dps[i]), set(dps[j])
            if len(a - b) == 1 and len(b - a) == 1:
                pairs.append((dps[i], dps[j]))
    return pairs


def _build_rl(nfa: _NFA, inst, dps: List[Tuple[str, ...]], order: Dict[str, int]) -> _Frag:
    """
    Build the RL fragment.

    The printed construction sums over DP *pairs*: RL = sum_{l,l'} (eps+L_l)(L_l
    L_l')*(eps+L_l').  That only admits schedules visiting at most two
    distillation programs, so it is infeasible whenever a schedule must visit
    three or more DPs (e.g. to refill a charging tank whose demand exceeds its
    initial level).  We use the natural generalisation that matches Algorithm 4:
    a walk over the whole DP-neighbour graph, executing each visited DP's block.

    Construction: one block L_l per DP; an epsilon move from a global start into
    every DP block (start in any DP); an epsilon move from each block's exit to
    the entry of every *neighbour* DP (switch DP by executing the new charge);
    and a global accept reachable (via epsilon) from the start and from every
    block exit (a schedule may end after any completed block).  Blocks are
    revisited through the neighbour loops, so re-execution rounds are captured
    with a single block copy per DP.  For two DPs this reduces to the printed
    pairwise expression.
    """
    start = nfa._state()
    accept = nfa._state()
    nfa.eps[start].add(accept)                       # empty word accepted

    entry: Dict[Tuple[str, ...], int] = {}
    exit_: Dict[Tuple[str, ...], int] = {}
    for l in dps:
        block = _build_block(nfa, inst, l, order)
        entry[l] = block.start
        exit_[l] = block.accept
        nfa.eps[start].add(block.start)              # start in any DP
        nfa.eps[block.accept].add(accept)            # accept after this block

    adj: Dict[Tuple[str, ...], List[Tuple[str, ...]]] = defaultdict(list)
    for a, b in _neighbour_pairs(dps):
        adj[a].append(b)
        adj[b].append(a)
    for l in dps:                                     # switch to a neighbour DP
        for lp in adj[l]:
            nfa.eps[exit_[l]].add(entry[lp])

    return _Frag(start, accept)


# ---------------------------------------------------------------------------
# NFA -> DFA (subset construction) -> minimal DFA (Moore)
# ---------------------------------------------------------------------------

def _eps_closure(nfa: _NFA, states: Iterable[int]) -> FrozenSet[int]:
    stack = list(states)
    seen = set(stack)
    while stack:
        s = stack.pop()
        for t in nfa.eps.get(s, ()):
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return frozenset(seen)


def _determinize(nfa: _NFA, start: int, accept: int):
    """Subset construction.  Returns (n_states, trans, start_id, accept_ids)."""
    start_set = _eps_closure(nfa, [start])
    index: Dict[FrozenSet[int], int] = {start_set: 0}
    order = [start_set]
    trans: Dict[Tuple[int, str], int] = {}
    queue = deque([start_set])
    while queue:
        S = queue.popleft()
        i = index[S]
        moves: Dict[str, Set[int]] = defaultdict(set)
        for s in S:
            for label, t in nfa.delta.get(s, {}).items():
                moves[label].add(t)
        for label, targets in moves.items():
            cl = _eps_closure(nfa, targets)
            j = index.get(cl)
            if j is None:
                j = len(order)
                index[cl] = j
                order.append(cl)
                queue.append(cl)
            trans[(i, label)] = j
    accept_ids = {i for subset, i in index.items() if accept in subset}
    return len(order), trans, 0, accept_ids


def _minimize(n_states: int, trans: Dict[Tuple[int, str], int], start: int,
              accept: Set[int]):
    """Moore partition-refinement minimisation of a (partial) DFA."""
    alphabet = sorted({label for (_, label) in trans})

    part = [0 if s in accept else 1 for s in range(n_states)]
    n_groups = len(set(part))
    while True:
        sigs: Dict[Tuple, int] = {}
        new_part = [0] * n_states
        for s in range(n_states):
            sig = (part[s],) + tuple(
                part[trans[(s, a)]] if (s, a) in trans else -1 for a in alphabet
            )
            gid = sigs.setdefault(sig, len(sigs))
            new_part[s] = gid
        part = new_part
        if len(sigs) == n_groups:
            break
        n_groups = len(sigs)

    new_start = part[start]
    new_accept = {part[s] for s in accept}
    new_trans: Dict[Tuple[int, str], int] = {}
    for (s, a), t in trans.items():
        new_trans[(part[s], a)] = part[t]
    return n_groups, new_trans, new_start, new_accept


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_sos_re_dfa(inst: CrudeOilFullInstance, allow_idle: bool = True) -> DFA:
    """
    Build the minimal SOS+RE DFA for a crude-oil instance.

    Returns a :class:`DFA` whose ``final_states`` are the regular-language
    accept states (plus the idle sink when ``allow_idle``).  Arc labels are
    operation names in ``inst.W`` (idle arcs use IDLE_LABEL).
    """
    cdu_inlet_ops = {cdu: [w for w in inst.W_D if inst.op_outlet[w] == cdu]
                     for cdu in inst.R_D}
    outlet_of_charging = {w: inst.op_inlet[w] for w in inst.W_D}
    dps = enumerate_distillation_programs(inst.R_D, cdu_inlet_ops, outlet_of_charging)
    if not dps:
        raise ValueError("SOS+RE requires at least one distillation program")

    order = {op: i for i, op in enumerate(inst.W)}

    nfa = _NFA()
    rl = _build_rl(nfa, inst, dps, order)
    n_states, trans, start, accept = _determinize(nfa, rl.start, rl.accept)
    n_states, trans, start, accept = _minimize(n_states, trans, start, accept)

    # Assemble arcs (one per (state, label) -- the automaton is deterministic).
    arcs: List[Tuple[int, int, str]] = [
        (s, t, label) for (s, label), t in sorted(trans.items())
    ]
    states = list(range(n_states))
    final = set(accept)
    idle_state = None

    if allow_idle:
        idle_state = n_states
        states.append(idle_state)
        for s in sorted(final):                 # idle only from accept states
            arcs.append((s, idle_state, IDLE_LABEL))
        arcs.append((idle_state, idle_state, IDLE_LABEL))
        final.add(idle_state)

    # state_of is informational only (the model does not use it for SOS+RE).
    state_of: Dict[str, int] = {}
    for s, t, label in arcs:
        if label != IDLE_LABEL and label not in state_of:
            state_of[label] = t

    return DFA(
        states=states,
        initial=start,
        arcs=arcs,
        state_of=state_of,
        operation_order=list(inst.W),
        idle_state=idle_state,
        final_states=sorted(final),
    )
