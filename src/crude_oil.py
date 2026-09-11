"""
Crude-oil scheduling instantiation of the priority-slot formulation.

Provides:
  - the CrudeOilInstance dataclass;
  - enumeration of Distillation Programs (DPs);
  - precedence and non-overlapping graph construction;
  - sub-DFA construction per DP and full DFA assembly.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .dfa_builder import build_dfa, DFA, topological_sort


# ---------------------------------------------------------------------------
# Instance dataclass
# ---------------------------------------------------------------------------

@dataclass
class CrudeOilInstance:
    """
    Data for a crude-oil scheduling instance.

    Network structure
    -----------------
    Vessels (R_V) -> [unload W_U] -> Storage tanks (R_S)
                  -> [transfer W_T] -> Charging tanks (R_C)
                  -> [charge W_D] -> CDUs (R_D)

    For every operation w:
      inlet[w]  = the resource w draws from (its input resource)
      outlet[w] = the resource w deposits into (its output resource)

    inlet_ops[r]  = list of operations w for which outlet[w] == r
                    (operations that *feed into* resource r)
    outlet_ops[r] = list of operations w for which inlet[w] == r
                    (operations that *draw from* resource r)

    Logistic constraints encoded in the graphs
    ------------------------------------------
    (i)   Only one berth: any two unloads are non-overlapping.
    (ii)  Arrival order: unloads ordered by vessel arrival time.
    (iii) No simultaneous inlet/outlet on a tank.
    (iv)  A charging tank charges at most one CDU at a time.
    (v)   A CDU is charged by at most one tank at a time.
    (vi)  CDU continuity: handled via Distillation Programs.
    """
    # Resources
    vessels: List[str]
    storage_tanks: List[str]
    charging_tanks: List[str]
    cdus: List[str]

    # Operations
    W_U: List[str]   # unloading:  vessel -> storage tank
    W_T: List[str]   # transfers:  storage tank -> charging tank
    W_D: List[str]   # charging:   charging tank -> CDU

    # Timing and connectivity
    arrival_times: Dict[str, float]   # w in W_U -> vessel arrival time
    inlet: Dict[str, str]             # operation -> source resource
    outlet: Dict[str, str]            # operation -> destination resource

    # Derived lookup tables (built from inlet/outlet)
    inlet_ops: Dict[str, List[str]]   # resource -> ops depositing into it
    outlet_ops: Dict[str, List[str]]  # resource -> ops drawing from it

    # Horizon and slot count
    horizon: float
    n_slots: int

    @property
    def W(self) -> List[str]:
        return self.W_U + self.W_T + self.W_D

    @classmethod
    def build_lookup_tables(cls, inlet: dict, outlet: dict,
                            resources: list) -> Tuple[dict, dict]:
        """
        Helper: build inlet_ops and outlet_ops from inlet/outlet maps.
        Call this before constructing the instance.
        """
        inlet_ops = defaultdict(list)
        outlet_ops = defaultdict(list)
        for w, r in outlet.items():
            inlet_ops[r].append(w)     # w feeds into r
        for w, r in inlet.items():
            outlet_ops[r].append(w)    # w draws from r
        return dict(inlet_ops), dict(outlet_ops)


# ---------------------------------------------------------------------------
# Algorithm 3 — Distillation Program enumeration
# ---------------------------------------------------------------------------

def enumerate_distillation_programs(
    cdus: List[str],
    cdu_inlet_ops: Dict[str, List[str]],
    outlet_of: Dict[str, str],
) -> List[Tuple[str, ...]]:
    """
    Algorithm 3: enumerate all valid Distillation Programs (DPs).

    A DP is a d-tuple (w_{i1}, ..., w_{id}) with one charging operation per
    CDU such that all operations use *distinct* charging tanks (their outlet
    resources are pairwise different).  This encodes constraints (iv) and (v).

    Parameters
    ----------
    cdus : list[str]
        CDU IDs in a fixed order.
    cdu_inlet_ops : dict[str, list[str]]
        Maps each CDU to the charging operations that feed it
        (i.e., outlet_ops[cdu] in the instance).
    outlet_of : dict[str, str]
        Maps each charging operation to its charging tank (outlet resource).

    Returns
    -------
    list[tuple[str, ...]]
        Each tuple is a valid DP; index j corresponds to cdus[j].
    """
    d = len(cdus)
    result = []

    def recurse(k: int, partial: list, used_tanks: set):
        if k == d:
            result.append(tuple(partial))
            return
        cdu = cdus[k]
        for w in cdu_inlet_ops.get(cdu, []):
            tank = outlet_of[w]
            if tank not in used_tanks:
                partial.append(w)
                used_tanks.add(tank)
                recurse(k + 1, partial, used_tanks)
                partial.pop()
                used_tanks.discard(tank)

    recurse(0, [], set())
    return result


# ---------------------------------------------------------------------------
# Compatible transfer operations for a DP
# ---------------------------------------------------------------------------

def compatible_transfers(
    W_T: List[str],
    dp_ops: Tuple[str, ...],
    outlet_of: Dict[str, str],
    inlet_of: Dict[str, str],
) -> List[str]:
    """
    W_T(l): transfer operations compatible with Distillation Program l.

    A transfer w is incompatible if its inlet tank is committed to CDU
    charging under l, i.e. if inlet_of[w] == outlet_of[dp_op] for some dp_op.

    Parameters
    ----------
    W_T : list[str]
    dp_ops : tuple[str, ...]
        Current DP.
    outlet_of : dict[str, str]
        outlet_of[w] = charging tank for w in W_D.
    inlet_of : dict[str, str]
        inlet_of[w] = source tank for w in W_T.
    """
    committed = {outlet_of[op] for op in dp_ops if op in outlet_of}
    return [w for w in W_T if inlet_of.get(w) not in committed]


# ---------------------------------------------------------------------------
# Graph construction for one DP
# ---------------------------------------------------------------------------

def build_precedence_edges(
    W_U: List[str],
    arrival_times: Dict[str, float],
) -> List[Tuple[str, str]]:
    """
    Constraint (ii): unloads ordered by vessel arrival time.
    Returns directed arcs (v, w) for each pair with arrival_times[v] < arrival_times[w].
    """
    edges = []
    for v in W_U:
        for w in W_U:
            if v != w and arrival_times[v] < arrival_times[w]:
                edges.append((v, w))
    return edges


def build_no_edges_for_dp(
    instance: CrudeOilInstance,
    dp_ops: Tuple[str, ...],
    W_T_l: List[str],
) -> List[FrozenSet]:
    """
    Non-overlapping edges for operations in W_U ∪ W_T(l).

    Constraint (i)  : all pairs of unloads.
    Constraint (iii): inlet op and outlet op on the same tank cannot overlap.
                      For tank r: ops depositing into r vs ops drawing from r.
    """
    active_ops = set(instance.W_U) | set(W_T_l)
    edges = set()

    def add(v, w):
        if v != w and v in active_ops and w in active_ops:
            edges.add(frozenset([v, w]))

    # (i) All unload pairs
    for i, v in enumerate(instance.W_U):
        for w in instance.W_U[i + 1:]:
            add(v, w)

    # (iii) Inlet/outlet conflicts on every tank
    all_tanks = instance.storage_tanks + instance.charging_tanks
    for tank in all_tanks:
        # ops that deposit INTO tank (outlet[w] == tank)
        depositers = [w for w in active_ops if instance.outlet.get(w) == tank]
        # ops that draw FROM tank (inlet[w] == tank)
        drawers = [w for w in active_ops if instance.inlet.get(w) == tank]
        for v in depositers:
            for w in drawers:
                add(v, w)

    return list(edges)


# ---------------------------------------------------------------------------
# Sub-DFA per DP and full DFA assembly
# ---------------------------------------------------------------------------

@dataclass
class CrudeOilDFAResult:
    """
    Full DFA result for a crude-oil instance.

    The combined state space is (dp_index, sub_state) where sub_state is a
    state in the sub-DFA for that DP.  Charging operations link sub-DFAs.

    Attributes
    ----------
    dps : list of DPs (each a tuple of charging ops)
    sub_dfas : dict[dp, DFA]  sub-DFA for W_U ∪ W_T(l)
    W_T_per_dp : dict[dp, list[str]]  compatible transfers per DP
    dp_neighbors : dict[dp, list[(dp', new_charging_op)]]
        Neighboring DPs (differ in exactly one CDU assignment) and the
        charging operation that moves from dp to dp'.
    prec_edges : list of precedence arcs used in sub-DFAs
    """
    dps: list
    sub_dfas: dict
    W_T_per_dp: dict
    dp_neighbors: dict
    prec_edges: list

    def report(self):
        lines = [f"Distillation Programs: {len(self.dps)}"]
        total_arcs = sum(d.n_arcs for d in self.sub_dfas.values())
        total_states = sum(d.n_states for d in self.sub_dfas.values())
        lines.append(f"Sub-DFA totals: {total_states} states, {total_arcs} arcs")
        for i, dp in enumerate(self.dps):
            dfa = self.sub_dfas[dp]
            nbrs = self.dp_neighbors.get(dp, [])
            lines.append(f"  DP {i}: {dp}  |  {dfa.n_arcs} arcs  |  {len(nbrs)} neighbors")
        return "\n".join(lines)


def build_crude_oil_dfa(instance: CrudeOilInstance) -> CrudeOilDFAResult:
    """
    Construct the full DFA for crude-oil scheduling.

    Steps:
    1. Enumerate all DPs (Algorithm 3).
    2. For each DP l, build a sub-DFA for W_U ∪ W_T(l) (Algorithm 2).
    3. Identify neighboring DPs (differ in exactly one CDU assignment).

    The sub-DFAs are linked by charging operations at the caller's level
    (in the AMPL model each DP sub-DFA is a separate layer connected via
    the distillation program transitions).
    """
    # CDU inlet ops: charging ops that feed each CDU (outlet[w] == cdu)
    cdu_inlet_ops = {
        cdu: instance.inlet_ops.get(cdu, [])
        for cdu in instance.cdus
    }

    # outlet_of for charging operations (their source charging tank)
    outlet_of_charging = {w: instance.inlet[w] for w in instance.W_D if w in instance.inlet}

    # 1. Enumerate DPs
    dps = enumerate_distillation_programs(
        instance.cdus, cdu_inlet_ops, outlet_of_charging
    )

    # Precedence edges (same for all sub-DFAs: arrival order on W_U)
    prec_edges = build_precedence_edges(instance.W_U, instance.arrival_times)

    # 2. Build sub-DFA for each DP
    sub_dfas = {}
    W_T_per_dp = {}
    for dp in dps:
        W_T_l = compatible_transfers(
            instance.W_T, dp,
            outlet_of=instance.inlet,
            inlet_of=instance.outlet,
        )
        W_T_per_dp[dp] = W_T_l
        W_active = instance.W_U + W_T_l
        no_edges = build_no_edges_for_dp(instance, dp, W_T_l)
        # Crude-oil operations are uncapped (eta_w = n), so every state of a
        # sub-DFA carries a Rule-4 self-loop.
        sub_dfas[dp] = build_dfa(W_active, prec_edges, no_edges,
                                 repeatable=True)

    # 3. Neighbor relation: DPs that differ in exactly one CDU assignment
    dp_neighbors: Dict[tuple, list] = defaultdict(list)
    for i, dp1 in enumerate(dps):
        for j in range(i + 1, len(dps)):
            dp2 = dps[j]
            set1, set2 = set(dp1), set(dp2)
            diff1 = set1 - set2  # op leaving dp1
            diff2 = set2 - set1  # op entering dp2
            if len(diff1) == 1 and len(diff2) == 1:
                new_for_2 = next(iter(diff2))
                new_for_1 = next(iter(diff1))
                dp_neighbors[dp1].append((dp2, new_for_2))
                dp_neighbors[dp2].append((dp1, new_for_1))

    return CrudeOilDFAResult(
        dps=dps,
        sub_dfas=sub_dfas,
        W_T_per_dp=W_T_per_dp,
        dp_neighbors=dict(dp_neighbors),
        prec_edges=prec_edges,
    )
