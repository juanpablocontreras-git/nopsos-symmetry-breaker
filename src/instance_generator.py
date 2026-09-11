"""
Synthetic crude-oil instance generator.

Produces random refinery instances in the JSON schema used by the crude-oil
instances (see crude_oil_ampl.load_instance_json).  Topologies and parameter
values are randomised within reasonable ranges, and the data is balanced so that
a trivial schedule -- distil each charging tank's initial seed crude once -- is
always feasible.  That gives a benchmark suite whose difficulty comes from the
logistic/sequencing structure, not from accidental infeasibility.

Design choices that keep instances feasible by construction
-----------------------------------------------------------
* Topology follows the reference refineries: one unload per vessel
  (V_i -> ST_i, 1:1), a random connected bipartite storage->charging transfer
  graph, and a random charging->CDU charge graph in which every charging tank
  has at least one outgoing charge (so its demand can be met) and every CDU at
  least one inflow.
* Each storage and charging tank is seeded with a distinct crude type, so
  |C| = |R_S| + |R_C| (as in the reference instances).
* Demand D[tank] = Lt0[tank]: distilling the seed once satisfies it, so no
  refill transfer is ever strictly required and tanks never need to exceed
  capacity.  ND = |R_C| (one charge per charging tank).
* Charge property windows bracket the seed crude's property values, so the
  seed-only solution meets the quality specs.  Unload/transfer windows are the
  inert [0, 1].

The construction targets the MILP step (bilinear composition dropped).  Each
instance should still be validated by an actual solve; see
``run_experiments``-style checking or the ``__main__`` block below.
"""

import json
import random
import string
from pathlib import Path
from typing import Dict, List, Tuple

# Volume/flow constants shared by all synthetic instances.
PARCEL = 1000.0          # vessel parcel size (unload total volume, fixed)
TANK_CAP = 1000.0        # tank capacity upper bound
FR_UB = 500.0            # flowrate upper bound (all operations)
CHARGE_FR_LB = 50.0      # minimum charging flowrate (CDUs run at >= this rate)
PROP_MARGIN = 0.01       # half-width of the charge property window around seed


def _crude_name(idx: int) -> str:
    """A, B, ..., Z, then C27, C28, ... (only needed beyond 26 crude types)."""
    if idx < 26:
        return string.ascii_uppercase[idx]
    return f"C{idx + 1}"


# ---------------------------------------------------------------------------
# Size schedule
# ---------------------------------------------------------------------------

def suite_sizes(n_instances: int) -> List[Dict[str, int]]:
    """
    Resource-count schedule for a suite of ``n_instances`` increasing-size
    instances.  Sizes ramp smoothly from small (around the smallest reference
    refinery) up to "large but not too much" -- somewhat beyond the larger
    reference refineries.  CDUs are capped at 3 and charging tanks at
    5 to keep the distillation-program structure (and the MILP) tractable.
    """
    sizes = []
    for i in range(n_instances):
        s = i / max(1, n_instances - 1)          # 0 .. 1
        n_cdu = 1 + round(2 * s)                  # 1 .. 3
        n_charge = max(n_cdu + 1, 2 + round(4 * s))  # 2 .. 6, always > n_cdu
        n_storage = 2 + round(4 * s)              # 2 .. 6
        n_props = 1 + round(s)                    # 1 .. 2
        horizon = 8.0 + 8.0 * s                   # 8 .. 16
        sizes.append({
            "n_vessel": n_storage,                # 1:1 vessel<->storage
            "n_storage": n_storage,
            "n_charge": n_charge,
            "n_cdu": n_cdu,
            "n_props": n_props,
            "horizon": round(horizon, 1),
        })
    return sizes


# ---------------------------------------------------------------------------
# Single-instance generation
# ---------------------------------------------------------------------------

def generate_instance(
    *,
    n_vessel: int,
    n_storage: int,
    n_charge: int,
    n_cdu: int,
    n_props: int,
    horizon: float,
    rng: random.Random,
    comment: str = "",
    favorable: bool = False,
    demand_slack: float = 0.0,
) -> dict:
    """
    Generate one synthetic instance as a JSON-ready dict.

    Parameters mirror :func:`suite_sizes` entries; ``rng`` makes the topology
    and values reproducible.  Returns a dict in the crude-oil instance schema.

    favorable
        If True, all vessels share a common arrival time (0).  The only
        precedence in crude-oil scheduling comes from the vessel arrival order,
        so equal arrivals leave the unloads freely permutable, and the DFA-SBS
        removes the large family of symmetric slot-sequences this creates -- it
        multiplies the automaton's symmetry reduction several-fold (see
        :func:`generate_favorable_crude_suite`).  A seed-only schedule stays
        feasible, since demand equals the seed level regardless of arrivals.
    """
    if n_charge < n_cdu:
        raise ValueError("need at least one charging tank per CDU")

    vessels = [f"V{i+1}" for i in range(n_vessel)]
    storages = [f"ST{i+1}" for i in range(n_storage)]
    chargings = [f"CT{i+1}" for i in range(n_charge)]
    cdus = [f"CDU{i+1}" for i in range(n_cdu)]
    props = [f"prop{i+1}" for i in range(n_props)]

    # --- Crude types: one distinct seed per tank (storage then charging) ---
    n_crude = n_storage + n_charge
    crudes = [_crude_name(i) for i in range(n_crude)]
    tank_seed = {}                                   # tank -> seed crude
    for i, st in enumerate(storages):
        tank_seed[st] = crudes[i]
    for j, ct in enumerate(chargings):
        tank_seed[ct] = crudes[n_storage + j]

    # --- Transfer topology: random connected bipartite storage -> charging ---
    transfer_edges: List[Tuple[str, str]] = []
    seen = set()

    def add_transfer(st, ct):
        if (st, ct) not in seen:
            seen.add((st, ct))
            transfer_edges.append((st, ct))

    for ct in chargings:                              # each charging fed by 1-2 storages
        for st in rng.sample(storages, k=min(len(storages), rng.randint(1, 2))):
            add_transfer(st, ct)
    for st in storages:                               # ensure every storage feeds something
        if not any(e[0] == st for e in transfer_edges):
            add_transfer(st, rng.choice(chargings))

    # --- Charge topology: random charging -> CDU, every tank drains, every CDU fed ---
    charge_edges: List[Tuple[str, str]] = []
    seenc = set()

    def add_charge(ct, cdu):
        if (ct, cdu) not in seenc:
            seenc.add((ct, cdu))
            charge_edges.append((ct, cdu))

    for cdu in cdus:                                  # each CDU fed by 1-2 charging tanks
        for ct in rng.sample(chargings, k=min(len(chargings), rng.randint(1, 2))):
            add_charge(ct, cdu)
    for ct in chargings:                              # ensure every charging tank can drain
        if not any(e[0] == ct for e in charge_edges):
            add_charge(ct, rng.choice(cdus))

    # --- Operation names: u* unloads, t* transfers, d* charges, numbered globally ---
    op_inlet: Dict[str, str] = {}
    op_outlet: Dict[str, str] = {}
    W_U, W_T, W_D = [], [], []
    counter = 1

    for v, st in zip(vessels, storages):              # unload V_i -> ST_i
        name = f"u{counter}"; counter += 1
        W_U.append(name); op_inlet[name] = v; op_outlet[name] = st

    for st, ct in sorted(transfer_edges):             # transfer ST -> CT
        name = f"t{counter}"; counter += 1
        W_T.append(name); op_inlet[name] = st; op_outlet[name] = ct

    for ct, cdu in sorted(charge_edges):              # charge CT -> CDU
        name = f"d{counter}"; counter += 1
        W_D.append(name); op_inlet[name] = ct; op_outlet[name] = cdu

    # --- Numeric parameters ---
    # Vessel arrival times.  Favorable: all equal (0) -> no unload precedence, so
    # the unloads are freely permutable and the DFA-SBS removes a large family of
    # symmetric sequences.  Otherwise: 0, then increasing over the first ~60% of H.
    if favorable:
        s_r = {v: 0.0 for v in vessels}
    else:
        arrivals = sorted(round(rng.uniform(0, 0.6 * horizon), 1) for _ in vessels)
        arrivals[0] = 0.0
        s_r = {v: a for v, a in zip(vessels, arrivals)}

    # Initial tank levels (multiples of 50, within capacity) and seed crude.
    Lt0 = {t: float(rng.randrange(150, int(TANK_CAP) - 100, 50)) for t in tank_seed}
    L0 = {(t, c): 0.0 for t in tank_seed for c in crudes}
    for t, c in tank_seed.items():
        L0[(t, c)] = Lt0[t]

    # Crude properties and gross margins.
    x_prop = {(c, k): round(rng.uniform(0.01, 0.4), 4) for c in crudes for k in props}
    G_crude = {c: round(rng.uniform(1.0, 10.0), 2) for c in crudes}

    # Flowrate / volume bounds.
    FR_lb = {w: 0.0 for w in W_U + W_T}
    FR_lb.update({w: CHARGE_FR_LB for w in W_D})
    FR_ub = {w: FR_UB for w in W_U + W_T + W_D}
    Vt_lb = {w: PARCEL for w in W_U}                  # vessel delivers a full parcel
    Vt_lb.update({w: 0.0 for w in W_T + W_D})
    Vt_ub = {w: PARCEL for w in W_U + W_T + W_D}

    # Tank capacities.
    Lt_lb = {t: 0.0 for t in tank_seed}
    Lt_ub = {t: TANK_CAP for t in tank_seed}

    # Demand around the initial seed level.  With demand_slack = 0 the window
    # collapses to the seed level, so distilling the seed once meets demand
    # exactly and the operational layer is nearly determined: most instances
    # then solve at the root and leave no tree for a symmetry breaker to act
    # on.  A positive slack widens the window, which keeps the seed-only
    # schedule feasible (it lies inside the window for any slack >= 0) while
    # letting the model choose how much to distil, so blending and vessel
    # deliveries become real decisions.
    lo, hi = 1.0 - demand_slack, 1.0 + demand_slack
    D_lb = {ct: round(lo * Lt0[ct], 4) for ct in chargings}
    D_ub = {ct: round(hi * Lt0[ct], 4) for ct in chargings}

    # Distillation count used only to size the schedule (below).  The ND bound
    # itself is left loose (ND_lb=0, ND_ub=n_slots) so the two formulations are
    # compared on the same physical problem: a fixed ND is not equivalent across
    # them (SOS+RE re-executes continuing charges, inflating the count), while
    # the per-tank demand D_lb already forces every charging tank to be drained.
    ND = len(chargings)

    # Property windows: charges bracket their tank's seed crude; others inert.
    x_lb: Dict[Tuple[str, str], float] = {}
    x_ub: Dict[Tuple[str, str], float] = {}
    for w in W_U + W_T:
        for k in props:
            x_lb[(w, k)] = 0.0
            x_ub[(w, k)] = 1.0
    for w in W_D:
        seed = tank_seed[op_inlet[w]]
        for k in props:
            pv = x_prop[(seed, k)]
            x_lb[(w, k)] = round(max(0.0, pv - PROP_MARGIN), 4)
            x_ub[(w, k)] = round(min(1.0, pv + PROP_MARGIN), 4)

    # Schedule length: schedulable maximum (idle sink absorbs any surplus).
    n_slots = len(W_U) + len(W_T) + ND

    def join_pair(d):
        return {f"{a}__{b}": v for (a, b), v in d.items()}

    return {
        "_comment": comment or (
            f"Synthetic instance: {n_vessel} vessels, {n_storage} storage, "
            f"{n_charge} charging, {n_cdu} CDU(s); {len(W_U)+len(W_T)+len(W_D)} "
            f"operations. Generated by src.instance_generator; demand window "
            f"is the seed level widened by {demand_slack:.0%}, which keeps a "
            f"seed-only schedule feasible."
        ),
        "R_V": vessels, "R_S": storages, "R_C": chargings, "R_D": cdus,
        "W_U": W_U, "W_T": W_T, "W_D": W_D,
        "op_inlet": op_inlet, "op_outlet": op_outlet,
        "C": crudes, "K": props,
        "H": float(horizon), "n_slots": n_slots,
        "s_r": s_r,
        "FR_lb": FR_lb, "FR_ub": FR_ub,
        "Vt_lb": Vt_lb, "Vt_ub": Vt_ub,
        "Lt_lb": Lt_lb, "Lt_ub": Lt_ub, "Lt0": Lt0,
        "D_lb": D_lb, "D_ub": D_ub,
        "ND_lb": 0, "ND_ub": n_slots,
        "x_lb": join_pair(x_lb), "x_ub": join_pair(x_ub),
        "x_prop": join_pair(x_prop),
        "G_crude": G_crude,
        "L0": join_pair(L0),
    }


# ---------------------------------------------------------------------------
# Suite generation
# ---------------------------------------------------------------------------

def generate_suite(
    n_instances: int = 30,
    out_dir: str = "instances",
    seed: int = 20240529,
    prefix: str = "synth",
    demand_slack: float = 0.0,
) -> List[str]:
    """
    Generate ``n_instances`` increasing-size synthetic instances and write them
    to ``out_dir/<prefix>_NN.json``.  Returns the list of written paths.

    ``demand_slack`` widens each charging tank's demand window around its seed
    level; see :func:`generate_instance`.  The default of zero reproduces the
    original suite exactly, so existing instances are unaffected.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sizes = suite_sizes(n_instances)
    width = len(str(n_instances))
    paths = []
    for i, size in enumerate(sizes, start=1):
        rng = random.Random(seed + i)                # reproducible per instance
        name = f"{prefix}_{i:0{width}d}"
        inst = generate_instance(rng=rng, comment="",
                                 demand_slack=demand_slack, **size)
        inst["_comment"] = (
            f"Synthetic instance {i}/{n_instances} (size rank {i}). " + inst["_comment"]
        )
        path = out / f"{name}.json"
        with open(path, "w") as fh:
            json.dump(inst, fh, indent=2)
        paths.append(str(path))
    return paths


def generate_favorable_crude_suite(
    n_instances: int = 10,
    out_dir: str = "instances",
    seed: int = 20240702,
    prefix: str = "crude_fav",
) -> List[str]:
    """
    Generate ``n_instances`` crude-oil refineries engineered so the DFA-SBS has an
    advantage: moderate size (the middle of the standard size ramp, where the
    symmetry reduction is large but the MILP still solves) and equal vessel
    arrivals (``favorable=True``), which removes the unload precedence and so
    frees the large family of symmetric slot-sequences the DFA collapses.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Skip the three smallest sizes; take a moderate-size window of the ramp.
    sizes = suite_sizes(n_instances + 3)[3:]
    width = len(str(n_instances))
    paths = []
    for i, size in enumerate(sizes, start=1):
        rng = random.Random(seed + i)
        name = f"{prefix}_{i:0{width}d}"
        inst = generate_instance(rng=rng, comment="", favorable=True, **size)
        inst["_comment"] = (
            f"Favorable crude-oil instance {i}/{n_instances}: equal vessel "
            f"arrivals (no unload precedence -> large symmetric family removed by "
            f"the DFA-SBS). " + inst["_comment"]
        )
        path = out / f"{name}.json"
        with open(path, "w") as fh:
            json.dump(inst, fh, indent=2)
        paths.append(str(path))
    return paths


# ===========================================================================
# Job-shop scheduling instances (second NOPSOS application)
# ===========================================================================
#
# A classic job-shop instance is a set of jobs, each a fixed routing through a
# set of machines, with one operation per (job, machine).  It maps onto the two
# NOPSOS relations with no operational layer at all:
#
#   * precedence      -- the routing chain inside each job (operation k must
#                        finish before operation k+1 of the same job starts);
#   * non-overlapping -- every pair of operations that share a machine (each
#                        machine is a unary resource, i.e. a clique in G_NO).
#
# Processing times are stored explicitly on each operation, so the instance is
# self-contained (unlike the crude-oil scheduling core, which hashes durations
# from the instance name).  The scheduling core models/scheduling.mod solves the
# makespan objective on exactly these two relations.

JOBSHOP_PROC_MIN, JOBSHOP_PROC_MAX = 1, 20   # processing-time range


def jobshop_suite_sizes(n_instances: int) -> List[Dict[str, int]]:
    """
    (n_jobs, n_machines) schedule for ``n_instances`` job-shop instances of
    increasing difficulty.  Job-shop hardness grows quickly with the number of
    operations |W| = n_jobs * n_machines, so the ramp stays modest: from a
    2x2 = 4-operation warm-up up to a 7x6 = 42-operation instance that is
    already challenging for a slot-based formulation.
    """
    # Hand-tuned so |W| increases monotonically and both dimensions grow.
    ladder = [
        (2, 2),   # 4
        (3, 2),   # 6
        (3, 3),   # 9
        (4, 3),   # 12
        (4, 4),   # 16
        (5, 4),   # 20
        (5, 5),   # 25
        (6, 5),   # 30
        (6, 6),   # 36
        (7, 6),   # 42
    ]
    if n_instances <= len(ladder):
        chosen = ladder[:n_instances]
    else:
        # Extend by growing jobs then machines, keeping the monotone ramp.
        chosen = list(ladder)
        j, m = ladder[-1]
        while len(chosen) < n_instances:
            j += 1
            chosen.append((j, m))
            if len(chosen) < n_instances:
                m += 1
                chosen.append((j, m))
    return [{"n_jobs": j, "n_machines": m} for j, m in chosen]


def generate_jobshop_instance(
    *,
    n_jobs: int,
    n_machines: int,
    rng: random.Random,
    comment: str = "",
    proc_choices=None,
) -> dict:
    """
    Generate one random job-shop instance as a JSON-ready dict.

    Each job visits every machine exactly once in a random order (the standard
    square job-shop of the OR-Library / Taillard benchmarks).  Operation names
    are ``J{job}P{position}`` -- purely alphanumeric so they are safe as AMPL set
    members and DFA arc labels.  The machine and integer processing time of each
    operation are recorded on the operation.

    proc_choices
        If given (a list of integers), processing times are drawn uniformly from
        it; otherwise they are drawn from ``[JOBSHOP_PROC_MIN, JOBSHOP_PROC_MAX]``.
        A short list (e.g. ``[2, 3]``) yields low time-diversity, which makes many
        schedules cost-equivalent and so multiplies the symmetric optima the
        DFA-SBS removes -- see :func:`generate_favorable_jobshop_suite`.
    """
    machines = [f"M{m+1}" for m in range(n_machines)]

    def proc():
        return (rng.choice(proc_choices) if proc_choices
                else rng.randint(JOBSHOP_PROC_MIN, JOBSHOP_PROC_MAX))

    jobs = []
    for _ in range(n_jobs):
        routing = rng.sample(machines, k=n_machines)      # random machine order
        ops = [[mach, proc()] for mach in routing]
        jobs.append(ops)

    n_ops = n_jobs * n_machines
    return {
        "_comment": comment or (
            f"Synthetic job-shop instance: {n_jobs} jobs x {n_machines} "
            f"machines = {n_ops} operations. Each job visits every machine once "
            f"in a random order. Generated by src.instance_generator."
        ),
        "problem": "jobshop",
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "machines": machines,
        # jobs[j] = list of [machine, proc_time] in routing order for job j+1
        "jobs": jobs,
    }


def generate_jobshop_suite(
    n_instances: int = 10,
    out_dir: str = "instances",
    seed: int = 20240530,
    prefix: str = "jobshop",
) -> List[str]:
    """
    Generate ``n_instances`` increasing-difficulty job-shop instances and write
    them to ``out_dir/<prefix>_NN.json``.  Returns the list of written paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sizes = jobshop_suite_sizes(n_instances)
    width = len(str(n_instances))
    paths = []
    for i, size in enumerate(sizes, start=1):
        rng = random.Random(seed + i)                    # reproducible per instance
        name = f"{prefix}_{i:0{width}d}"
        inst = generate_jobshop_instance(rng=rng, comment="", **size)
        inst["_comment"] = (
            f"Job-shop instance {i}/{n_instances} (size rank {i}). " + inst["_comment"]
        )
        path = out / f"{name}.json"
        with open(path, "w") as fh:
            json.dump(inst, fh, indent=2)
        paths.append(str(path))
    return paths


# ===========================================================================
# Open-shop scheduling instances (third NOPSOS application)
# ===========================================================================
#
# Open-shop is job-shop without the routing order: each job still has one
# operation per machine, but those operations may be processed in any order.
# The mapping onto the two NOPSOS relations therefore has an EMPTY precedence
# graph and a non-overlapping graph made of two clique families -- one per
# machine (a machine runs one operation at a time) and one per job (a job is
# worked on by one machine at a time).  Dropping precedence maximises the
# priority-slot symmetry, so open-shop stresses the automatic symmetry breaker
# harder than job-shop does.  The JSON schema matches the job-shop one (the
# order of a job's operations simply carries no meaning here).


def favorable_jobshop_sizes(n_instances: int = 10):
    """
    (n_jobs, n_machines) ladder for the ``jobshop_fav`` suite: moderate sizes in
    the regime where the DFA-SBS has a clear advantage -- large enough for the
    priority-slot symmetry to bite, small enough that the model still solves and
    the flow-layer overhead does not dominate.  Balanced-to-wide shapes keep the
    machine cliques small (more independent operation pairs).
    """
    ladder = [(3, 3), (3, 4), (4, 3), (4, 4), (3, 5),
              (5, 3), (4, 5), (5, 4), (5, 5), (4, 6)]
    return [{"n_jobs": j, "n_machines": m} for j, m in ladder[:n_instances]]


def generate_favorable_jobshop_suite(
    n_instances: int = 10,
    out_dir: str = "instances",
    seed: int = 20240701,
    prefix: str = "jobshop_fav",
    proc_choices=(2, 3),
) -> List[str]:
    """
    Generate ``n_instances`` job-shop instances engineered so the DFA-SBS has an
    advantage: moderate size (:func:`favorable_jobshop_sizes`) and low
    processing-time diversity (``proc_choices``, default ``{2, 3}``).  Uniform-ish
    times make many schedules cost-equivalent, so the bare formulation must
    enumerate a large family of symmetric optima that the DFA collapses to one.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sizes = favorable_jobshop_sizes(n_instances)
    width = len(str(n_instances))
    paths = []
    for i, size in enumerate(sizes, start=1):
        rng = random.Random(seed + i)
        name = f"{prefix}_{i:0{width}d}"
        inst = generate_jobshop_instance(rng=rng, comment="",
                                         proc_choices=list(proc_choices), **size)
        inst["_comment"] = (
            f"Favorable job-shop instance {i}/{n_instances}: {size['n_jobs']} jobs "
            f"x {size['n_machines']} machines, processing times in "
            f"{sorted(set(proc_choices))} (low diversity -> many symmetric optima). "
            f"Engineered so the DFA-SBS has a clear advantage over the bare "
            f"formulation. Generated by src.instance_generator."
        )
        path = out / f"{name}.json"
        with open(path, "w") as fh:
            json.dump(inst, fh, indent=2)
        paths.append(str(path))
    return paths


def generate_openshop_instance(
    *,
    n_jobs: int,
    n_machines: int,
    rng: random.Random,
    comment: str = "",
) -> dict:
    """
    Generate one random open-shop instance as a JSON-ready dict.

    Each job has an integer processing time on every machine; there is no
    routing order.  Operation names are ``J{job}P{position}`` with position
    indexing the machine list, purely alphanumeric for AMPL/DFA use.
    """
    machines = [f"M{m+1}" for m in range(n_machines)]
    jobs = [[[mach, rng.randint(JOBSHOP_PROC_MIN, JOBSHOP_PROC_MAX)]
             for mach in machines]                      # no routing order
            for _ in range(n_jobs)]

    n_ops = n_jobs * n_machines
    return {
        "_comment": comment or (
            f"Synthetic open-shop instance: {n_jobs} jobs x {n_machines} "
            f"machines = {n_ops} operations. No routing order (open shop). "
            f"Generated by src.instance_generator."
        ),
        "problem": "openshop",
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "machines": machines,
        # jobs[j] = list of [machine, proc_time]; order carries no precedence
        "jobs": jobs,
    }


def generate_openshop_suite(
    n_instances: int = 10,
    out_dir: str = "instances",
    seed: int = 20240531,
    prefix: str = "openshop",
) -> List[str]:
    """
    Generate ``n_instances`` increasing-difficulty open-shop instances (same size
    ladder as the job-shop suite) and write them to ``out_dir/<prefix>_NN.json``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sizes = jobshop_suite_sizes(n_instances)
    width = len(str(n_instances))
    paths = []
    for i, size in enumerate(sizes, start=1):
        rng = random.Random(seed + i)                    # reproducible per instance
        name = f"{prefix}_{i:0{width}d}"
        inst = generate_openshop_instance(rng=rng, comment="", **size)
        inst["_comment"] = (
            f"Open-shop instance {i}/{n_instances} (size rank {i}). " + inst["_comment"]
        )
        path = out / f"{name}.json"
        with open(path, "w") as fh:
            json.dump(inst, fh, indent=2)
        paths.append(str(path))
    return paths


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    if argv and argv[0] == "crude_fav":
        n = int(argv[1]) if len(argv) > 1 else 10
        written = generate_favorable_crude_suite(n)
        print(f"Wrote {len(written)} favorable crude-oil instances (equal arrivals):")
        for p in written:
            print(f"  {Path(p).stem}")
    elif argv and argv[0] == "jobshop_fav":
        n = int(argv[1]) if len(argv) > 1 else 10
        written = generate_favorable_jobshop_suite(n)
        print(f"Wrote {len(written)} favorable job-shop instances (low time-diversity):")
        for size, p in zip(favorable_jobshop_sizes(n), written):
            stem = Path(p).stem
            n_ops = size["n_jobs"] * size["n_machines"]
            print(f"  {stem}: jobs={size['n_jobs']} machines={size['n_machines']} "
                  f"ops={n_ops}")
    elif argv and argv[0] == "openshop":
        n = int(argv[1]) if len(argv) > 1 else 10
        written = generate_openshop_suite(n)
        print(f"Wrote {len(written)} open-shop instances:")
        for size, p in zip(jobshop_suite_sizes(n), written):
            stem = Path(p).stem
            n_ops = size["n_jobs"] * size["n_machines"]
            print(f"  {stem}: jobs={size['n_jobs']} machines={size['n_machines']} "
                  f"ops={n_ops}")
    elif argv and argv[0] == "jobshop":
        n = int(argv[1]) if len(argv) > 1 else 10
        written = generate_jobshop_suite(n)
        print(f"Wrote {len(written)} job-shop instances:")
        for size, p in zip(jobshop_suite_sizes(n), written):
            stem = Path(p).stem
            n_ops = size["n_jobs"] * size["n_machines"]
            print(f"  {stem}: jobs={size['n_jobs']} machines={size['n_machines']} "
                  f"ops={n_ops}")
    else:
        n = int(argv[0]) if argv else 30
        written = generate_suite(n)
        print(f"Wrote {len(written)} instances:")
        for size, p in zip(suite_sizes(n), written):
            stem = Path(p).stem
            print(f"  {stem}: V={size['n_vessel']} S={size['n_storage']} "
                  f"C={size['n_charge']} D={size['n_cdu']} "
                  f"props={size['n_props']} H={size['horizon']}")
