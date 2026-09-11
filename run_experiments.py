"""
Unified experiment driver for the priority-slot scheduling experiments.

One entry point, one option to pick the objective.  Two models sit underneath:

  * ``models/crude_oil.mod``   -- crude-oil economic model (operational layer),
    used by the ``gross_margin`` objective.
  * ``models/scheduling.mod``  -- pure priority-slot scheduling core (fixed
    durations, one execution per operation), used by the ``makespan``, ``wct``
    and ``feasibility`` objectives to benchmark the symmetry breaker in isolation.

Objectives (``--objective``)
    gross_margin  crude-oil gross margin (MILP relaxation), NOPSOS+DFA vs SOS+RE.
    makespan      schedule completion time on the scheduling core.
    wct           total weighted completion time on the scheduling core.
    feasibility   prove infeasibility of a deadline just below the optimal
                  makespan on the scheduling core.
    size          DFA-vs-RE automaton size (structural, no solve).

Configurations (``--config``, one or more)
    dfa           NOPSOS + DFA-SBS (Gurobi symmetry off).
    re            SOS + RE-SBS (Gurobi symmetry off) -- gross_margin / size only.
    nosbs-auto    bare NOPSOS, no SBS, Gurobi symmetry at its default (automatic).
    nosbs-off     bare NOPSOS, no SBS, Gurobi symmetry off.

Examples
    python run_experiments.py --objective gross_margin --instances mouret
    python run_experiments.py --objective makespan     --instances suiteB
    python run_experiments.py --objective wct          --instances suiteB --config dfa nosbs-auto
    python run_experiments.py --objective size         --instances suiteA
    python run_experiments.py --running-example
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.dfa_builder   import (build_dfa, accepts, dfa_arc_counts, plot_dfa,
                               forbidden_pairs,
                               topological_sort)
from src.crude_oil     import CrudeOilInstance, build_precedence_edges
from src.crude_oil_ampl import (CrudeOilFullInstance, load_instance_json,
                                 write_crude_oil_dat)
from src.sos_re import build_sos_re_dfa
from src.ampl_interface import (write_nopsos_dat, parse_gurobi_log,
                                 run_crude_oil_experiment, save_results_csv,
                                 ExperimentResult)

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------
AMPL_DIR   = Path(r"C:\Users\juan.contreras\AMPL")
LOCAL_AMPL_EXE = str(AMPL_DIR / "ampl.exe")
AMPLPY_LICENSE_UUID = os.environ.get("AMPL_LICENSE_UUID")
AMPLPY_MODULES = ["highs", "gurobi", "xpress", "cplex"]
NLP_SOLVER = "ipopt"

CRUDE_MOD  = str(Path(__file__).parent / "models" / "crude_oil.mod")
SCHED_MOD  = str(Path(__file__).parent / "models" / "scheduling.mod")
CLASSICAL_MOD = str(Path(__file__).parent / "models" / "jobshop_classical.mod")
DAT_DIR    = str(Path(__file__).parent / "instances")
RESULTS_DIR = str(Path(__file__).parent / "results")

DEFAULT_TIME_LIMIT = 300
DEFAULT_MIP_GAP    = 1e-4
THREADS            = 1
PROC_MIN, PROC_MAX = 1, 10     # processing-time range (scheduling core)
WEIGHT_MIN, WEIGHT_MAX = 1, 10  # WCT weight range
DEADLINE_SLACK = 0.5           # feasibility deadline = makespan_opt - slack

# --config -> (formulation, disable_sbs, gurobi_symmetry)
CONFIGS = {
    "dfa":        ("NOPSOS+DFA", False, 0),
    "re":         ("SOS+RE",     False, 0),
    # Pairwise encoding of the same DFA language: the base automaton is
    # 2-local, so acceptance is equivalent to forbidding each (w,v) pair in
    # consecutive slots.  Flow constraints are dropped (disable_sbs=True) and
    # FPAIRS is filled instead.
    "pairwise":   ("NOPSOS+PAIR", True,  0),
    "nosbs-auto": ("NOPSOS",     True, -1),
    "nosbs-off":  ("NOPSOS",     True,  0),
    # "classical" is not a NOPSOS config: it solves a different model entirely
    # (models/jobshop_classical.mod, the Manne disjunctive MILP) and is handled
    # by run_jobshop_classical, not the shared scheduling core.  jobshop/openshop
    # objectives only.  The tuple is a placeholder so it is a valid --config choice.
    "classical":  ("CLASSICAL",  None,  0),
}
SCHED_OBJECTIVE = {"makespan": "Makespan", "wct": "WCT", "feasibility": "Feasibility"}


def resolve_ampl_exe() -> str:
    """Prefer the AMPL executable managed by amplpy modules and licensed by UUID."""
    try:
        import amplpy.modules as ampl_modules
        from amplpy import ampl_notebook

        installed_modules = set(ampl_modules.installed())
        ampl_exe = ampl_modules.find("ampl")
        if not ampl_exe or not set(AMPLPY_MODULES).issubset(installed_modules):
            if not AMPLPY_LICENSE_UUID:
                raise RuntimeError("Set AMPL_LICENSE_UUID before installing AMPL modules.")
            ampl_notebook(modules=AMPLPY_MODULES, license_uuid=AMPLPY_LICENSE_UUID, verbose=False)
            ampl_exe = ampl_modules.find("ampl")
        if ampl_exe:
            return ampl_exe
    except Exception as exc:
        print(f"[WARN] Could not initialize amplpy AMPL modules: {exc}")
    return LOCAL_AMPL_EXE


# ---------------------------------------------------------------------------
# Shared crude-oil helpers
# ---------------------------------------------------------------------------

def make_dfa_instance(inst: CrudeOilFullInstance) -> CrudeOilInstance:
    resources = inst.R_V + inst.R_S + inst.R_C + inst.R_D
    inlet_ops, outlet_ops = CrudeOilInstance.build_lookup_tables(
        inst.op_inlet, inst.op_outlet, resources)
    return CrudeOilInstance(
        vessels=inst.R_V, storage_tanks=inst.R_S, charging_tanks=inst.R_C,
        cdus=inst.R_D, W_U=inst.W_U, W_T=inst.W_T, W_D=inst.W_D,
        arrival_times={w: inst.s_r[inst.op_inlet[w]] for w in inst.W_U},
        inlet=inst.op_inlet, outlet=inst.op_outlet,
        inlet_ops=inlet_ops, outlet_ops=outlet_ops,
        horizon=inst.H, n_slots=inst.n_slots)


def build_full_operation_no_edges(inst: CrudeOilFullInstance):
    """
    All-operation non-overlapping (NO) graph.

    Every tank, vessel, and CDU is a *unary* resource: any two operations that
    share a resource -- as an inlet or an outlet -- are non-overlapping.  The
    single docking berth additionally makes every pair of vessel unloads
    non-overlapping.
    """
    active_ops = set(inst.W)
    edges = set()

    def add(v, w):
        if v != w and v in active_ops and w in active_ops:
            edges.add(frozenset([v, w]))

    for i, v in enumerate(inst.W_U):
        for w in inst.W_U[i + 1:]:
            add(v, w)
    for r in inst.R_V + inst.R_S + inst.R_C + inst.R_D:
        touch = [w for w in active_ops
                 if inst.op_inlet.get(w) == r or inst.op_outlet.get(w) == r]
        for a in range(len(touch)):
            for b in range(a + 1, len(touch)):
                add(touch[a], touch[b])
    return list(edges)


def repeatable_ops(inst):
    """Operations whose execution cap allows re-execution (eta_w >= 2).

    Mirrors the eta written into the .dat by crude_oil_ampl: inst._eta when
    present, otherwise the uncapped default eta_w = n_slots.  These are the
    operations that get a Rule-4 self-loop in the DFA.
    """
    eta_map = getattr(inst, "_eta", None) or {}
    return [w for w in inst.W if int(eta_map.get(w, inst.n_slots)) >= 2]


def build_instance_dfa(inst: CrudeOilFullInstance, formulation="NOPSOS+DFA",
                       allow_idle=True):
    """Build the automaton + connectivity data for a crude-oil instance."""
    dfa_inst = make_dfa_instance(inst)
    prec_edges = build_precedence_edges(dfa_inst.W_U, dfa_inst.arrival_times)
    no_edges_fs = build_full_operation_no_edges(inst)
    if formulation == "SOS+RE":
        dfa = build_sos_re_dfa(inst, allow_idle=allow_idle)
    else:
        dfa = build_dfa(inst.W, prec_edges, no_edges_fs, allow_idle=allow_idle,
                        repeatable=repeatable_ops(inst))
    return dfa, prec_edges, no_edges_fs


def plot_dfa_for_instance(json_path, output_path=None, show_labels=True):
    inst = load_instance_json(json_path)
    dfa, _, _ = build_instance_dfa(inst)
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, f"{Path(json_path).stem}_dfa.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plot_dfa(dfa, output_path=output_path, title=f"{Path(json_path).stem} DFA",
             show_labels=show_labels)
    print(f"[DFA plot] {output_path}")
    return output_path


def resolve_instance_path(instance_name: str) -> str:
    path = Path(instance_name)
    if path.suffix:
        return str(path)
    return str(Path(DAT_DIR) / f"{instance_name}.json")


MOURET_REFINERIES = ["problem1_mouret", "problem2_mouret",
                     "problem3_mouret", "problem4_mouret"]


def resolve_instances(spec: str) -> list[str]:
    """Resolve the --instances option into a list of JSON paths.

    Named collections: ``mouret`` (the four reference refineries), ``suiteA``
    and ``suiteB`` (the two synthetic suites), ``crude`` (the earlier bundled
    crude-oil instances), ``jobshop``, ``openshop`` and ``synth`` (any such
    files present in the instances directory), and ``all`` (mouret + crude).
    Any other value is treated as a comma-separated list of instance names or
    paths.
    """
    if spec == "mouret":
        names = MOURET_REFINERIES
    elif spec == "suiteA":
        names = [p.stem for p in sorted(Path(DAT_DIR).glob("synthd_*.json"))]
    elif spec == "suiteB":
        names = [p.stem for p in sorted(Path(DAT_DIR).glob("synth100_*.json"))]
    elif spec == "crude":
        names = [p.stem for p in sorted(Path(DAT_DIR).glob("crude_*.json"))]
    elif spec == "jobshop":
        names = [p.stem for p in sorted(Path(DAT_DIR).glob("jobshop_*.json"))]
    elif spec == "openshop":
        names = [p.stem for p in sorted(Path(DAT_DIR).glob("openshop_*.json"))]
    elif spec == "synth":
        names = [p.stem for p in sorted(Path(DAT_DIR).glob("synth_*.json"))]
    elif spec == "all":
        names = (MOURET_REFINERIES
                 + [p.stem for p in sorted(Path(DAT_DIR).glob("crude_*.json"))])
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]
    return [resolve_instance_path(n) for n in names]


def fmt_optional(value, suffix="", precision=2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}{suffix}"
    return f"{value}{suffix}"


# ---------------------------------------------------------------------------
# Objective: gross_margin  (crude-oil economic model)
# ---------------------------------------------------------------------------

def run_gross_margin(json_path, config, time_limit, mip_gap):
    """Solve the crude-oil MILP relaxation (gross margin) for one config."""
    formulation, disable_sbs, symmetry = CONFIGS[config]
    inst = load_instance_json(json_path)
    name = Path(json_path).stem
    tag = f"{name}_{formulation}".replace("+", "_").replace(" ", "_")

    dfa, prec_edges, no_edges_fs = build_instance_dfa(inst, formulation, allow_idle=True)
    inst.dfa = dfa
    if config == "pairwise":
        # Same language, no flow variables: forbid each (w,v) pair in
        # consecutive slots and keep the idle slots in a suffix (IdleTail).
        inst._fpairs = forbidden_pairs(dfa)
    no_directed = []
    for e in no_edges_fs:
        v, w = tuple(e)
        no_directed += [(v, w), (w, v)]
    inst._prec_edges = prec_edges
    inst._no_edges_directed = no_directed

    dat_path = os.path.join(DAT_DIR, f"{tag}.dat")
    os.makedirs(DAT_DIR, exist_ok=True)
    write_crude_oil_dat(dat_path, inst)

    extra = f"symmetry={symmetry}" if symmetry is not None else ""
    return run_crude_oil_experiment(
        instance_name=name, formulation=config, mod_path=CRUDE_MOD,
        dat_path=dat_path, dfa=dfa, W=inst.W, n_slots=inst.n_slots,
        results_dir=RESULTS_DIR, ampl_exe=resolve_ampl_exe(), nlp_solver=NLP_SOLVER,
        milp_time_limit=time_limit, nlp_time_limit=1, mip_gap=mip_gap,
        threads=THREADS, run_nlp=False, disable_sbs=disable_sbs,
        extra_gurobi_opts=extra, objective_name="GrossMargin")


def run_gross_margin_twostep(json_path, config, tl_milp, tl_nlp, mip_gap,
                             max_iter=20):
    """
    Economic comparison via the iterative MILP -> NLP procedure of Mouret et al.

    Step 1 (MILP): solve the bilinear relaxation (composition dropped) to obtain
    a slot sequence.  Step 2 (NLP): fix that sequence, restore the bilinear
    perfect-mixing constraint, and re-solve with Gurobi in nonconvex mode.

    A single pass is not enough.  The relaxation may pick a sequence that meets a
    property bound only because a transfer carries a blend its source tank does
    not hold; restoring perfect mixing then leaves the fixed sequence infeasible.
    When that happens we exclude exactly that assignment with a no-good cut,
    re-solve the MILP, and repeat.  The loop ends on the first sequence that
    admits a bilinear-feasible completion, whose objective is a true gross margin
    comparable with the published figures.

    Both DFA and RE go through the identical procedure, so the comparison stays
    fair; the number of iterations each needs is itself reported.
    """
    from amplpy import AMPL
    formulation, disable_sbs, symmetry = CONFIGS[config]
    inst = load_instance_json(json_path)
    name = Path(json_path).stem
    tag = f"{name}_{formulation}".replace("+", "_").replace(" ", "_")
    dfa, prec_edges, no_edges_fs = build_instance_dfa(inst, formulation, allow_idle=True)
    inst.dfa = dfa
    nod = []
    for e in no_edges_fs:
        v, w = tuple(e)
        nod += [(v, w), (w, v)]
    inst._prec_edges = prec_edges
    inst._no_edges_directed = nod
    dat = os.path.join(DAT_DIR, f"{tag}.dat")
    write_crude_oil_dat(dat, inst)

    row = {"instance_name": name, "config": config, "n_ops": len(inst.W),
           "n_dfa_states": dfa.n_states, "n_dfa_arcs": dfa.n_arcs,
           "milp_obj": None, "milp_status": None, "milp_time": None,
           "nlp_obj": None, "nlp_status": None, "nlp_time": None,
           "iterations": 0, "cuts": 0}

    a = AMPL()
    a.read(CRUDE_MOD)
    a.read_data(dat)
    # Repeated fix/unfix cycles leave solution values carrying rounding noise,
    # which presolve otherwise reports as an inconsistent bound at 1e-17.
    a.eval("option presolve_eps 1e-9;")
    a.eval("objective GrossMargin;")
    # ---- Step 1: MILP relaxation (composition dropped) -> slot sequence ----
    a.eval("drop CompositionBilinear;")
    if disable_sbs:
        a.eval("drop FlowAssign; drop FlowStart; drop FlowOnePerSlot; "
               "drop FlowConserve; drop FlowEnd;")
        a.eval("fix {i in T,(aa,q,r) in ARCS} f[i,aa,q,r] := 0;")
    milp_opts = (f"timelimit={tl_milp} mipgap={mip_gap} threads={THREADS} "
                 f"symmetry={symmetry} dualreductions=0")
    # dualreductions=0 in the NLP step too, otherwise Gurobi answers "infeasible
    # or unbounded", which AMPL reports as a limit and hides a rejected sequence.
    nlp_opts = (f"nonconvex=2 timelimit={tl_nlp} mipgap={mip_gap} "
                f"threads={THREADS} dualreductions=0")
    a.set_option("solver", "gurobi")

    milp_time = nlp_time = 0.0
    for it in range(1, max_iter + 1):
        row["iterations"] = it
        a.set_option("gurobi_options", milp_opts)
        a.solve()
        row["milp_status"] = a.get_value("solve_result")
        milp_time += float(a.get_value("_solve_time") or 0.0)
        if row["milp_status"] not in ("solved", "limit"):
            break                      # the cuts have exhausted the sequences
        row["milp_obj"] = a.get_objective("GrossMargin").value()

        chosen = [k for k, v in a.get_variable("Z").get_values().to_dict().items()
                  if v > 0.5]

        a.eval("fix {i in T, v in W} Z[i,v];")
        if not disable_sbs:
            a.eval("fix {i in T,(aa,q,r) in ARCS} f[i,aa,q,r];")
        a.eval("restore CompositionBilinear;")
        a.set_option("gurobi_options", nlp_opts)
        a.solve()
        row["nlp_status"] = a.get_value("solve_result")
        nlp_time += float(a.get_value("_solve_time") or 0.0)

        if row["nlp_status"] == "solved":
            row["nlp_obj"] = a.get_objective("GrossMargin").value()
            break                      # this sequence admits a true schedule

        # Rejected. Release the sequence first: a no-good cut added while Z is
        # still fixed is violated on the spot, and presolve then reads the model
        # as inconsistent.
        a.eval("drop CompositionBilinear;")
        a.eval("unfix {i in T, v in W} Z[i,v];")
        if not disable_sbs:
            a.eval("unfix {i in T,(aa,q,r) in ARCS} f[i,aa,q,r];")
        terms = " + ".join(f"Z[{int(float(i))},'{v}']" for i, v in chosen)
        a.eval(f"subject to NoGood{it}: {terms} <= {len(chosen) - 1};")
        row["cuts"] = it

    row["milp_time"] = milp_time
    row["nlp_time"] = nlp_time
    a.close()
    return row


# ---------------------------------------------------------------------------
# Objectives on the scheduling core: makespan / wct / feasibility
# ---------------------------------------------------------------------------

def _seed(name: str) -> int:
    return int.from_bytes(hashlib.md5(name.encode()).digest()[:4], "little")


def _instance_field(name, key):
    """Return a named field of an instance file, or None if absent."""
    path = Path(DAT_DIR, f"{name}.json")
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh).get(key)


def _durations(name, W):
    """Processing times for the scheduling core.

    Uses the ``sched_p`` values stored in the instance file when present (so an
    instance carries its own processing times); otherwise falls back to a
    reproducible per-name pseudo-random draw in [PROC_MIN, PROC_MAX].
    """
    stored = _instance_field(name, "sched_p")
    if stored is not None:
        return {v: float(stored[v]) for v in W}
    rng = random.Random(_seed(name))
    return {v: float(rng.randint(PROC_MIN, PROC_MAX)) for v in W}


def _weights(name, W):
    """Completion-time weights for the scheduling core (see :func:`_durations`)."""
    stored = _instance_field(name, "sched_w")
    if stored is not None:
        return {v: float(stored[v]) for v in W}
    rng = random.Random(_seed(name) ^ 0x9E3779B9)
    return {v: float(rng.randint(WEIGHT_MIN, WEIGHT_MAX)) for v in W}


def build_scheduling_dat(name, H, extra_params):
    """Write the scheduling-core .dat (one execution per op, fixed durations)."""
    inst = load_instance_json(str(Path(DAT_DIR, f"{name}.json")))
    W = list(inst.W)
    di = make_dfa_instance(inst)
    prec = build_precedence_edges(di.W_U, di.arrival_times)
    no_fs = build_full_operation_no_edges(inst)
    dfa = build_dfa(W, prec, no_fs, allow_idle=False)
    dat = str(Path(DAT_DIR, f"{name}_sched.dat").resolve())
    write_nopsos_dat(dat, W, len(W), H, prec, no_fs, dfa, extra_params=extra_params)
    return dat, dfa, W


def solve_scheduling(name, dat, objective, disable_sbs, symmetry, time_limit,
                     mip_gap, tag, warmstart=None, fpairs=None, lp_only=False,
                     seed=0):
    """
    Solve the scheduling core with amplpy for one objective/config.

    warmstart
        Optional dict {var_name: {index_tuple: value}} of initial variable values
        loaded as a genuine Gurobi user MIP start (option ``mipstart=1``; the log
        reports "Loaded user MIP start with objective ...").  Note ``mipstart=2``
        would only set variable *hints*, which do not guarantee an incumbent.
        The flow entries are dropped for no-SBS configs, where ``f`` is fixed
        to zero.
    """
    from amplpy import AMPL
    log = str(Path(RESULTS_DIR, f"{name}_{tag}_sched.log").resolve()).replace("\\", "/")
    if os.path.exists(log):
        os.remove(log)
    a = AMPL()
    a.read(SCHED_MOD)
    a.read_data(dat)
    if disable_sbs:
        a.eval("drop FlowAssign; drop FlowStart; drop FlowConserve;")
        a.eval("fix {i in T,(aa,q,r) in ARCS} f[i,aa,q,r] := 0;")
    if fpairs:
        a.get_set("FPAIRS").set_values([tuple(pr) for pr in fpairs])
    a.eval(f"objective {objective};")
    if lp_only:
        a.eval("option relax_integrality 1;")
    if warmstart:
        for vname, vals in warmstart.items():
            if disable_sbs and vname == "f":
                continue                     # f is fixed to 0 in the no-SBS model
            var = a.get_variable(vname)
            if isinstance(vals, dict):
                var.set_values(vals)         # indexed variable (Z, S, D, f)
            else:
                var.set_value(vals)          # scalar (Cmax)
    mipstart_opt = "mipstart=1 " if warmstart else ""
    a.set_option("solver", "gurobi")
    a.set_option("gurobi_options",
                 f"timelimit={time_limit} mipgap={mip_gap} threads={THREADS} "
                 f"symmetry={symmetry} dualreductions=0 seed={seed} "
                 f"{mipstart_opt}logfile='{log}'")
    a.solve()
    try:
        obj = a.get_objective(objective).value()
    except Exception:
        obj = None
    sr = a.get_value("solve_result")
    st = a.get_value("_solve_time")
    a.close()
    gi = parse_gurobi_log(log)
    gap = gi.get("gap")
    return {"obj": obj, "status": sr, "time": st, "nodes": gi.get("n_nodes"),
            "best_bound": gi.get("best_bound"),
            "root_relax": gi.get("root_relax"),
            "gap_pct": 100.0 * gap if gap is not None else None}


def makespan_optimum(name, time_limit, mip_gap):
    """Solve the makespan (DFA-SBS) to get the certified optimal makespan, or None."""
    W = list(load_instance_json(str(Path(DAT_DIR, f"{name}.json"))).W)
    p = _durations(name, W)
    dat, _, _ = build_scheduling_dat(name, sum(p.values()), {"p": p})
    r = solve_scheduling(name, dat, "Makespan", False, 0, time_limit, mip_gap, "mkopt")
    if r["status"] == "solved" and r["obj"] is not None:
        return round(r["obj"])
    return None


def run_scheduling(name, objective, config, time_limit, mip_gap, seed=0):
    """Run one scheduling-core objective for one config on one instance."""
    _, disable_sbs, symmetry = CONFIGS[config]
    obj_name = SCHED_OBJECTIVE[objective]
    W = list(load_instance_json(str(Path(DAT_DIR, f"{name}.json"))).W)
    p = _durations(name, W)

    if objective == "feasibility":
        opt = makespan_optimum(name, time_limit, mip_gap)
        if opt is None:
            return None, None            # optimal makespan not certified -> skip
        H = opt - DEADLINE_SLACK
        dat, dfa, _ = build_scheduling_dat(name, H, {"p": p})
        deadline = H
    else:
        extra = {"p": p}
        if objective == "wct":
            extra["w"] = _weights(name, W)
        dat, dfa, _ = build_scheduling_dat(name, sum(p.values()), extra)
        deadline = None

    fpairs = forbidden_pairs(dfa) if config == "pairwise" else None

    # LP bound of this encoding, before branching and before Gurobi's cuts.
    lp = solve_scheduling(name, dat, obj_name, disable_sbs, symmetry, time_limit,
                          mip_gap, config + "_lp", fpairs=fpairs, lp_only=True)

    r = solve_scheduling(name, dat, obj_name, disable_sbs, symmetry, time_limit,
                         mip_gap, config, fpairs=fpairs, seed=seed)
    row = {"instance_name": name, "config": config, "objective": objective,
           "seed": seed,
           "status": r["status"], "obj": r["obj"], "best_bound": r["best_bound"],
           "gap_pct": r["gap_pct"], "time": r["time"], "n_nodes": r["nodes"],
           "lp_bound": lp["obj"], "root_relax": r["root_relax"],
           "n_sbs_rows": (len(fpairs) * (len(W) - 1)) if fpairs is not None else None,
           "deadline": deadline, "n_ops": len(W),
           "n_dfa_states": dfa.n_states, "n_dfa_arcs": dfa.n_arcs}
    return row, dfa


# ---------------------------------------------------------------------------
# Objective: jobshop  (job-shop makespan on the scheduling core)
# ---------------------------------------------------------------------------
#
# A second NOPSOS application.  A job-shop instance maps onto the two NOPSOS
# relations with no operational layer: the routing chain of each job gives the
# precedence graph, and each machine (a unary resource) gives a non-overlapping
# clique.  The scheduling core models/scheduling.mod then solves the makespan
# objective unchanged, and the SBS-on/off configs measure the symmetry breaker
# exactly as for the crude-oil scheduling core.

def load_jobshop_json(json_path):
    """Load a raw job-shop instance dict (see instance_generator.generate_jobshop_instance)."""
    with open(json_path) as fh:
        return json.load(fh)


def build_jobshop_graphs(inst):
    """
    Build the NOPSOS inputs for a job-shop instance.

    Returns
    -------
    W : list[str]
        Operation names ``J{job}P{position}`` in job/position order.
    prec_edges : list[tuple[str, str]]
        Routing precedence (v, w): op v immediately precedes op w in its job.
    no_edges : list[frozenset]
        One undirected edge per pair of operations sharing a machine.
    p : dict[str, float]
        Processing time of each operation.
    """
    jobs = inst["jobs"]
    W, p = [], {}
    prec_edges = []
    machine_ops = defaultdict(list)          # machine -> operations on it

    for j, ops in enumerate(jobs, start=1):
        prev = None
        for k, (machine, proc) in enumerate(ops, start=1):
            name = f"J{j}P{k}"
            W.append(name)
            p[name] = float(proc)
            machine_ops[machine].append(name)
            if prev is not None:
                prec_edges.append((prev, name))   # routing chain
            prev = name

    no_edges = []
    for ops in machine_ops.values():             # machine clique in G_NO
        for a in range(len(ops)):
            for b in range(a + 1, len(ops)):
                no_edges.append(frozenset([ops[a], ops[b]]))

    return W, prec_edges, no_edges, p


def build_openshop_graphs(inst):
    """
    Build the NOPSOS inputs for an open-shop instance.

    Open-shop drops the routing order, so the precedence graph is empty and the
    non-overlapping graph is the union of two clique families: one per machine
    (a machine runs one operation at a time) and one per job (a job is worked on
    by one machine at a time).  Returns (W, prec_edges=[], no_edges, p).
    """
    jobs = inst["jobs"]
    W, p = [], {}
    machine_ops = defaultdict(list)              # machine -> operations on it
    job_ops = defaultdict(list)                  # job -> its operations

    for j, ops in enumerate(jobs, start=1):
        for k, (machine, proc) in enumerate(ops, start=1):
            name = f"J{j}P{k}"
            W.append(name)
            p[name] = float(proc)
            machine_ops[machine].append(name)
            job_ops[j].append(name)

    no_edges = []
    for group in list(machine_ops.values()) + list(job_ops.values()):
        for a in range(len(group)):              # machine and job cliques in G_NO
            for b in range(a + 1, len(group)):
                no_edges.append(frozenset([group[a], group[b]]))

    return W, [], no_edges, p                    # no routing precedence


def jobshop_cmax_lb(inst):
    """
    Structural makespan lower bound for a job-shop instance: the largest amount
    of processing that must happen sequentially on any single machine (machine
    load) or within any single job (job length).  Both are sets of operations
    that cannot overlap, so Cmax is at least each such sum; the binding one is
    their maximum.
    """
    jobs = inst["jobs"]
    job_len = [sum(proc for _m, proc in ops) for ops in jobs]
    mach_load = defaultdict(float)
    for ops in jobs:
        for m, proc in ops:
            mach_load[m] += proc
    return float(max(max(job_len), max(mach_load.values())))


def jobshop_wct_weights(name, inst):
    """
    Per-job weights for the total weighted completion time objective.

    Job-shop WCT is the standard sum_j w_j C_j, where C_j is the completion time
    of job j (its last operation).  The scheduling core's WCT objective weights
    every operation, so we realise the per-job form by putting the (reproducible,
    random) job weight w_j on each job's terminal operation and 0 on the others.
    """
    seed = int.from_bytes(hashlib.md5(f"wct:{name}".encode()).digest()[:4], "little")
    rng = random.Random(seed)
    w = {}
    for j, ops in enumerate(inst["jobs"], start=1):
        wj = float(rng.randint(WEIGHT_MIN, WEIGHT_MAX))
        last = len(ops)
        for pos in range(1, last + 1):
            w[f"J{j}P{pos}"] = wj if pos == last else 0.0
    return w


def completion_upper_bounds(W, prec_edges, p, H):
    """
    Per-operation upper bound on completion time: ``H - tail[w]``, where
    ``tail[w]`` is the longest precedence chain of processing that must run after
    w (the classic Carlier tail).  Those successors form a chain that must finish
    by H, so w must complete by H - tail[w].  Reduces to H for operations with no
    successors.  Used as the tightened big-M ``Mno`` in the non-overlapping rows.
    """
    succ = defaultdict(list)
    for v, w in prec_edges:                      # v precedes w
        succ[v].append(w)
    tail = {w: 0.0 for w in W}
    for w in reversed(topological_sort(W, prec_edges)):
        tail[w] = max((p[s] + tail[s] for s in succ[w]), default=0.0)
    return {w: H - tail[w] for w in W}


def build_jobshop_dat(name, objective="makespan", cmax_cut=False, tight_bigm=False,
                      tight_horizon=False):
    """
    Write the scheduling-core .dat for a job-shop instance (one exec per op).

    objective
        "makespan" or "wct".  For "wct" the per-job completion-time weights are
        written (see jobshop_wct_weights); the horizon must stay at sum(p),
        since the tight (list-schedule makespan) horizon is a valid bound only
        for the makespan objective -- a WCT optimum may use a larger makespan.
    cmax_cut
        If True, supply the machine-load / job-length makespan lower bound as
        ``cmax_lb`` (activates the CmaxLowerBound constraint).  Off by default:
        experiments showed the aggregate bound is usually slack and can perturb
        the solver onto a worse trajectory (e.g. it doubled the solve time on
        jobshop_06), helping only when it happens to equal the optimum.
    tight_bigm
        If True, supply the per-operation big-M ``Mno[w] = H - tail[w]`` for the
        non-overlapping rows instead of the loose horizon H.  Correctness-
        preserving (a valid completion upper bound) and tightens the LP directly
        where the O(|NO| n^2) rows are loose.
    tight_horizon
        If True, set H to the makespan of the DFA-order list schedule (a valid
        upper bound that exploits machine parallelism) instead of the loose
        sum(p).  Shrinking H tightens every big-M -- both Activate and NoOverlap
        (whose Mno defaults to H) -- and every time-variable range at once.  The
        same list schedule is returned as ``ws_start`` for a matching warm start.

    Returns (dat_path, dfa, W, p, ws_start), where ws_start is the list-schedule
    start-time dict when tight_horizon is on, else None (serial warm start).
    """
    inst = load_jobshop_json(str(Path(DAT_DIR, f"{name}.json")))
    W, prec, no_fs, p = build_jobshop_graphs(inst)
    dfa = build_dfa(W, prec, no_fs, allow_idle=False)

    ws_start = None
    if tight_horizon:
        ws_start, H = jobshop_list_schedule(inst, dfa, p)   # parallel makespan
    else:
        H = sum(p.values())                                 # trivial upper bound

    extra = {"p": p}
    if objective == "wct":
        extra["w"] = jobshop_wct_weights(name, inst)
    if cmax_cut:
        extra["cmax_lb"] = jobshop_cmax_lb(inst)
    if tight_bigm:
        extra["Mno"] = completion_upper_bounds(W, prec, p, H)
    dat = str(Path(DAT_DIR, f"{name}_jobshop.dat").resolve())
    write_nopsos_dat(dat, W, len(W), H, prec, no_fs, dfa, extra_params=extra)
    return dat, dfa, W, p, ws_start


def jobshop_list_schedule(inst, dfa, p):
    """
    List-schedule the operations in the DFA operation order and return their
    start times and the resulting makespan.

    We dispatch the operations in ``dfa.operation_order`` -- a topological order
    (so a job's operations are released in routing order) that is also DFA-
    accepted.  Each operation starts as early as possible: the later of its
    machine becoming free and its job's previous operation finishing.  Because we
    dispatch in slot order and each machine's free time only advances, two
    operations sharing a machine end up in slot-order = time-order, so the
    non-overlapping rows hold; releasing a job's operations in routing order
    honours precedence.  Unlike the serial schedule (makespan sum(p)), this
    exploits machine parallelism, so the makespan -- used both as the tightened
    horizon H and as the warm-start incumbent -- is far below sum(p).
    """
    op_machine, op_job = {}, {}
    for j, ops in enumerate(inst["jobs"]):
        for k, (machine, _proc) in enumerate(ops):
            op_machine[f"J{j+1}P{k+1}"] = machine
            op_job[f"J{j+1}P{k+1}"] = j
    machine_free = defaultdict(float)
    job_ready = defaultdict(float)
    start = {}
    for op in dfa.operation_order:
        m, j = op_machine[op], op_job[op]
        st = max(machine_free[m], job_ready[j])
        start[op] = st
        end = st + p[op]
        machine_free[m] = end
        job_ready[j] = end
    makespan = max((start[op] + p[op] for op in dfa.operation_order), default=0.0)
    return start, makespan


def jobshop_warmstart(dfa, p, start_times=None):
    """
    A feasible MIP-start for the scheduling core over the DFA operation order.

    The word ``dfa.operation_order`` is a topological order (so it respects
    precedence when read slot by slot) and is always accepted by the DFA -- it
    uses only the Rule-1 arc out of q0 and Rule-2 forward arcs.  Two timings give
    a feasible schedule over this order:

    * ``start_times=None`` -- schedule serially (each operation starts when the
      previous finishes); no two operations overlap, makespan = sum(p).
    * ``start_times`` given -- use the supplied start times (e.g. from
      :func:`jobshop_list_schedule`), which pack operations in parallel across
      machines for a much tighter makespan.

    Because ``f`` is continuous, ``Z`` is the only integer variable, so this Z
    (plus the consistent S, D, f, Cmax) is all Gurobi needs to seed an incumbent.

    Returns a dict {var_name: {index_tuple: value}} for solve_scheduling.
    """
    order = dfa.operation_order
    arc_index = {(fr, to, lab): a
                 for a, (fr, to, lab) in enumerate(dfa.arcs, start=1)}
    Z, S, D, f = {}, {}, {}, {}
    serial_t = 0.0
    prev_state = dfa.initial
    for i, op in enumerate(order, start=1):          # slot i (1-based)
        to_state = dfa.state_of[op]
        Z[(i, op)] = 1
        S[(i, op)] = serial_t if start_times is None else start_times[op]
        D[(i, op)] = p[op]
        a = arc_index[(prev_state, to_state, op)]    # Rule-1 (i=1) or Rule-2 arc
        f[(i, a, prev_state, to_state)] = 1.0
        serial_t += p[op]
        prev_state = to_state
    cmax = max((S[(i, op)] + D[(i, op)] for i, op in enumerate(order, start=1)),
               default=0.0)
    return {"Z": Z, "S": S, "D": D, "f": f, "Cmax": cmax}


def run_jobshop(name, config, time_limit, mip_gap, objective="makespan",
                warmstart=True, cmax_cut=False, tight_bigm=False,
                tight_horizon=False, seed=0):
    """Solve one job-shop objective (makespan or wct) for one config on one instance."""
    _, disable_sbs, symmetry = CONFIGS[config]
    obj_name = {"makespan": "Makespan", "wct": "WCT"}[objective]
    if objective == "wct":
        tight_horizon = False        # list-schedule makespan is a valid H only for makespan
    dat, dfa, W, p, ws_start = build_jobshop_dat(
        name, objective=objective, cmax_cut=cmax_cut, tight_bigm=tight_bigm,
        tight_horizon=tight_horizon)
    ws = jobshop_warmstart(dfa, p, start_times=ws_start) if warmstart else None
    fpairs = forbidden_pairs(dfa) if config == "pairwise" else None

    # LP bound of this encoding, before branching and before Gurobi's cuts.
    lp = solve_scheduling(name, dat, obj_name, disable_sbs, symmetry,
                          time_limit, mip_gap, config + "_lp", fpairs=fpairs,
                          lp_only=True)

    r = solve_scheduling(name, dat, obj_name, disable_sbs, symmetry,
                         time_limit, mip_gap, config, warmstart=ws, fpairs=fpairs,
                         seed=seed)
    row = {"instance_name": name, "config": config, "objective": f"jobshop_{objective}",
           "seed": seed,
           "status": r["status"], "obj": r["obj"], "best_bound": r["best_bound"],
           "gap_pct": r["gap_pct"], "time": r["time"], "n_nodes": r["nodes"],
           "lp_bound": lp["obj"], "root_relax": r["root_relax"],
           "n_sbs_rows": (len(fpairs) * (len(W) - 1)) if fpairs is not None else None,
           "n_ops": len(W), "n_dfa_states": dfa.n_states, "n_dfa_arcs": dfa.n_arcs}
    return row, dfa


# ---------------------------------------------------------------------------
# Classical disjunctive (Manne) job-shop MILP -- external Gurobi baseline
# ---------------------------------------------------------------------------
#
# Not a symmetry-breaker configuration: this solves a *different* formulation
# (models/jobshop_classical.mod) with continuous start times and one binary
# ordering variable per machine-sharing pair, so there are no priority slots and
# no slot-permutation symmetry at all.  It is the "textbook alternative" column
# that shows where the priority-slot NOPSOS model stands against the compact
# formulation Gurobi is built to solve.  Both makespan and WCT are supported;
# the WCT weights and the list-schedule warm start match the NOPSOS runs so the
# comparison is fair (same instances, same incumbent seed, same time limit).

def write_jobshop_classical_dat(dat_path, W, p, prec_edges, mpairs, M, weights=None):
    """Write the .dat for the classical disjunctive job-shop model."""
    out = [f"set OPS := {' '.join(W)} ;", "param p :="]
    out += [f"  {v} {p[v]}" for v in W]
    out += [";", f"param M := {M} ;"]
    if weights is not None:
        out.append("param w :=")
        out += [f"  {v} {weights.get(v, 0.0)}" for v in W]
        out.append(";")
    out.append("set PREC :=")
    out += [f"  {v} {ww}" for (v, ww) in prec_edges]
    out.append(";")
    out.append("set MPAIR :=")
    out += [f"  {o} {o2}" for (o, o2) in mpairs]
    out.append(";")
    with open(dat_path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def run_jobshop_classical(name, time_limit, mip_gap, objective="makespan",
                          warmstart=True):
    """Solve one job-shop objective with the classical disjunctive MILP baseline."""
    from amplpy import AMPL
    inst = load_jobshop_json(str(Path(DAT_DIR, f"{name}.json")))
    W, prec, no_fs, p = build_jobshop_graphs(inst)
    pos = {v: k for k, v in enumerate(W)}
    mpairs = []
    for e in no_fs:                                  # machine-sharing pairs
        a, b = tuple(e)
        mpairs.append((a, b) if pos[a] < pos[b] else (b, a))
    M = sum(p.values())                              # valid horizon / big-M
    weights = jobshop_wct_weights(name, inst) if objective == "wct" else None

    dat = str(Path(DAT_DIR, f"{name}_classical.dat").resolve())
    write_jobshop_classical_dat(dat, W, p, prec, mpairs, M, weights)

    # Warm start: the same DFA-order list schedule the NOPSOS runs use, expressed
    # in the classical variables (start times + the induced machine orderings).
    ws_start = None
    if warmstart:
        dfa = build_dfa(W, prec, no_fs, allow_idle=False)
        ws_start, _ = jobshop_list_schedule(inst, dfa, p)

    obj_name = {"makespan": "Makespan", "wct": "WCT"}[objective]
    log = str(Path(RESULTS_DIR, f"{name}_classical_{objective}.log").resolve()).replace("\\", "/")
    if os.path.exists(log):
        os.remove(log)

    a = AMPL()
    a.read(CLASSICAL_MOD)
    a.read_data(dat)
    a.eval(f"objective {obj_name};")
    mipstart_opt = ""
    if ws_start is not None:
        a.get_variable("s").set_values(ws_start)
        a.get_variable("Cmax").set_value(max(ws_start[v] + p[v] for v in W))
        xvals = {(o, o2): (1 if ws_start[o] <= ws_start[o2] else 0)
                 for (o, o2) in mpairs}
        a.get_variable("x").set_values(xvals)
        mipstart_opt = "mipstart=1 "
    a.set_option("solver", "gurobi")
    a.set_option("gurobi_options",
                 f"timelimit={time_limit} mipgap={mip_gap} threads={THREADS} "
                 f"dualreductions=0 {mipstart_opt}logfile='{log}'")
    a.solve()
    try:
        obj = a.get_objective(obj_name).value()
    except Exception:
        obj = None
    sr = a.get_value("solve_result")
    st = a.get_value("_solve_time")
    a.close()
    gi = parse_gurobi_log(log)
    gap = gi.get("gap")
    row = {"instance_name": name, "config": "classical",
           "objective": f"jobshop_{objective}", "status": sr, "obj": obj,
           "best_bound": gi.get("best_bound"),
           "gap_pct": 100.0 * gap if gap is not None else None,
           "time": st, "n_nodes": gi.get("n_nodes"), "n_ops": len(W),
           "n_dfa_states": "", "n_dfa_arcs": ""}
    return row


# ---------------------------------------------------------------------------
# Objective: openshop  (open-shop makespan on the scheduling core)
# ---------------------------------------------------------------------------
#
# A third NOPSOS application.  Open-shop drops the routing order, so precedence
# is empty and the non-overlapping graph is the union of machine cliques and job
# cliques (build_openshop_graphs).  The scheduling core solves the makespan
# unchanged; with no precedence the priority-slot symmetry is maximal, so this
# stresses the DFA-SBS harder than job-shop.  The list-schedule warm start and
# the tight horizon carry over verbatim (jobshop_list_schedule / jobshop_warmstart
# are structure-agnostic: they dispatch in the DFA order and honour whatever
# unary resources the two graphs encode).

def build_openshop_dat(name, tight_horizon=False):
    """Write the scheduling-core .dat for an open-shop instance (makespan)."""
    inst = load_jobshop_json(str(Path(DAT_DIR, f"{name}.json")))
    W, prec, no_fs, p = build_openshop_graphs(inst)
    dfa = build_dfa(W, prec, no_fs, allow_idle=False)
    ws_start = None
    if tight_horizon:
        ws_start, H = jobshop_list_schedule(inst, dfa, p)   # parallel makespan
    else:
        H = sum(p.values())                                 # trivial upper bound
    dat = str(Path(DAT_DIR, f"{name}_openshop.dat").resolve())
    write_nopsos_dat(dat, W, len(W), H, prec, no_fs, dfa, extra_params={"p": p})
    return dat, dfa, W, p, ws_start


def run_openshop(name, config, time_limit, mip_gap, warmstart=True,
                 tight_horizon=False):
    """Solve the open-shop makespan for one config on one instance."""
    _, disable_sbs, symmetry = CONFIGS[config]
    dat, dfa, W, p, ws_start = build_openshop_dat(name, tight_horizon=tight_horizon)
    ws = jobshop_warmstart(dfa, p, start_times=ws_start) if warmstart else None
    r = solve_scheduling(name, dat, "Makespan", disable_sbs, symmetry,
                         time_limit, mip_gap, config, warmstart=ws)
    row = {"instance_name": name, "config": config, "objective": "openshop",
           "status": r["status"], "obj": r["obj"], "best_bound": r["best_bound"],
           "gap_pct": r["gap_pct"], "time": r["time"], "n_nodes": r["nodes"],
           "n_ops": len(W), "n_dfa_states": dfa.n_states, "n_dfa_arcs": dfa.n_arcs}
    return row, dfa


# ---------------------------------------------------------------------------
# Objective: size  (DFA vs RE, structural, no solve)
# ---------------------------------------------------------------------------

def run_size(json_path):
    inst = load_instance_json(json_path)
    name = Path(json_path).stem
    dfa, _, _ = build_instance_dfa(inst, "NOPSOS+DFA", allow_idle=True)
    re = build_sos_re_dfa(inst, allow_idle=True)
    ns = inst.n_slots
    return {"instance": name, "n_ops": len(inst.W), "n_slots": ns,
            "dfa_states": dfa.n_states, "dfa_arcs": dfa.n_arcs,
            "dfa_sbs_vars": ns * dfa.n_arcs,
            "re_states": re.n_states, "re_arcs": re.n_arcs,
            "re_sbs_vars": ns * re.n_arcs,
            "state_ratio": round(re.n_states / dfa.n_states, 2),
            "arc_ratio": round(re.n_arcs / dfa.n_arcs, 2)}


# ---------------------------------------------------------------------------
# Objective: symmetry  (how many admissible slot-sequences the DFA removes)
# ---------------------------------------------------------------------------

SYMMETRY_CAP = 20   # max |W| for exact subset-DP enumeration (2^|W| memory/time)


def _count_linext(n, pred_bits):
    """Number of permutations of {0..n-1} respecting the precedence bitmasks."""
    f = [0] * (1 << n)
    f[0] = 1
    for S in range(1 << n):
        fs = f[S]
        if fs == 0:
            continue
        for v in range(n):
            b = 1 << v
            if S & b or (pred_bits[v] & ~S):
                continue          # v already placed or a predecessor is missing
            f[S | b] += fs
    return f[(1 << n) - 1]


def _count_dfa_perms(n, adj_bits, pred_bits=None):
    """Permutations accepted by the DFA and respecting precedence.

    Every consecutive pair must be an arc of the automaton (adj_bits) and every
    operation must follow its precedence predecessors (pred_bits), which is what
    constraint (5) of the model enforces.  Counting without pred_bits would admit
    sequences the model rejects anyway and understate the reduction.
    """
    if pred_bits is None:
        pred_bits = [0] * n
    size = 1 << n
    g = [[0] * n for _ in range(size)]
    for v in range(n):
        if pred_bits[v] == 0:     # Rule 1: any op with no predecessor may start
            g[1 << v][v] = 1
    for S in range(size):
        gs = g[S]
        for u in range(n):
            c = gs[u]
            if c == 0:
                continue
            m = adj_bits[u] & ~S
            while m:
                b = m & (-m)
                v = b.bit_length() - 1
                if not (pred_bits[v] & ~S):
                    g[S | b][v] += c
                m ^= b
    return sum(g[size - 1])


def run_symmetry(json_path, cap=SYMMETRY_CAP):
    """
    Solver-independent symmetry reduction on the scheduling core: number of
    precedence-respecting slot-sequences admitted by the bare model vs. accepted
    by the DFA (each operation once).  The ratio is how many symmetric duplicates
    the automaton removes.  Exact subset-DP enumeration, feasible for |W| <= cap.
    """
    inst = load_instance_json(json_path)
    name = Path(json_path).stem
    W = list(inst.W)
    n = len(W)
    idx = {v: i for i, v in enumerate(W)}
    di = make_dfa_instance(inst)
    prec = build_precedence_edges(di.W_U, di.arrival_times)
    no_fs = build_full_operation_no_edges(inst)
    dfa = build_dfa(W, prec, no_fs, allow_idle=False)
    row = {"instance": name, "n_ops": n, "n_dfa_states": dfa.n_states,
           "n_dfa_arcs": dfa.n_arcs, "n_seq_no_sbs": None, "n_seq_dfa": None,
           "reduction": None}
    if n > cap:
        return row

    rank = {v: i for i, v in enumerate(dfa.operation_order)}
    no_set = set()
    for e in no_fs:
        a, b = tuple(e)
        no_set.add((a, b))
        no_set.add((b, a))
    pred_bits = [0] * n
    for (u, v) in prec:
        pred_bits[idx[v]] |= 1 << idx[u]
    adj_bits = [0] * n
    for u in W:
        for v in W:
            if u == v:
                continue
            # Rule 2 (forward in the total order) or Rule 3 (backward NO arc).
            if rank[u] < rank[v] or ((u, v) in no_set and rank[v] < rank[u]):
                adj_bits[idx[u]] |= 1 << idx[v]

    n_prec = _count_linext(n, pred_bits)
    n_dfa = _count_dfa_perms(n, adj_bits, pred_bits)
    row["n_seq_no_sbs"] = n_prec
    row["n_seq_dfa"] = n_dfa
    row["reduction"] = round(n_prec / n_dfa, 1) if n_dfa else None
    return row


# ---------------------------------------------------------------------------
# Small worked example (no solve)
# ---------------------------------------------------------------------------

def running_example():
    W = ["w1", "w2", "w3", "w4", "w5"]
    prec_edges = [("w3","w2"),("w3","w4"),("w2","w1"),("w1","w5"),("w2","w5")]
    no_edges   = [frozenset({"w1","w2"}), frozenset({"w3","w4"}),
                  frozenset({"w3","w5"})]
    dfa = build_dfa(W, prec_edges, no_edges)
    r1, r2, r3, _ = dfa_arc_counts(dfa)
    print("=" * 60)
    print("Small worked example")
    print("=" * 60)
    print(f"Total order : {dfa.operation_order}")
    print(f"DFA         : {dfa.n_states} states, {dfa.n_arcs} arcs "
          f"(Rule1={r1} Rule2={r2} Rule3={r3})")
    a, b = ["w3","w2","w1","w4","w5"], ["w3","w2","w4","w1","w5"]
    acc_a, acc_b = accepts(dfa, a), accepts(dfa, b)
    assert acc_a != acc_b, "Symmetry not broken!"
    print(f"  Accepted (canonical): {a if acc_a else b}")
    print(f"  Rejected (duplicate): {a if not acc_a else b}")
    dat_path = os.path.join(DAT_DIR, "running_example.dat")
    os.makedirs(DAT_DIR, exist_ok=True)
    write_nopsos_dat(dat_path, W, len(W), 100.0, prec_edges, no_edges, dfa)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def append_rows(rows, csv_path, fresh):
    if not rows:
        return
    exists = os.path.exists(csv_path) and not fresh
    mode = "a" if exists else "w"
    with open(csv_path, mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective",
                    choices=["gross_margin", "makespan", "wct", "feasibility",
                             "jobshop", "openshop", "size", "symmetry"],
                    help="Which objective / experiment to run.")
    ap.add_argument("--instances", default=None,
                    help="'mouret', 'suiteA', 'suiteB', 'crude', 'jobshop', "
                         "'openshop', 'all', or a comma-separated list. "
                         "Default: mouret for gross_margin, jobshop/openshop "
                         "for those, crude otherwise.")
    ap.add_argument("--config", nargs="+", default=None,
                    choices=list(CONFIGS.keys()),
                    help="Configurations to run. Default depends on --objective.")
    ap.add_argument("--seeds", default="0",
                    help="Comma-separated Gurobi seeds; every run is repeated once "
                         "per seed (variability study).")
    ap.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT)
    ap.add_argument("--mip-gap", type=float, default=DEFAULT_MIP_GAP)
    ap.add_argument("--cmax-cut", action="store_true",
                    help="jobshop only: add the machine-load / job-length makespan "
                         "lower-bound cut (usually slack; off by default).")
    ap.add_argument("--tight-bigm", action="store_true",
                    help="jobshop only: use the per-operation big-M H - tail[w] in "
                         "the non-overlapping rows instead of the horizon H.")
    ap.add_argument("--tight-horizon", action="store_true",
                    help="jobshop only: set H to the DFA-order list-schedule "
                         "makespan (a parallel upper bound) instead of sum(p); the "
                         "same schedule seeds the warm start.")
    ap.add_argument("--jobshop-objective", choices=["makespan", "wct"],
                    default="makespan",
                    help="jobshop only: objective to solve (default makespan). "
                         "wct = total weighted job completion time.")
    ap.add_argument("--nlp", action="store_true",
                    help="gross_margin only: run the MILP->NLP two-step and report "
                         "the true (bilinear-feasible) objective for a fair comparison.")
    ap.add_argument("--nlp-time-limit", type=int, default=120,
                    help="Time limit (s) for the NLP step of --nlp.")
    ap.add_argument("--out", default=None, help="Output CSV path.")
    ap.add_argument("--running-example", action="store_true",
                    help="Print the small worked example and exit.")
    return ap.parse_args()


def default_config(objective):
    if objective == "gross_margin":
        return ["dfa", "re"]
    # makespan / wct / feasibility / jobshop
    return ["dfa", "nosbs-auto", "nosbs-off"]


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DAT_DIR, exist_ok=True)

    if args.running_example or args.objective is None:
        running_example()
        if args.objective is None:
            print("\nNo --objective given; ran the running example only.")
            return

    obj = args.objective
    default_spec = {"gross_margin": "mouret", "jobshop": "jobshop",
                    "openshop": "openshop"}.get(obj, "crude")
    instances_spec = args.instances or default_spec
    instance_files = resolve_instances(instances_spec)
    out_tag = f"jobshop_{args.jobshop_objective}" if obj == "jobshop" else obj
    out = args.out or os.path.join(RESULTS_DIR, f"{out_tag}_experiments.csv")

    print("=" * 72)
    print(f"objective={obj}  instances={instances_spec} ({len(instance_files)})  "
          f"time_limit={args.time_limit}s")
    print("=" * 72)

    # ---- size: structural, no solve -------------------------------------
    if obj == "size":
        rows = []
        for jp in instance_files:
            if not os.path.exists(jp):
                print(f"[WARN] not found: {jp}"); continue
            r = run_size(jp)
            rows.append(r)
            print(f"  {r['instance']:<10} DFA {r['dfa_states']:>3}st/{r['dfa_arcs']:>4}arcs  "
                  f"RE {r['re_states']:>3}st/{r['re_arcs']:>4}arcs  "
                  f"state x{r['state_ratio']}")
        append_rows(rows, out, fresh=True)
        return

    # ---- symmetry: sequence-count reduction, no solve -------------------
    if obj == "symmetry":
        rows = []
        for jp in instance_files:
            if not os.path.exists(jp):
                print(f"[WARN] not found: {jp}"); continue
            r = run_symmetry(jp)
            rows.append(r)
            if r["reduction"] is not None:
                print(f"  {r['instance']:<10} |W|={r['n_ops']:>2}  "
                      f"no-SBS seqs={float(r['n_seq_no_sbs']):.3g}  "
                      f"DFA seqs={float(r['n_seq_dfa']):.3g}  "
                      f"reduction x{r['reduction']}")
            else:
                print(f"  {r['instance']:<10} |W|={r['n_ops']:>2}  "
                      f"(> {SYMMETRY_CAP} ops, not enumerated)")
        append_rows(rows, out, fresh=True)
        return

    # ---- gross_margin: crude-oil model ----------------------------------
    if obj == "gross_margin":
        configs = args.config or default_config(obj)
        if args.nlp:
            # Fair MILP->NLP two-step: report the true (bilinear) objective.
            out = args.out or os.path.join(RESULTS_DIR, "gross_margin_nlp_experiments.csv")
            rows = []
            for jp in instance_files:
                if not os.path.exists(jp):
                    print(f"[WARN] not found: {jp}"); continue
                print(f"\n{Path(jp).stem}")
                for cfg in configs:
                    r = run_gross_margin_twostep(jp, cfg, args.time_limit,
                                                 args.nlp_time_limit, args.mip_gap)
                    rows.append(r)
                    print(f"  {cfg:<11} MILP obj={fmt_optional(r['milp_obj'])} "
                          f"({r['milp_status']})  ->  NLP(true) obj={fmt_optional(r['nlp_obj'])} "
                          f"({r['nlp_status']})  dfa={r['n_dfa_states']}/{r['n_dfa_arcs']}")
            append_rows(rows, out, fresh=True)
            return
        results = []
        for jp in instance_files:
            if not os.path.exists(jp):
                print(f"[WARN] not found: {jp}"); continue
            print(f"\n{Path(jp).stem}")
            for cfg in configs:
                r = run_gross_margin(jp, cfg, args.time_limit, args.mip_gap)
                results.append(r)
                print(f"  {cfg:<11} status={r.status} obj={fmt_optional(r.obj_value)} "
                      f"gap={fmt_optional(r.gap_pct(), '%')} "
                      f"time={fmt_optional(r.milp_time, 's', precision=1)} "
                      f"dfa={r.n_dfa_states}/{r.n_dfa_arcs}")
        save_results_csv(results, out)
        return

    # ---- jobshop: job-shop makespan on the scheduling core --------------
    if obj == "jobshop":
        configs = args.config or default_config(obj)
        configs = [c for c in configs if c != "re"]   # RE applies to crude-oil only
        rows = []
        fresh = True
        for jp in instance_files:
            name = Path(jp).stem
            if not os.path.exists(jp):
                print(f"[WARN] not found: {jp}"); continue
            print(f"\n{name}")
            for cfg in configs:
                if cfg == "classical":
                    row = run_jobshop_classical(name, args.time_limit, args.mip_gap,
                                                objective=args.jobshop_objective)
                else:
                    row, _dfa = run_jobshop(name, cfg, args.time_limit, args.mip_gap,
                                            objective=args.jobshop_objective,
                                            cmax_cut=args.cmax_cut,
                                            tight_bigm=args.tight_bigm,
                                            tight_horizon=args.tight_horizon)
                rows.append(row)
                print(f"  {cfg:<11} status={row['status']} "
                      f"obj={fmt_optional(row['obj'])} nodes={row['n_nodes']} "
                      f"time={fmt_optional(row['time'], 's', precision=1)} "
                      f"dfa={row['n_dfa_states']}/{row['n_dfa_arcs']}")
            if rows:
                append_rows(rows, out, fresh=fresh)
                fresh = False
                rows = []
        return

    # ---- openshop: open-shop makespan on the scheduling core ------------
    if obj == "openshop":
        configs = args.config or default_config(obj)
        configs = [c for c in configs if c != "re"]   # RE applies to crude-oil only
        rows = []
        fresh = True
        for jp in instance_files:
            name = Path(jp).stem
            if not os.path.exists(jp):
                print(f"[WARN] not found: {jp}"); continue
            print(f"\n{name}")
            for cfg in configs:
                row, dfa = run_openshop(name, cfg, args.time_limit, args.mip_gap,
                                        tight_horizon=args.tight_horizon)
                rows.append(row)
                print(f"  {cfg:<11} status={row['status']} "
                      f"obj={fmt_optional(row['obj'])} nodes={row['n_nodes']} "
                      f"time={fmt_optional(row['time'], 's', precision=1)} "
                      f"dfa={row['n_dfa_states']}/{row['n_dfa_arcs']}")
            if rows:
                append_rows(rows, out, fresh=fresh)
                fresh = False
                rows = []
        return

    # ---- makespan / wct / feasibility: scheduling core ------------------
    configs = args.config or default_config(obj)
    if obj == "feasibility":
        configs = [c for c in configs if c != "re"]
    rows = []
    fresh = True
    for jp in instance_files:
        name = Path(jp).stem
        if not os.path.exists(jp):
            print(f"[WARN] not found: {jp}"); continue
        print(f"\n{name}")
        for cfg in configs:
            if cfg == "re":
                print(f"  {cfg:<11} skipped (RE applies to gross_margin/size only)")
                continue
            for sd in [int(x) for x in args.seeds.split(",") if x.strip()]:
                row, dfa = run_scheduling(name, obj, cfg, args.time_limit,
                                          args.mip_gap, seed=sd)
                if row is None:
                    print(f"  {cfg:<11} skipped (optimal makespan not certified)")
                    continue
                rows.append(row)
                print(f"  {cfg:<11} seed={sd} status={row['status']} "
                      f"obj={fmt_optional(row['obj'])} nodes={row['n_nodes']} "
                      f"time={fmt_optional(row['time'], 's', precision=1)}")
        if rows:
            append_rows(rows, out, fresh=fresh)
            fresh = False
            rows = []


if __name__ == "__main__":
    main()
