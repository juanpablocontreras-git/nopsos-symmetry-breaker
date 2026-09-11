# -*- coding: utf-8 -*-
"""Solve the Wang et al. (2024) networks on the NOPSOS sequencing core.

    python run_wang.py --config dfa pairwise nosbs --time-limit 600

The two relations are read off the network exactly as for crude oil: every
resource -- parcel, tank, CDU -- is unary, and the single berth makes any two
unloadings non-overlapping.  Operation-level precedence is empty; the order of
the parcels of a vessel is a group relation and lives in the operational layer
(models/wang_core.mod).
"""
import argparse, csv, json, os
from pathlib import Path

from src.dfa_builder import build_dfa, forbidden_pairs, IDLE_LABEL

HERE = Path(__file__).parent
DAT_DIR = HERE / "instances"
RESULTS = HERE / "results"
MOD = str(HERE / "models" / "wang_core.mod")
INSTANCES = ["wang_ex%d" % i for i in range(1, 7)]


def load(name):
    return json.load(open(DAT_DIR / (name + ".json")))


def graphs(inst):
    """Non-overlapping edges: shared resource, plus the single berth."""
    W = inst["W_unload"] + inst["W_charge"]
    no = set()

    def add(a, b):
        if a != b:
            no.add(frozenset((a, b)))

    for i, a in enumerate(inst["W_unload"]):       # single berth
        for b in inst["W_unload"][i + 1:]:
            add(a, b)
    for key in ("op_parcel", "op_tank", "op_cdu"):  # unary resources
        by_res = {}
        for w, r in inst[key].items():
            by_res.setdefault(r, []).append(w)
        for ops in by_res.values():
            for i, a in enumerate(ops):
                for b in ops[i + 1:]:
                    add(a, b)
    return W, [tuple(e) for e in no]


def write_dat(path, inst, dfa, fpairs=None):
    W, no = graphs(inst)
    q_final = dfa.final()
    L = []
    L.append("set W := " + " ".join(W) + " ;")
    L.append("param n_slots := %d ;" % inst["n_slots"])
    L.append("param H := %g ;" % inst["H"])
    L.append("set NO :=")
    for a, b in no:
        L.append("  %s %s" % (a, b))
        L.append("  %s %s" % (b, a))
    L.append(";")
    L.append("param p_dur :=")
    for w in W:
        L.append("  %s %g" % (w, inst["duration"][w]))
    L.append(";")
    L.append("set PARCEL := " + " ".join(inst["parcels"]) + " ;")
    for p in inst["parcels"]:
        cand = [w for w, pp in inst["op_parcel"].items() if pp == p]
        L.append("set CAND[%s] := %s ;" % (p, " ".join(cand)))
    L.append("param release :=")
    for p in inst["parcels"]:
        L.append("  %s %g" % (p, inst["release"][p]))
    L.append(";")
    L.append("set CHARGE := " + " ".join(inst["W_charge"]) + " ;")
    L.append("set PGROUP :=")
    order = inst["vlcc_of_parcel"]
    ps = inst["parcels"]
    for i, p in enumerate(ps):                     # parcels are listed in order
        for q in ps[i + 1:]:
            L.append("  %s %s" % (p, q))           # p before q (same or later VLCC)
    L.append(";")
    L.append("set Q := " + " ".join(str(q) for q in dfa.states) + " ;")
    L.append("param q_initial := %d ;" % dfa.initial)
    L.append("set FINAL := " + " ".join(str(q) for q in q_final) + " ;")
    L.append("set ARCS :=")
    for a, (fr, to, lab) in enumerate(dfa.arcs, start=1):
        L.append("  %d %d %d" % (a, fr, to))
    L.append(";")
    L.append("param arc_label :=")
    for a, (fr, to, lab) in enumerate(dfa.arcs, start=1):
        L.append("  %d %d %d %s" % (a, fr, to, lab))
    L.append(";")
    if fpairs:
        L.append("set FPAIRS :=")
        for a, b in fpairs:
            L.append("  %s %s" % (a, b))
        L.append(";")
    open(path, "w").write("\n".join(L) + "\n")


def solve(name, config, time_limit, mip_gap, seed=0, objective="TCT"):
    from amplpy import AMPL
    inst = load(name)
    W, no = graphs(inst)
    dfa = build_dfa(W, [], no, allow_idle=False)
    fpairs = forbidden_pairs(dfa) if config == "pairwise" else None
    dat = str(DAT_DIR / ("%s_%s.dat" % (name, config)))
    write_dat(dat, inst, dfa, fpairs)

    log = str(RESULTS / ("%s_%s_wang.log" % (name, config)))
    if os.path.exists(log):
        os.remove(log)
    a = AMPL()
    a.read(MOD)
    a.read_data(dat)
    if config in ("pairwise", "nosbs"):
        a.eval("drop FlowAssign; drop FlowStart; drop FlowOnePerSlot; "
               "drop FlowConserve; drop FlowEnd;")
        a.eval("fix {i in T,(aa,q,r) in ARCS} f[i,aa,q,r] := 0;")
    a.eval("objective %s;" % objective)
    a.set_option("solver", "gurobi")
    a.set_option("gurobi_options",
                 "timelimit=%d mipgap=%g threads=1 symmetry=0 dualreductions=0 "
                 "seed=%d logfile='%s'" % (time_limit, mip_gap, seed,
                                           log.replace("\\", "/")))
    a.solve()
    obj = None
    try:
        obj = a.get_objective(objective).value()
    except Exception:
        pass
    res = {"instance": name, "config": config,
           "n_ops": len(W), "n_slots": inst["n_slots"],
           "n_dfa_states": dfa.n_states, "n_dfa_arcs": dfa.n_arcs,
           "obj": obj, "status": a.get_value("solve_result"),
           "time": a.get_value("_solve_time")}
    a.close()
    from src.ampl_interface import parse_gurobi_log
    gi = parse_gurobi_log(log)
    res["n_nodes"] = gi.get("n_nodes")
    res["gap_pct"] = 100.0 * gi["gap"] if gi.get("gap") is not None else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["dfa", "pairwise", "nosbs"])
    ap.add_argument("--instances", default=",".join(INSTANCES))
    ap.add_argument("--objective", choices=["TCT", "Makespan"], default="TCT")
    ap.add_argument("--time-limit", type=int, default=600)
    ap.add_argument("--mip-gap", type=float, default=1e-4)
    ap.add_argument("--out", default=str(RESULTS / "wang_core.csv"))
    args = ap.parse_args()

    rows, fresh = [], True
    for name in [s.strip() for s in args.instances.split(",") if s.strip()]:
        print("\n%s" % name)
        for cfg in args.config:
            r = solve(name, cfg, args.time_limit, args.mip_gap,
                      objective=args.objective)
            rows.append(r)
            print("  %-9s status=%-8s obj=%-8s nodes=%-8s time=%.1fs  dfa=%d/%d"
                  % (cfg, r["status"], r["obj"], r["n_nodes"], r["time"] or 0,
                     r["n_dfa_states"], r["n_dfa_arcs"]))
        mode = "w" if fresh else "a"
        with open(args.out, mode, newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if fresh:
                w.writeheader()
            w.writerows(rows)
        fresh, rows = False, []
    print("\n[CSV] %s" % args.out)


if __name__ == "__main__":
    main()
