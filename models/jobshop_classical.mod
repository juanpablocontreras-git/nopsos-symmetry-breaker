# =============================================================================
# Classical disjunctive job-shop MILP (Manne, 1960) -- Gurobi baseline
#
# The standard compact job-shop formulation, included as an external reference
# point for the NOPSOS priority-slot model.  It has NO priority slots and hence
# NO slot-permutation symmetry: each operation has a single continuous start
# time, and each pair of operations sharing a machine is sequenced by one binary
# ordering variable.  This is the model Gurobi is built to solve well, so it
# shows where the priority-slot formulation (with or without the DFA-SBS) stands
# against the best-known compact representation.
#
# It is deliberately a *different formulation*, not a symmetry-breaker toggle:
# the DFA-vs-no-SBS comparison lives on models/scheduling.mod (same model, SBS
# on/off); this file is the "textbook alternative" column.
#
# Selectable objectives (choose one with `objective <name>;` in the run script):
#   Makespan  minimise the schedule completion time,
#   WCT       minimise the total weighted completion time.
# =============================================================================

set OPS ordered;
param p{OPS} > 0;                 # processing times
param w{OPS} default 0;           # completion-time weights (WCT objective)
param M > 0;                      # big-M (a valid horizon, e.g. sum of p)

set PREC  within {OPS, OPS};      # routing precedence (v before w in its job)
set MPAIR within {OPS, OPS};      # unordered pairs sharing a machine (o listed first)

var s{OPS} >= 0;                  # operation start time
var x{MPAIR} binary;             # 1 iff the first op of the pair precedes the second
var Cmax >= 0;

# routing precedence within a job
subject to Precedence{(v,ww) in PREC}:
    s[ww] >= s[v] + p[v];

# machine disjunction: the two operations sharing a machine cannot overlap.
# x = 1 forces o before o2; x = 0 forces o2 before o.
subject to Disj1{(o,o2) in MPAIR}:
    s[o2] >= s[o] + p[o] - M * (1 - x[o,o2]);
subject to Disj2{(o,o2) in MPAIR}:
    s[o] >= s[o2] + p[o2] - M * x[o,o2];

# makespan definition
subject to MakespanDef{o in OPS}:
    Cmax >= s[o] + p[o];

# --- objectives (select one in the run script) ------------------------------
minimize Makespan:
    Cmax;

minimize WCT:
    sum{o in OPS} w[o] * (s[o] + p[o]);
