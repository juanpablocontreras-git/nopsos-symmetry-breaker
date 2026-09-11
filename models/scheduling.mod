# =============================================================================
# NOPSOS scheduling core + DFA Symmetry Breaker
#
# Pure priority-slot scheduling problem used to benchmark the symmetry breaker
# in isolation (no operational layer).  Each operation runs exactly once
# (n_slots = |W|, AssignSlot = 1, SingleExec <= 1) on unary resources (the NO
# graph) with precedence (PREC) and fixed processing times.  Because every
# objective below depends only on the physical start times and durations, it is
# invariant to slot relabelling: the DFA-SBS and the bare model reach the same
# optimum, so the comparison measures only the effect of symmetry breaking.
#
# Selectable objectives (choose one with `objective <name>;` in the run script):
#   Makespan     minimise the schedule completion time,
#   WCT          minimise the total weighted completion time,
#   Feasibility  constant objective (feasibility / infeasibility probe; set H to
#                the deadline of interest).
#
# SBS on  : keep FlowAssign / FlowStart / FlowConserve.
# SBS off : drop those three and fix f := 0 (bare NOPSOS core).
# =============================================================================

set W ordered;
param n_slots integer > 0;
set T = 1..n_slots;
param H > 0;

set PREC within {W, W};
set NO   within {W, W};

param p{W} > 0;             # processing times
param w{W} default 1;      # completion-time weights (WCT objective)

# Valid lower bound on the makespan, supplied as data (0 = inactive).  Any set
# of operations that must run sequentially -- a non-overlapping clique (e.g. all
# operations on one machine) or a precedence chain (e.g. one job's routing) --
# forces Cmax >= sum of its processing times.  Since there is a single Cmax, all
# such group bounds collapse to their maximum, passed here as cmax_lb.  Big-M
# non-overlap constraints hide this structural bound from the LP relaxation, so
# stating it explicitly tightens the relaxation at negligible cost.
param cmax_lb default 0;

# Per-operation big-M for the non-overlapping constraint: a valid upper bound on
# the completion time of the operation.  Defaults to the horizon H (the loose
# bound); callers may supply the tighter H - tail[w], where tail[w] is the
# longest precedence chain of processing that must run after w (so w must finish
# by H - tail[w]).  A smaller coefficient tightens the LP relaxation of the
# O(|NO| n^2) non-overlapping rows without changing the feasible set.
param Mno{W} default H;

# DFA / SBS
set Q;
param q_initial integer >= 0;
set ARCS dimen 3;
param arc_label{(a,q,r) in ARCS} symbolic;

# Forbidden consecutive pairs.  The DFA of Algorithm 2 keeps one state per
# operation, so its language is 2-local and acceptance is equivalent to
# forbidding these pairs in consecutive slots (constraint LocalPair below).
# Empty by default, so the flow encoding is unaffected; the "pairwise" config
# fills it and drops FlowAssign/FlowStart/FlowConserve instead.
set FPAIRS within {W, W} default {};

var Z{T, W} binary;
var S{T, W} >= 0;
var D{T, W} >= 0;
# Flow variables of the Symmetry Breaker System.  Declared continuous in [0,1]
# rather than binary: the SBS is a unit path-flow through the layered DFA graph,
# whose constraint matrix (FlowStart/FlowConserve/FlowAssign) is totally
# unimodular, so once Z is integer the flow takes 0/1 values automatically.
# Keeping f binary only inflates the branch-and-bound (n_slots*|ARCS| extra
# integer variables) with no change to the feasible set or the optimum.
var f{T, ARCS} >= 0, <= 1;
var Cmax >= 0;

# (2) exactly one operation per slot
subject to AssignSlot{i in T}:
    sum{v in W} Z[i,v] = 1;

# (3) each operation in at most one slot (with n_slots = |W|, exactly one)
subject to SingleExec{v in W}:
    sum{i in T} Z[i,v] <= 1;

# (4) activation
subject to Activate{i in T, v in W}:
    S[i,v] + D[i,v] <= H * Z[i,v];

# fixed processing time when the op is scheduled in slot i
subject to DurFix{i in T, v in W}:
    D[i,v] = p[v] * Z[i,v];

# (5) precedence — slot ordering
subject to PrecSlot{i in T, w0 in W, v in W: (v,w0) in PREC}:
    (if i > 1 then sum{j in T: j < i} Z[j,v] else 0) >= Z[i,w0];

# (6) precedence — timing.  Aggregate form: valid here because this model is
# single-execution (SingleExec <= 1 with n_slots = |W|), so each sum has one
# non-zero term.  models/crude_oil.mod, where operations may repeat, uses the
# per-occurrence big-M form instead.
subject to PrecTime{w0 in W, v in W: (v,w0) in PREC}:
    sum{i in T} (S[i,v] + D[i,v]) <= sum{i in T} S[i,w0];

# (7) non-overlapping (unary resources); big-M = Mno[w0] (<= H) is a valid upper
# bound on w0's completion, so the row is slack exactly when Z[j,v] = 0.
subject to NoOverlap{i in T, j in T, w0 in W, v in W:
        (w0,v) in NO and i < j}:
    S[i,w0] + D[i,w0] <= S[j,v] + Mno[w0] * (1 - Z[j,v]);

# (8) flow-assignment
subject to FlowAssign{i in T, v in W}:
    sum{(a,q,r) in ARCS: arc_label[a,q,r] = v} f[i,a,q,r] = Z[i,v];

# (9) flow start
subject to FlowStart:
    sum{(a,q,r) in ARCS: q = q_initial} f[1,a,q,r] = 1;

# (10) flow conservation
subject to FlowConserve{i in T, q in Q: i > 1 and q <> q_initial}:
    sum{(a,s,r) in ARCS: r = q} f[i-1,a,s,r]
    =
    sum{(a,s,r) in ARCS: s = q} f[i,a,s,r];

# (8') pairwise encoding of the same language, used instead of (8)-(10)
subject to LocalPair{i in T, (w0,v) in FPAIRS: i < n_slots}:
    Z[i,w0] + Z[i+1,v] <= 1;

# completion time of every scheduled operation
subject to MakespanDef{i in T, v in W}:
    Cmax >= S[i,v] + D[i,v];

# structural makespan lower bound (machine-load / job-length cut); inactive when
# cmax_lb = 0, so it never affects models that do not supply it.
subject to CmaxLowerBound:
    Cmax >= cmax_lb;

# --- objectives (select one in the run script) ------------------------------
minimize Makespan:
    Cmax;

minimize WCT:
    sum{v in W} w[v] * sum{i in T} (S[i,v] + D[i,v]);

minimize Feasibility:
    0;
