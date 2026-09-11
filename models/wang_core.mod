# =============================================================================
# NOPSOS sequencing core with operation choice -- Wang et al. (2024) networks
#
# The networks of Wang, Yang, Dai et al. (2024), Comput. Chem. Eng. 187, 108714,
# describe *alternatives*: a tank may receive any of several parcels and may
# charge any of several CDUs.  The set W therefore holds candidate operations and
# only n = (one unloading per parcel) + (every charging run) of them execute.
#
# The NOPSOS core is unchanged: it reads the non-overlapping relation NO and
# (here empty) operation-level precedence.  What this family adds lives in the
# operational layer, as Section 3.2 of the paper prescribes:
#
#   ParcelOnce      exactly one candidate unloading per parcel is executed
#   ChargeOnce      every charging run is executed
#   GroupPrec*      the parcels of one vessel are unloaded in order, a relation
#                   between *groups* of candidates rather than between operations
#
# Group precedence cannot be written as the core's PrecSlot: that constraint
# forbids w when its predecessor v does not run, which is exactly what happens
# when another candidate of the same parcel is chosen instead.
# =============================================================================

set W ordered;
param n_slots integer > 0;
set T = 1..n_slots;
param H > 0;

set NO within {W, W};             # non-overlapping, both orientations
param p_dur{W} > 0;               # fixed processing time

set PARCEL;                       # parcels to be unloaded
set CAND{PARCEL} within W;        # candidate unloadings of each parcel
set CHARGE within W;              # charging runs, all of which execute
set PGROUP within {PARCEL, PARCEL};   # (p,q): every parcel p precedes q
param release{PARCEL} default 0;  # earliest arrival of the vessel carrying it

# DFA / SBS
set Q;
param q_initial integer >= 0;
set ARCS dimen 3;
param arc_label{(a,q,r) in ARCS} symbolic;
set FINAL within Q;

var Z{T, W} binary;
var S{T, W} >= 0;
var D{T, W} >= 0;
var f{T, ARCS} >= 0, <= 1;
var Cmax >= 0;

# --- NOPSOS core -------------------------------------------------------------
# Exactly one operation per slot: coverage below makes the count of executed
# operations equal to n_slots, so no slot is idle.
subject to AssignSlot{i in T}:
    sum{v in W} Z[i,v] = 1;

subject to SingleExec{v in W}:
    sum{i in T} Z[i,v] <= 1;

subject to Activate{i in T, v in W}:
    S[i,v] + D[i,v] <= H * Z[i,v];

subject to DurFix{i in T, v in W}:
    D[i,v] = p_dur[v] * Z[i,v];

subject to NoOverlap{i in T, j in T, w in W, v in W:
        (w,v) in NO and i < j}:
    S[i,w] + D[i,w] <= S[j,v] + H * (1 - Z[j,v]);

# --- operational layer of this family ----------------------------------------
subject to ParcelOnce{p in PARCEL}:
    sum{v in CAND[p], i in T} Z[i,v] = 1;

subject to VesselArrival{i in T, p in PARCEL, v in CAND[p]}:
    S[i,v] >= release[p] * Z[i,v];

subject to ChargeOnce{v in CHARGE}:
    sum{i in T} Z[i,v] = 1;

subject to GroupPrecSlot{i in T, (p,q) in PGROUP, w in CAND[q]}:
    (if i > 1 then sum{j in T, v in CAND[p]: j < i} Z[j,v] else 0) >= Z[i,w];

subject to GroupPrecTime{i in T, j in T, (p,q) in PGROUP,
                         v in CAND[p], w in CAND[q]}:
    S[i,v] + D[i,v] <= S[j,w] + H * (1 - Z[j,w]);

# --- Symmetry Breaker System, flow encoding ----------------------------------
subject to FlowAssign{i in T, v in W}:
    sum{(a,q,r) in ARCS: arc_label[a,q,r] = v} f[i,a,q,r] = Z[i,v];

subject to FlowStart:
    sum{(a,q,r) in ARCS: q = q_initial} f[1,a,q,r] = 1;

subject to FlowOnePerSlot{i in T}:
    sum{(a,q,r) in ARCS} f[i,a,q,r] = 1;

subject to FlowConserve{i in T, q in Q: i > 1 and q <> q_initial}:
    sum{(a,s,r) in ARCS: r = q} f[i-1,a,s,r]
    =
    sum{(a,s,r) in ARCS: s = q} f[i,a,s,r];

subject to FlowEnd:
    sum{(a,s,r) in ARCS: r in FINAL} f[n_slots,a,s,r] = 1;

# --- Symmetry Breaker System, pairwise encoding ------------------------------
set FPAIRS within {W, W} default {};
subject to LocalPair{i in T, (w0,v) in FPAIRS: i < n_slots}:
    Z[i,w0] + Z[i+1,v] <= 1;

# --- objective ---------------------------------------------------------------
subject to MakespanDef{i in T, v in W}:
    Cmax >= S[i,v] + D[i,v];

minimize Makespan:
    Cmax;

# Total completion time.  The makespan is pinned to H whenever the charging runs
# of one CDU tile the horizon, which they do by construction here, so this is the
# objective that discriminates between schedules.  Like the makespan it depends
# only on start times and durations, so it is invariant under the slot symmetry.
minimize TCT:
    sum{i in T, v in W} (S[i,v] + D[i,v]);
