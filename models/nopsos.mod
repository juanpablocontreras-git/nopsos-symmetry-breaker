# =============================================================================
# NOPSOS formulation with DFA-based Symmetry Breaker System
#
# Core decision variables
#   Z[i,v]  binary  1 iff operation v is assigned to slot i
#   S[i,v]  >= 0    start time of v in slot i  (0 when Z[i,v]=0)
#   D[i,v]  >= 0    duration of v in slot i    (0 when Z[i,v]=0)
#
# SBS flow variables
#   f[i,(a,q,r)]  binary  1 iff DFA arc a=(q,r) is traversed at slot-step i
#
# The operational layer (objective + application constraints) is included
# from a separate file so this core file remains reusable.
# =============================================================================

# -----------------------------------------------------------------------------
# SETS AND PARAMETERS — NOPSOS core
# -----------------------------------------------------------------------------
set W ordered;                      # operations
param n_slots integer > 0;          # number of priority slots n
set T = 1..n_slots;                 # slot indices {1,...,n}
param H > 0;                        # time horizon length

set PREC within {W, W};             # precedence: (v,w) => v must precede w
set NO   within {W, W};             # non-overlapping (directed; include both
                                    # (v,w) and (w,v) for each undirected edge)

# -----------------------------------------------------------------------------
# SETS AND PARAMETERS — DFA Symmetry Breaker System
# -----------------------------------------------------------------------------
set Q;                              # DFA states (integers; 0 = initial state q0)
param q_initial integer >= 0;       # initial state (always 0)
set ARCS dimen 3;                   # DFA arcs (arc_id, from_state, to_state)
param arc_label{(a,q,r) in ARCS} symbolic;  # operation label of arc (a,q,r)

# -----------------------------------------------------------------------------
# DECISION VARIABLES — NOPSOS core
# -----------------------------------------------------------------------------
var Z{T, W} binary;
var S{T, W} >= 0;
var D{T, W} >= 0;

# -----------------------------------------------------------------------------
# DECISION VARIABLES — SBS arc flow
# -----------------------------------------------------------------------------
var f{T, ARCS} binary;

# -----------------------------------------------------------------------------
# NOPSOS CORE CONSTRAINTS
# -----------------------------------------------------------------------------

# (2) Exactly one operation per slot
subject to AssignSlot{i in T}:
    sum{v in W} Z[i,v] = 1;

# (3) Each operation occupies at most one slot
#     (equality holds when |W| = n_slots, i.e. every op is scheduled)
subject to SingleExec{v in W}:
    sum{i in T} Z[i,v] <= 1;

# (4) Activation: S and D are zero when the slot is unassigned
subject to Activate{i in T, v in W}:
    S[i,v] + D[i,v] <= H * Z[i,v];

# (5) Precedence — slot ordering: w must be in a strictly later slot than v
#     sum_{j<i} Z[j,v] >= Z[i,w]  for each predecessor v of w
subject to PrecSlot{i in T, w in W, v in W: (v,w) in PREC}:
    (if i > 1 then sum{j in T: j < i} Z[j,v] else 0) >= Z[i,w];

# (6) Precedence — timing: v finishes before w starts
subject to PrecTime{w in W, v in W: (v,w) in PREC}:
    sum{i in T} (S[i,v] + D[i,v]) <= sum{i in T} S[i,w];

# (7) Non-overlapping: if w is in slot i and v is in a later slot j,
#     w must finish before v starts
subject to NoOverlap{i in T, j in T, w in W, v in W:
        (w,v) in NO and i < j}:
    S[i,w] + D[i,w] <= S[j,v] + H * (1 - Z[j,v]);

# -----------------------------------------------------------------------------
# SYMMETRY BREAKER SYSTEM
# -----------------------------------------------------------------------------

# (8) Flow-assignment: the arc traversed at slot i must carry the label of
#     the operation assigned to that slot
subject to FlowAssign{i in T, v in W}:
    sum{(a,q,r) in ARCS: arc_label[a,q,r] = v} f[i,a,q,r] = Z[i,v];

# (9) Flow start: exactly one arc leaves the initial state at slot 1
subject to FlowStart:
    sum{(a,q,r) in ARCS: q = q_initial} f[1,a,q,r] = 1;

# (10) Flow conservation: state entered after slot i-1 is the state
#      departed from at slot i  (holds for all non-initial states, i >= 2)
subject to FlowConserve{i in T, q in Q: i > 1 and q <> q_initial}:
    sum{(a,s,r) in ARCS: r = q} f[i-1,a,s,r]
    =
    sum{(a,s,r) in ARCS: s = q} f[i,a,s,r];

# -----------------------------------------------------------------------------
# NOTE: No objective is declared here.
# The application model (e.g. crude_oil_full.mod) provides the objective.
# This file can be loaded standalone for testing the NOPSOS core + SBS only.
# -----------------------------------------------------------------------------
