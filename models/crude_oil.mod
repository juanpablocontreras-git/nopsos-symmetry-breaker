# =============================================================================
# Crude-oil scheduling: priority-slot core + DFA-SBS + operational layer
#
# Current experiment procedure:
#   MILP step : solve this file with constraint CompositionBilinear DROPPED.
#               The nonlinear constraint remains here for optional later use.
#
# In the run scripts:
#   drop CompositionBilinear;     # before MILP solve
#   restore CompositionBilinear;  # optional later nonlinear solve
# =============================================================================

# =============================================================================
# PART 1 — PRIORITY-SLOT CORE + SBS
# =============================================================================

set W ordered;           # operations
param n_slots integer > 0;
set T = 1..n_slots;
param H > 0;

set PREC within {W, W};  # precedence arcs (v,w): v must finish before w
set NO   within {W, W};  # non-overlapping, directed, both orientations

# DFA / SBS
set Q;
param q_initial integer >= 0;
set ARCS dimen 3;        # (arc_id, from_state, to_state)
# Arc labels are operation names in W, except idle-sink arcs which carry a
# reserved label not in W (so FlowAssign, which matches arc_label = v for v in
# W, never counts them).  Hence the domain is unrestricted symbolic.
param arc_label{(a,q,r) in ARCS} symbolic;
# Accepting states.  The slot-n state must be final (FlowEnd below).  For the
# NOPSOS DFA every state is final, so the writer lists all of Q and FlowEnd is
# vacuous; for SOS+RE only the regular-language accept states (and the idle
# sink) are listed, which forces a schedule to complete a valid DP word.
set FINAL within Q;

# Core variables
var Z{T, W} binary;
var S{T, W} >= 0;
var D{T, W} >= 0;

# SBS flow variables
var f{T, ARCS} binary;

# (2) At most one operation per slot.  Relaxed from "= 1" to "<= 1": when the
# DFA carries an idle sink (allow_idle), surplus slots take the idle arc (whose
# label is not in W) and assign no operation, so n_slots is an upper bound on
# the schedule length.  With an idle-free DFA, flow conservation still forces a
# full unit through every slot, recovering the original "= 1" behaviour.
subject to AssignSlot{i in T}:
    sum{v in W} Z[i,v] <= 1;

# (3) Execution cap: operation v may occupy at most eta[v] slots.  eta[v] = 1 is
# the single-execution convention (each op runs at most once, as in job-shop);
# eta[v] = n_slots imposes no cap so v may recur across slots (crude-oil), with
# admissible repetitions governed by the DFA.  Intermediate values give a
# per-operation re-execution budget.  Written by write_crude_oil_dat.
param eta{W} integer >= 1, <= n_slots, default n_slots;
subject to MaxExec{v in W}:
    sum{i in T} Z[i,v] <= eta[v];

# (4) Activation
subject to Activate{i in T, v in W}:
    S[i,v] + D[i,v] <= H * Z[i,v];

# (5) Precedence — slot ordering
subject to PrecSlot{i in T, w in W, v in W: (v,w) in PREC}:
    (if i > 1 then sum{j in T: j < i} Z[j,v] else 0) >= Z[i,w];

# (6) Precedence — timing, per occurrence.  Every execution of v must finish
# before every execution of w starts.  When Z[j,w]=0 the row is slack, and when
# Z[i,v]=0 the left side vanishes through Activate, so only real executions are
# constrained.  The aggregate form sum_i (S+D) <= sum_i S is equivalent to this
# one when v and w each run once (eta=1), and is neither valid nor sufficient
# otherwise, which matters here because crude-oil operations are uncapped.
subject to PrecTime{i in T, j in T, w in W, v in W: (v,w) in PREC}:
    S[i,v] + D[i,v] <= S[j,w] + H * (1 - Z[j,w]);

# (7) Non-overlapping
subject to NoOverlap{i in T, j in T, w in W, v in W:
        (w,v) in NO and i < j}:
    S[i,w] + D[i,w] <= S[j,v] + H * (1 - Z[j,v]);

# (8) Flow-assignment
subject to FlowAssign{i in T, v in W}:
    sum{(a,q,r) in ARCS: arc_label[a,q,r] = v} f[i,a,q,r] = Z[i,v];

# (9) Flow start
subject to FlowStart:
    sum{(a,q,r) in ARCS: q = q_initial} f[1,a,q,r] = 1;

# (9b) Exactly one arc (one token) per slot.  Without this, flow conservation
# (which excludes q_initial) lets the solver inject extra flow units that leave
# q_initial at slots > 1, so the "flow" is no longer a single DFA path and the
# SBS is unsound (e.g. odd charge counts on an even-only automaton).
subject to FlowOnePerSlot{i in T}:
    sum{(a,q,r) in ARCS} f[i,a,q,r] = 1;

# (8') Pairwise encoding of the same language.  The DFA of Algorithm 2 keeps
# one state per operation, so its language is 2-local: a word is accepted iff
# no consecutive pair is forbidden and, once a slot is idle, every later slot
# is idle too (the idle sink is absorbing).  Both are pairwise conditions on Z,
# so the flow layer can be replaced by LocalPair + IdleTail.  Empty/0 by
# default; the "pairwise" config fills them and drops (8)-(11).
set FPAIRS within {W, W} default {};
param use_pairwise default 0;

subject to LocalPair{i in T, (w0,v) in FPAIRS: i < n_slots}:
    Z[i,w0] + Z[i+1,v] <= 1;

subject to IdleTail{i in T: i < n_slots and use_pairwise > 0}:
    sum{v in W} Z[i+1,v] <= sum{v in W} Z[i,v];

# (10) Flow conservation
subject to FlowConserve{i in T, q in Q: i > 1 and q <> q_initial}:
    sum{(a,s,r) in ARCS: r = q} f[i-1,a,s,r]
    =
    sum{(a,s,r) in ARCS: s = q} f[i,a,s,r];

# (11) Terminal: the state after the last slot must be accepting.  Vacuous for
# the NOPSOS DFA (FINAL = Q); for SOS+RE it forces completion of a valid word.
subject to FlowEnd:
    sum{(a,s,r) in ARCS: r in FINAL} f[n_slots,a,s,r] = 1;


# =============================================================================
# PART 2 — OPERATIONAL LAYER
# =============================================================================

# --- Resource sets ---
set R_V;                          # vessels
set R_S;                          # storage tanks
set R_C;                          # charging tanks
set R_D;                          # crude distillation units (CDUs)
set R = R_V union R_S union R_C union R_D;

# --- Operation sub-sets ---
set W_U within W;                 # unloading  (vessel -> storage)
set W_T within W;                 # transfer   (storage -> charging)
set W_D within W;                 # charging   (charging -> CDU)

# --- Crude types and properties ---
set C;                            # crude types
set K;                            # tracked properties (e.g. sulphur)

# --- Operation-resource connectivity ---
param op_inlet {W}  symbolic in R;   # resource operation draws FROM
param op_outlet{W}  symbolic in R;   # resource operation deposits INTO

# Derived: inlet ops (feed into r) and outlet ops (draw from r)
set I_r{r in R} = {v in W : op_outlet[v] = r};
set O_r{r in R} = {v in W : op_inlet [v] = r};

# --- Operational parameters ---
param s_r     {R_V};                   # vessel arrival time
param FR_lb   {W}   default 0;         # flowrate lower bound
param FR_ub   {W};                     # flowrate upper bound
param Vt_lb   {W}   default 0;         # total-volume lower bound per op
param Vt_ub   {W};                     # total-volume upper bound per op
param Lt_lb   {R_S union R_C} default 0;
param Lt_ub   {R_S union R_C};
param D_lb    {R_C} default 0;         # demand lower bound (charging tank)
param D_ub    {R_C};                   # demand upper bound
param ND_lb           default 0;       # min number of distillation ops
param ND_ub;                           # max number of distillation ops
param x_lb    {W, K} default -1e30;    # property lower bound on transfer
param x_ub    {W, K} default  1e30;    # property upper bound on transfer
param x_prop  {C, K};                  # property value of crude c
param G_crude {C};                     # gross margin of crude c  ($/volume)
param Lt0     {R_S union R_C};         # initial total tank level
param L0      {R_S union R_C, C} default 0;  # initial per-crude level

# --- Operational variables (all continuous, non-negative) ---
var Vt {T, W}                >= 0;    # total volume transferred
var Vc {T, W, C}             >= 0;    # per-crude volume transferred
var Lt {T, R_S union R_C}    >= 0;    # total tank level before slot i
var Lc {T, R_S union R_C, C} >= 0;   # per-crude level before slot i

# =============================================================================
# TRANSFER VOLUME CONSTRAINTS  (A.1 – A.4)
# =============================================================================

# (A.1) Volume upper bound
subject to VolUB{i in T, v in W}:
    Vt[i,v] <= Vt_ub[v] * Z[i,v];

# (A.2) Volume lower bound
subject to VolLB{i in T, v in W}:
    Vt[i,v] >= Vt_lb[v] * Z[i,v];

# (A.3) Volume decomposition by crude
subject to VolDecomp{i in T, v in W}:
    sum{c in C} Vc[i,v,c] = Vt[i,v];

# (A.4) Flowrate bounds (Vt = flowrate * duration)
subject to FlowrateLB{i in T, v in W}:
    FR_lb[v] * D[i,v] <= Vt[i,v];
subject to FlowrateUB{i in T, v in W}:
    Vt[i,v] <= FR_ub[v] * D[i,v];

# =============================================================================
# MATERIAL BALANCE AND TANK LEVELS  (A.5 – A.9)
# =============================================================================

# Initial level (before slot 1)
subject to LevelTotalInit{r in R_S union R_C}:
    Lt[1,r] = Lt0[r];
subject to LevelCrudeInit{r in R_S union R_C, c in C}:
    Lc[1,r,c] = L0[r,c];

# Level update: level before slot i+1 = level before slot i + net inflow at i
subject to LevelTotalUpdate{i in T, r in R_S union R_C: i < n_slots}:
    Lt[i+1,r] = Lt[i,r]
        + sum{v in I_r[r]} Vt[i,v]
        - sum{v in O_r[r]} Vt[i,v];

subject to LevelCrudeUpdate{i in T, r in R_S union R_C, c in C: i < n_slots}:
    Lc[i+1,r,c] = Lc[i,r,c]
        + sum{v in I_r[r]} Vc[i,v,c]
        - sum{v in O_r[r]} Vc[i,v,c];

# (A.7) Tank capacity during horizon
subject to CapDuringLB{i in T, r in R_S union R_C}:
    Lt[i,r] >= Lt_lb[r];
subject to CapDuringUB{i in T, r in R_S union R_C}:
    Lt[i,r] <= Lt_ub[r];

# (A.9) Tank capacity at end of horizon (after all operations)
subject to CapFinalLB{r in R_S union R_C}:
    Lt0[r]
    + sum{i in T} (sum{v in I_r[r]} Vt[i,v] - sum{v in O_r[r]} Vt[i,v])
    >= Lt_lb[r];
subject to CapFinalUB{r in R_S union R_C}:
    Lt0[r]
    + sum{i in T} (sum{v in I_r[r]} Vt[i,v] - sum{v in O_r[r]} Vt[i,v])
    <= Lt_ub[r];

# =============================================================================
# RESOURCE AVAILABILITY  (A.10)
# =============================================================================

# Vessel arrival: unloading cannot start before the vessel arrives
subject to VesselArrival{i in T, r in R_V, v in O_r[r]}:
    S[i,v] >= s_r[r] * Z[i,v];

# =============================================================================
# DISTILLATION AND DEMAND  (A.11 – A.12)
# =============================================================================

subject to DistCountLB:
    sum{i in T, v in W_D} Z[i,v] >= ND_lb;
subject to DistCountUB:
    sum{i in T, v in W_D} Z[i,v] <= ND_ub;

subject to DemandLB{r in R_C}:
    sum{i in T, v in O_r[r]} Vt[i,v] >= D_lb[r];
subject to DemandUB{r in R_C}:
    sum{i in T, v in O_r[r]} Vt[i,v] <= D_ub[r];

# =============================================================================
# PROPERTY CONSTRAINTS  (A.13)
# =============================================================================

subject to PropertyLB{i in T, v in W, k in K: x_lb[v,k] > -1e29}:
    sum{c in C} x_prop[c,k] * Vc[i,v,c] >= x_lb[v,k] * Vt[i,v];
subject to PropertyUB{i in T, v in W, k in K: x_ub[v,k] < 1e29}:
    sum{c in C} x_prop[c,k] * Vc[i,v,c] <= x_ub[v,k] * Vt[i,v];

# =============================================================================
# COMPOSITION CONSTRAINT — BILINEAR  (A.14)
# DROP for MILP step; RESTORE for NLP step.
# =============================================================================
subject to CompositionBilinear
    {i in T, r in R_S union R_C, v in O_r[r], c in C}:
    Vc[i,v,c] * Lt[i,r] = Lc[i,r,c] * Vt[i,v];

# =============================================================================
# OBJECTIVE  (A.15)
# Maximise total gross margin of crude processed by CDUs.
# =============================================================================
maximize GrossMargin:
    sum{i in T, r in R_D, v in I_r[r], c in C} G_crude[c] * Vc[i,v,c];
