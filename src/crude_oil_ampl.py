"""
Crude-oil instance dataclass (with operational layer) and AMPL data writer.

Covers all parameters of the operational layer.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .dfa_builder import DFA


# ---------------------------------------------------------------------------
# Complete crude-oil instance (NOPSOS core + operational layer)
# ---------------------------------------------------------------------------

@dataclass
class CrudeOilFullInstance:
    """
    All data for one crude-oil scheduling instance.

    Notation follows the operational-layer model.

    Resource sets
    -------------
    R_V : vessels
    R_S : storage tanks
    R_C : charging tanks
    R_D : CDUs

    Operation sets
    --------------
    W_U : unloading  (vessel   -> storage)
    W_T : transfers  (storage  -> charging)
    W_D : charging   (charging -> CDU)

    Connectivity
    ------------
    op_inlet[w]  : resource w draws from
    op_outlet[w] : resource w deposits into

    Parameters
    ----------
    H        : time horizon
    s_r      : dict vessel -> arrival time
    FR_lb/ub : dict op -> flowrate bounds
    Vt_lb/ub : dict op -> total-volume bounds
    Lt_lb/ub : dict tank -> capacity bounds
    D_lb/ub  : dict charging_tank -> demand bounds
    ND_lb/ub : int bounds on number of distillation ops
    x_lb/ub  : dict (op,prop) -> property bounds on transferred material
    x_prop   : dict (crude,prop) -> property value
    G_crude  : dict crude -> gross margin
    Lt0      : dict tank -> initial total level
    L0       : dict (tank,crude) -> initial per-crude level

    DFA
    ---
    dfa      : DFA object for SBS
    """
    # --- Topology ---
    R_V: List[str]
    R_S: List[str]
    R_C: List[str]
    R_D: List[str]

    W_U: List[str]
    W_T: List[str]
    W_D: List[str]

    op_inlet:  Dict[str, str]   # op -> source resource
    op_outlet: Dict[str, str]   # op -> destination resource

    # --- Crude types and properties ---
    C: List[str]   # crude types
    K: List[str]   # tracked properties

    # --- Timing ---
    H: float
    n_slots: int
    s_r: Dict[str, float]   # vessel arrival times

    # --- Flowrate and volume bounds ---
    FR_lb: Dict[str, float]
    FR_ub: Dict[str, float]
    Vt_lb: Dict[str, float]
    Vt_ub: Dict[str, float]

    # --- Tank capacity ---
    Lt_lb: Dict[str, float]
    Lt_ub: Dict[str, float]
    Lt0:   Dict[str, float]   # initial total level

    # --- Demand (charging tanks) ---
    D_lb: Dict[str, float]
    D_ub: Dict[str, float]

    # --- Distillation count ---
    ND_lb: int
    ND_ub: int

    # --- Property bounds on transfers ---
    # keys are (op, prop); missing entries mean unbounded
    x_lb: Dict[Tuple[str, str], float] = field(default_factory=dict)
    x_ub: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # --- Crude properties and gross margins ---
    x_prop:  Dict[Tuple[str, str], float] = field(default_factory=dict)
    G_crude: Dict[str, float]             = field(default_factory=dict)

    # --- Initial per-crude tank levels ---
    L0: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # --- DFA (set by the experiment runner after calling build_dfa) ---
    dfa: Optional[DFA] = field(default=None, repr=False)

    @property
    def W(self) -> List[str]:
        return self.W_U + self.W_T + self.W_D

    @property
    def R(self) -> List[str]:
        return self.R_V + self.R_S + self.R_C + self.R_D

    def inlet_ops(self, r: str) -> List[str]:
        """Operations that deposit INTO resource r (op_outlet[v] == r)."""
        return [v for v in self.W if self.op_outlet.get(v) == r]

    def outlet_ops(self, r: str) -> List[str]:
        """Operations that draw FROM resource r (op_inlet[v] == r)."""
        return [v for v in self.W if self.op_inlet.get(v) == r]


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------

def load_instance_json(path: str) -> CrudeOilFullInstance:
    """
    Load a CrudeOilFullInstance from a JSON file.

    Expected JSON structure
    -----------------------
    {
      "R_V": [...], "R_S": [...], "R_C": [...], "R_D": [...],
      "W_U": [...], "W_T": [...], "W_D": [...],
      "op_inlet":  {"w1": "vessel1", ...},
      "op_outlet": {"w1": "tank1",   ...},
      "C": [...], "K": [...],
      "H": 48.0, "n_slots": 20,
      "s_r":  {"vessel1": 0.0, ...},
      "FR_lb": {"w1": 100.0, ...}, "FR_ub": {...},
      "Vt_lb": {"w1": 0.0,   ...}, "Vt_ub": {...},
      "Lt_lb": {"tank1": 100.0, ...}, "Lt_ub": {...}, "Lt0": {...},
      "D_lb":  {"ct1": 500.0, ...},  "D_ub": {...},
      "ND_lb": 3, "ND_ub": 9,
      "x_lb": {"w1__sulfur": 0.0, ...},  (key format: "op__prop")
      "x_ub": {...},
      "x_prop": {"crude1__sulfur": 0.03, ...},
      "G_crude": {"crude1": 22.5, ...},
      "L0": {"tank1__crude1": 50.0, ...}
    }
    Pair keys use double underscore as separator.
    """
    with open(path) as fh:
        d = json.load(fh)

    def parse_pair_dict(raw: dict) -> dict:
        """Convert "a__b" keys to (a, b) tuple keys."""
        return {tuple(k.split("__", 1)): v for k, v in raw.items()}

    return CrudeOilFullInstance(
        R_V=d["R_V"], R_S=d["R_S"], R_C=d["R_C"], R_D=d["R_D"],
        W_U=d["W_U"], W_T=d["W_T"], W_D=d["W_D"],
        op_inlet=d["op_inlet"],
        op_outlet=d["op_outlet"],
        C=d["C"], K=d["K"],
        H=d["H"], n_slots=d["n_slots"],
        s_r=d["s_r"],
        FR_lb=d["FR_lb"], FR_ub=d["FR_ub"],
        Vt_lb=d["Vt_lb"], Vt_ub=d["Vt_ub"],
        Lt_lb=d["Lt_lb"], Lt_ub=d["Lt_ub"], Lt0=d["Lt0"],
        D_lb=d["D_lb"],   D_ub=d["D_ub"],
        ND_lb=d["ND_lb"], ND_ub=d["ND_ub"],
        x_lb=parse_pair_dict(d.get("x_lb", {})),
        x_ub=parse_pair_dict(d.get("x_ub", {})),
        x_prop=parse_pair_dict(d.get("x_prop", {})),
        G_crude=d.get("G_crude", {}),
        L0=parse_pair_dict(d.get("L0", {})),
    )



# ---------------------------------------------------------------------------
# AMPL .dat writer for full crude-oil model
# ---------------------------------------------------------------------------

def write_crude_oil_dat(dat_path: str, inst: CrudeOilFullInstance):
    """
    Write a complete AMPL .dat file for crude_oil_full.mod.

    The DFA must already be set (inst.dfa is not None).

    Parameters
    ----------
    dat_path : str
        Output path.
    inst : CrudeOilFullInstance
        Instance with dfa field populated.
    """
    if inst.dfa is None:
        raise ValueError("inst.dfa must be set before writing the .dat file.")

    dfa = inst.dfa
    W   = inst.W

    # Build directed NO edges from the DFA construction
    # (these were used to build the DFA; we need them for the AMPL model too)
    # The caller is responsible for passing them; here we derive them from
    # the DFA's backward arcs (Rule 3) as a minimal set.
    # For the full crude-oil model the NO set is computed by build_crude_oil_dfa
    # and passed in via inst.no_edges_directed (set by caller).
    no_directed = getattr(inst, '_no_edges_directed', [])

    # Build precedence edges from the DFA's operation order + prec_edges attribute
    prec_edges = getattr(inst, '_prec_edges', [])

    lines = [
        "# Auto-generated by crude_oil_ampl.write_crude_oil_dat",
        "",
        "# =============================================================",
        "# NOPSOS core data",
        "# =============================================================",
        f"set W := {' '.join(W)};",
        f"param n_slots := {inst.n_slots};",
        f"param H := {inst.H};",
        "",
    ]

    # --- PREC ---
    _write_set_pairs(lines, "PREC", prec_edges)
    lines.append("")

    # --- NO (directed, both orientations) ---
    _write_set_pairs(lines, "NO", no_directed)
    lines.append("")

    # --- DFA ---
    lines += [
        "# DFA states",
        f"set Q := {' '.join(str(s) for s in dfa.states)};",
        f"param q_initial := {dfa.initial};",
        "",
    ]
    # Pairwise (locality) encoding data, written only for the pairwise config.
    fpairs = getattr(inst, "_fpairs", None)
    if fpairs:
        lines.append("param use_pairwise := 1;")
        lines.append("set FPAIRS :=")
        for w, v in fpairs:
            lines.append(f"  {w} {v}")
        lines.append(";")

    if dfa.arcs:
        lines.append("set ARCS :=")
        for a, (f, t, lab) in enumerate(dfa.arcs, start=1):
            lines.append(f"  ({a}, {f}, {t})")
        lines.append(";")
        lines.append("")
        lines.append("param arc_label :=")
        for a, (f, t, lab) in enumerate(dfa.arcs, start=1):
            lines.append(f"  [{a},{f},{t}] '{lab}'")
        lines.append(";")
    else:
        lines.append("set ARCS := ;")
    lines.append("")
    # Accepting states (all states for NOPSOS; RE accept states for SOS+RE).
    lines.append(f"set FINAL := {' '.join(str(s) for s in dfa.final())};")
    lines.append("")
    # Execution cap eta[v]: operation v may occupy at most eta[v] slots.  eta[v]=1
    # is single-execution; eta[v]=n_slots imposes no cap so v may recur across
    # slots (re-execution governed by the DFA).  By default every operation is
    # uncapped (eta[v]=n_slots), reproducing the previous enforce_single=0
    # behaviour; a per-operation cap can be supplied via inst._eta (op -> cap).
    eta_map = getattr(inst, "_eta", None) or {}
    lines.append("param eta :=")
    for v in W:
        lines.append(f"  {v} {int(eta_map.get(v, inst.n_slots))}")
    lines.append(";")
    lines.append("")

    # =============================================================
    # Operational layer data
    # =============================================================
    lines += [
        "# =============================================================",
        "# Operational layer data",
        "# =============================================================",
        "",
    ]

    # Resource sets
    lines.append(f"set R_V := {' '.join(inst.R_V)};")
    lines.append(f"set R_S := {' '.join(inst.R_S)};")
    lines.append(f"set R_C := {' '.join(inst.R_C)};")
    lines.append(f"set R_D := {' '.join(inst.R_D)};")
    lines.append("")

    # Operation subsets
    lines.append(f"set W_U := {' '.join(inst.W_U)};")
    lines.append(f"set W_T := {' '.join(inst.W_T)};")
    lines.append(f"set W_D := {' '.join(inst.W_D)};")
    lines.append("")

    # Crude types and properties
    lines.append(f"set C := {' '.join(inst.C)};")
    lines.append(f"set K := {' '.join(inst.K)};")
    lines.append("")

    # Operation connectivity
    _write_param_dict(lines, "op_inlet",  inst.op_inlet)
    _write_param_dict(lines, "op_outlet", inst.op_outlet)
    lines.append("")

    # Vessel arrival times
    _write_param_dict(lines, "s_r", inst.s_r)

    # Flowrate bounds
    _write_param_dict(lines, "FR_lb", inst.FR_lb)
    _write_param_dict(lines, "FR_ub", inst.FR_ub)

    # Volume bounds
    _write_param_dict(lines, "Vt_lb", inst.Vt_lb)
    _write_param_dict(lines, "Vt_ub", inst.Vt_ub)

    # Tank capacity
    _write_param_dict(lines, "Lt_lb", inst.Lt_lb)
    _write_param_dict(lines, "Lt_ub", inst.Lt_ub)
    _write_param_dict(lines, "Lt0",   inst.Lt0)
    lines.append("")

    # Demand
    _write_param_dict(lines, "D_lb", inst.D_lb)
    _write_param_dict(lines, "D_ub", inst.D_ub)
    lines.append("")

    # Distillation count bounds
    lines.append(f"param ND_lb := {inst.ND_lb};")
    lines.append(f"param ND_ub := {inst.ND_ub};")
    lines.append("")

    # Property bounds (only non-trivial entries)
    if inst.x_lb:
        lines.append("param x_lb :=")
        for (v, k), val in inst.x_lb.items():
            lines.append(f"  [{v},{k}] {val}")
        lines.append(";")
    if inst.x_ub:
        lines.append("param x_ub :=")
        for (v, k), val in inst.x_ub.items():
            lines.append(f"  [{v},{k}] {val}")
        lines.append(";")
    lines.append("")

    # Crude properties
    if inst.x_prop:
        lines.append("param x_prop :=")
        for (c, k), val in inst.x_prop.items():
            lines.append(f"  [{c},{k}] {val}")
        lines.append(";")

    # Gross margins
    _write_param_dict(lines, "G_crude", inst.G_crude)
    lines.append("")

    # Initial per-crude levels (sparse; missing entries default to 0)
    if inst.L0:
        lines.append("param L0 :=")
        for (r, c), val in inst.L0.items():
            lines.append(f"  [{r},{c}] {val}")
        lines.append(";")
    lines.append("")

    Path(dat_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dat_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[DAT] Written: {dat_path}")
    _print_size_summary(inst, dfa)


# ---------------------------------------------------------------------------
# Helper writers
# ---------------------------------------------------------------------------

def _write_set_pairs(lines: list, name: str, pairs):
    if pairs:
        lines.append(f"set {name} :=")
        for v, w in pairs:
            lines.append(f"  ({v}, {w})")
        lines.append(";")
    else:
        lines.append(f"set {name} := ;")


def _write_param_dict(lines: list, name: str, d: dict,
                      symbolic: bool = False):
    if not d:
        return
    lines.append(f"param {name} :=")
    for k, v in d.items():
        if isinstance(k, tuple):
            key_str = " ".join(str(x) for x in k)
        else:
            key_str = str(k)
        val_str = f"'{v}'" if symbolic or isinstance(v, str) else str(v)
        lines.append(f"  {key_str} {val_str}")
    lines.append(";")


def _print_size_summary(inst: CrudeOilFullInstance, dfa: DFA):
    n = inst.n_slots
    W_size = len(inst.W)
    n_arcs = dfa.n_arcs
    n_Z = n * W_size
    n_f = n * n_arcs
    n_Vt = n * W_size
    n_Vc = n * W_size * len(inst.C)
    tanks = inst.R_S + inst.R_C
    n_Lt = n * len(tanks)
    n_Lc = n * len(tanks) * len(inst.C)
    print(f"  |W|={W_size}  n_slots={n}  |C|={len(inst.C)}  |K|={len(inst.K)}")
    print(f"  Binary vars : Z={n_Z}  f={n_f}  total={n_Z+n_f}")
    print(f"  Cont. vars  : Vt={n_Vt}  Vc={n_Vc}  Lt={n_Lt}  Lc={n_Lc}")
    print(f"  DFA         : {dfa.n_states} states  {dfa.n_arcs} arcs")
