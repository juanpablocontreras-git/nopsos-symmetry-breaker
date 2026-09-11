"""
Performance plots for the NOPSOS paper.

Reads the result CSVs in ``results/`` and writes publication figures (PDF for
LaTeX inclusion, PNG for preview) into ``figures/``, or into ``NOPSOS_FIGDIR``
when that environment variable is set.  The figures
summarise the experiments that the tables report row by row, so the paper can
keep one or two tables for exact numbers and show the trends as plots.

Design: two configurations are compared throughout -- NOPSOS+DFA-SBS versus the
bare no-SBS formulation (crude-oil size uses NOPSOS-DFA versus SOS-RE).  Series
are distinguished by a colourblind-safe blue/orange pair *and* by marker,
line style and hatch, so the figures also read in grayscale print.

Run:  python -m src.plots
"""

import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
# Set NOPSOS_FIGDIR to write straight into a manuscript image directory.
FIGDIR = Path(os.environ.get("NOPSOS_FIGDIR") or ROOT / "figures")

# Colourblind-safe categorical pair (blue/orange) + a green for the third series.
C_DFA   = "#2a78d6"    # NOPSOS + DFA-SBS
C_NOSBS = "#eb6834"    # bare no-SBS  /  SOS + RE
C_THIRD = "#008300"
C_AUTO  = "#7b5cd6"    # bare no-SBS, Gurobi symmetry = auto
C_OFF   = "#c0392b"    # bare no-SBS, Gurobi symmetry = off
TL = 600.0             # time limit (s)


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 140,
    })


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {FIGDIR / (name + '.pdf')}")


def _load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(row, key):
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _by(rows, config):
    """Index rows of one config by instance index (order of appearance)."""
    out = {}
    for r in rows:
        if r["config"] == config:
            out[r["instance_name"]] = r
    return out


# ---------------------------------------------------------------------------
# Figure 1: job-shop -- B&B nodes and solving time, DFA vs no-SBS, per objective
# ---------------------------------------------------------------------------

def _load_opt(path):
    """Load a CSV if present, else return an empty list (optional overlay)."""
    return _load(path) if path.exists() else []


def fig_jobshop():
    # Each objective merges the NOPSOS runs (DFA + no-SBS) with the optional
    # classical disjunctive baseline written to its own CSV by
    # `--config classical`, so the classical series only appears once it exists.
    data = {
        "Makespan": _load(RESULTS / "jobshop_experiments.csv")
                    + _load_opt(RESULTS / "jobshop_classical_experiments.csv"),
        "Weighted completion time":
                    _load(RESULTS / "jobshop_wct_experiments.csv")
                    + _load_opt(RESULTS / "jobshop_classical_wct_experiments.csv"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)

    for col, (obj, rows) in enumerate(data.items()):
        dfa, nos = _by(rows, "dfa"), _by(rows, "nosbs-off")
        cls = _by(rows, "classical")
        insts = sorted(dfa, key=lambda n: int(n.split("_")[1]))
        # series present in this figure (classical only if its CSV was loaded)
        series = [("NOPSOS+DFA-SBS", dfa, C_DFA, None),
                  ("no-SBS", nos, C_NOSBS, "///")]
        if cls:
            series.append(("classical MILP", cls, C_THIRD, ".."))
        n = len(series)
        w = 0.8 / n
        offs = [(-(n - 1) / 2 + k) * w for k in range(n)]
        x = range(len(insts))

        # --- row 0: nodes (log) ---
        ax = axes[0][col]
        for (label, cfg, color, hatch), off in zip(series, offs):
            vals = [max(_f(cfg.get(i, {}), "n_nodes") or 1, 1) for i in insts]
            ax.bar([i + off for i in x], vals, w, color=color, label=label,
                   edgecolor="white", linewidth=0.4, hatch=hatch)
        ax.set_yscale("log")
        ax.set_title(obj)
        if col == 0:
            ax.set_ylabel("branch-and-bound nodes")

        # --- row 1: time (log), TL capped and marked ---
        ax = axes[1][col]
        for (label, cfg, color, hatch), off in zip(series, offs):
            vals = [min(_f(cfg.get(i, {}), "time") or TL, TL) for i in insts]
            ax.bar([i + off for i in x], vals, w, color=color, edgecolor="white",
                   linewidth=0.4, hatch=hatch)
        ax.axhline(TL, color="#888888", linestyle=":", linewidth=0.9)
        ax.text(len(insts) - 0.5, TL, " TL", va="center", ha="left",
                fontsize=7, color="#666666")
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels([i.split("_")[1] for i in insts])
        ax.set_xlabel("instance")
        if col == 0:
            ax.set_ylabel("solving time (s)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, "fig_jobshop_performance")


# ---------------------------------------------------------------------------
# Figure 1b: job-shop -- Gurobi built-in symmetry detection is ineffective
# on the bare priority-slot model (symmetry=auto vs symmetry=off behave the
# same).  Both bare configs solve the identical no-SBS formulation; the only
# difference is Gurobi's `symmetry` parameter (-1 automatic vs 0 off).  If the
# solver's own symmetry handling helped, `auto` would beat `off`; it does not.
# ---------------------------------------------------------------------------

def fig_jobshop_symmetry():
    data = {
        "Makespan": _load(RESULTS / "jobshop_experiments.csv"),
        "Weighted completion time": _load(RESULTS / "jobshop_wct_experiments.csv"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)

    for col, (obj, rows) in enumerate(data.items()):
        auto, off = _by(rows, "nosbs-auto"), _by(rows, "nosbs-off")
        insts = sorted(set(auto) & set(off), key=lambda n: int(n.split("_")[1]))
        x = range(len(insts))
        w = 0.4

        # --- row 0: nodes (log) ---
        ax = axes[0][col]
        na = [max(_f(auto[i], "n_nodes") or 1, 1) for i in insts]
        no = [max(_f(off[i], "n_nodes") or 1, 1) for i in insts]
        ax.bar([i - w/2 for i in x], na, w, color=C_AUTO,
               label="no-SBS, Gurobi symmetry = auto",
               edgecolor="white", linewidth=0.4)
        ax.bar([i + w/2 for i in x], no, w, color=C_OFF,
               label="no-SBS, Gurobi symmetry = off",
               edgecolor="white", linewidth=0.4, hatch="///")
        ax.set_yscale("log")
        ax.set_title(obj)
        if col == 0:
            ax.set_ylabel("branch-and-bound nodes")

        # --- row 1: time (log), TL capped and marked ---
        ax = axes[1][col]
        ta = [min(_f(auto[i], "time") or TL, TL) for i in insts]
        to = [min(_f(off[i], "time") or TL, TL) for i in insts]
        ax.bar([i - w/2 for i in x], ta, w, color=C_AUTO, edgecolor="white",
               linewidth=0.4)
        ax.bar([i + w/2 for i in x], to, w, color=C_OFF, edgecolor="white",
               linewidth=0.4, hatch="///")
        ax.axhline(TL, color="#888888", linestyle=":", linewidth=0.9)
        ax.text(len(insts) - 0.5, TL, " TL", va="center", ha="left",
                fontsize=7, color="#666666")
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels([i.split("_")[1] for i in insts])
        ax.set_xlabel("instance")
        if col == 0:
            ax.set_ylabel("solving time (s)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, "fig_jobshop_gurobi_symmetry")


def fig_jobshop_symmetry_makespan():
    """Makespan only: B&B nodes and solving time side-by-side (auto vs off)."""
    rows = _load(RESULTS / "jobshop_experiments.csv")
    auto, off = _by(rows, "nosbs-auto"), _by(rows, "nosbs-off")
    insts = sorted(set(auto) & set(off), key=lambda n: int(n.split("_")[1]))
    x = range(len(insts))
    w = 0.4

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # --- left: nodes (log) ---
    ax = axes[0]
    na = [max(_f(auto[i], "n_nodes") or 1, 1) for i in insts]
    no = [max(_f(off[i], "n_nodes") or 1, 1) for i in insts]
    ax.bar([i - w/2 for i in x], na, w, color=C_AUTO,
           label="no-SBS, Gurobi symmetry = auto", edgecolor="white", linewidth=0.4)
    ax.bar([i + w/2 for i in x], no, w, color=C_OFF,
           label="no-SBS, Gurobi symmetry = off", edgecolor="white",
           linewidth=0.4, hatch="///")
    ax.set_yscale("log")
    ax.set_ylabel("branch-and-bound nodes")
    ax.set_xlabel("instance")
    ax.set_xticks(list(x))
    ax.set_xticklabels([i.split("_")[1] for i in insts])

    # --- right: time (log), TL capped and marked ---
    ax = axes[1]
    ta = [min(_f(auto[i], "time") or TL, TL) for i in insts]
    to = [min(_f(off[i], "time") or TL, TL) for i in insts]
    ax.bar([i - w/2 for i in x], ta, w, color=C_AUTO, edgecolor="white",
           linewidth=0.4)
    ax.bar([i + w/2 for i in x], to, w, color=C_OFF, edgecolor="white",
           linewidth=0.4, hatch="///")
    ax.axhline(TL, color="#888888", linestyle=":", linewidth=0.9)
    ax.text(len(insts) - 0.5, TL, " TL", va="center", ha="left",
            fontsize=7, color="#666666")
    ax.set_yscale("log")
    ax.set_ylabel("solving time (s)")
    ax.set_xlabel("instance")
    ax.set_xticks(list(x))
    ax.set_xticklabels([i.split("_")[1] for i in insts])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, "fig_jobshop_gurobi_symmetry_makespan")


# ---------------------------------------------------------------------------
# Figure 2: crude-oil isolation -- performance profiles (solving time)
# ---------------------------------------------------------------------------

def _finished(row):
    """A run that produced a certificate (optimal or infeasible), not a TL."""
    return row is not None and row.get("status") not in (None, "", "limit") \
        and _f(row, "time") is not None


def _perf_profile(ax, rows, title):
    dfa, nos = _by(rows, "dfa"), _by(rows, "nosbs-off")
    insts = [i for i in dfa if i in nos]
    series = {"NOPSOS+DFA-SBS": (dfa, C_DFA, "-", "o"),
              "no-SBS": (nos, C_NOSBS, "--", "s")}
    # best finite time per instance across the two configs
    best = {}
    for i in insts:
        ts = [_f(r[i], "time") for r in (dfa, nos) if _finished(r[i])]
        best[i] = min(ts) if ts else None

    taus = [1.0 * 1.15**k for k in range(0, 34)]     # 1 .. ~100
    for label, (cfg, color, ls, mk) in series.items():
        ratios = []
        for i in insts:
            if _finished(cfg[i]) and best[i]:
                ratios.append(_f(cfg[i], "time") / best[i])
        frac = [sum(r <= t for r in ratios) / len(insts) for t in taus]
        ax.step(taus, frac, where="post", color=color, linestyle=ls,
                linewidth=1.8, label=label)
    ax.set_xscale("log")
    ax.set_xlim(1, taus[-1])
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.set_xlabel(r"time within factor $\tau$ of best")


def fig_crude_perf():
    files = [("makespan_600.csv", "Makespan"),
             ("wct_600.csv", "Weighted completion time"),
             ("feasibility_600.csv", "Feasibility")]
    present = [(f, t) for f, t in files if (RESULTS / f).exists()]
    fig, axes = plt.subplots(1, len(present), figsize=(7.2, 2.7), sharey=True)
    if len(present) == 1:
        axes = [axes]
    for ax, (f, t) in zip(axes, present):
        _perf_profile(ax, _load(RESULTS / f), t)
    axes[0].set_ylabel("fraction of instances")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig_crude_perfprofile")


# ---------------------------------------------------------------------------
# Figure 3: crude-oil isolation -- node reduction scatter (DFA vs no-SBS)
# ---------------------------------------------------------------------------

def fig_crude_nodes():
    objs = [("makespan_600.csv", "makespan", "o"),
            ("wct_600.csv", "WCT", "^"),
            ("feasibility_600.csv", "feasibility", "s")]
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    lo, hi = 1, 1
    for f, label, mk in objs:
        if not (RESULTS / f).exists():
            continue
        rows = _load(RESULTS / f)
        dfa, nos = _by(rows, "dfa"), _by(rows, "nosbs-off")
        xs, ys = [], []
        for i in dfa:
            if i not in nos:
                continue
            nd, nn = _f(dfa[i], "n_nodes"), _f(nos[i], "n_nodes")
            if nd and nn and nd >= 1 and nn >= 1:
                xs.append(nn); ys.append(nd)
        if not xs:
            continue
        hi = max(hi, max(xs), max(ys))
        ax.scatter(xs, ys, s=26, marker=mk, facecolor=C_DFA, edgecolor="white",
                   linewidth=0.4, alpha=0.85, label=label)
    hi *= 1.6
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle="--", linewidth=1.0)
    ax.text(hi, hi, " y = x", fontsize=7, color="#666666", va="center")
    ax.fill_between([lo, hi], [lo, lo], [hi, hi], color=C_DFA, alpha=0.05)
    ax.annotate("DFA explores\nfewer nodes", xy=(hi*0.5, hi*0.03),
                fontsize=7.5, color="#555555", ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("no-SBS nodes")
    ax.set_ylabel("NOPSOS+DFA-SBS nodes")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig_crude_nodes")


# ---------------------------------------------------------------------------
# Figure 4: automaton size (RE vs DFA) and symmetry reduction, vs |W|
# ---------------------------------------------------------------------------

def fig_size_symmetry():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # --- automaton size ---
    if (RESULTS / "dfa_vs_re_size.csv").exists():
        rows = sorted(_load(RESULTS / "dfa_vs_re_size.csv"),
                      key=lambda r: _f(r, "n_ops") or 0)
        w = [_f(r, "n_ops") for r in rows]
        re = [_f(r, "re_states") for r in rows]
        df = [_f(r, "dfa_states") for r in rows]
        ax = axes[0]
        ax.scatter(w, re, s=24, marker="s", color=C_NOSBS, edgecolor="white",
                   linewidth=0.4, label="SOS+RE")
        ax.scatter(w, df, s=24, marker="o", color=C_DFA, edgecolor="white",
                   linewidth=0.4, label="NOPSOS+DFA")
        ax.set_xlabel(r"operations $|W|$")
        ax.set_ylabel("automaton states")
        ax.set_title("Symmetry-breaker size")
        ax.legend(frameon=False, loc="upper left")

    # --- symmetry reduction ---
    if (RESULTS / "symmetry_experiments.csv").exists():
        rows = [r for r in _load(RESULTS / "symmetry_experiments.csv")
                if _f(r, "reduction")]
        rows.sort(key=lambda r: _f(r, "n_ops") or 0)
        w = [_f(r, "n_ops") for r in rows]
        red = [_f(r, "reduction") for r in rows]
        ax = axes[1]
        ax.scatter(w, red, s=24, marker="D", color=C_THIRD, edgecolor="white",
                   linewidth=0.4)
        ax.set_yscale("log")
        ax.set_xlabel(r"operations $|W|$")
        ax.set_ylabel(r"sequence-count reduction ($\times$)")
        ax.set_title("Symmetry removed by the DFA")

    fig.tight_layout()
    _save(fig, "fig_size_symmetry")


# ---------------------------------------------------------------------------
# The 100-instance study: closure, profiles, and where the reduction pays
# ---------------------------------------------------------------------------

RUN100 = [("run100_wct.csv",  "Weighted completion time", "WCT"),
          ("run100_mks.csv",  "Makespan", "Makespan"),
          ("run100_inf.csv",  "Infeasible deadline", "Infeas. deadline"),
          ("run100_full.csv", "Gross margin (full model)", "Gross margin")]


def _run100_present():
    return [(f, t, n) for f, t, n in RUN100 if (RESULTS / f).exists()]


def _pairs(rows):
    """Instances with both configs present, as (instance, dfa_row, bare_row)."""
    dfa, nos = _by(rows, "dfa"), _by(rows, "nosbs-off")
    return [(i, dfa[i], nos[i]) for i in sorted(dfa, key=lambda s: int(s.split("_")[-1]))
            if i in nos]


def fig_run100_closure():
    """Instances certified within a time budget. The closure gap is the point."""
    _style()
    present = _run100_present()
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.2), sharex=True)
    flat = axes.ravel()
    for ax, (f, _full, short) in zip(flat, present):
        prs = _pairs(_load(RESULTS / f))
        n = len(prs)
        for label, idx, color, ls in (("NOPSOS+DFA-SBS", 1, C_DFA, "-"),
                                      ("no-SBS", 2, C_NOSBS, "--")):
            ts = sorted(_f(pr[idx], "time") for pr in prs if _finished(pr[idx]))
            xs = [0.01] + ts + [TL]
            ys = [0] + list(range(1, len(ts) + 1)) + [len(ts)]
            ax.step(xs, ys, where="post", color=color, linestyle=ls, linewidth=1.8,
                    label=label)
        ax.set_xscale("log")
        ax.set_xlim(0.05, TL)
        ax.set_ylim(0, n * 1.02)
        ax.set_title("%s (n=%d)" % (short, n))
    for ax in flat[len(present):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("instances certified")
    h, l = flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02),
               frameon=False)
    fig.supxlabel("time budget (s)", fontsize=9, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, "fig_run100_closure")


def fig_run100_perf():
    _style()
    present = _run100_present()
    fig, axes = plt.subplots(1, len(present), figsize=(7.4, 2.6), sharey=True)
    if len(present) == 1:
        axes = [axes]
    for ax, (f, _full, short) in zip(axes, present):
        _perf_profile(ax, _load(RESULTS / f), short)
        ax.set_xlabel("")
    axes[0].set_ylabel("fraction of instances")
    fig.supxlabel(r"time within factor $\tau$ of best", fontsize=9, y=0.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.11),
               frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, "fig_run100_perfprofile")


def _ratio_strip(ax, series, ylabel, title):
    """One column of points per objective, with the median marked."""
    for x, (label, vals) in enumerate(series):
        if not vals:
            continue
        jitter = [x + 0.26 * (((k * 7919) % 101) / 100.0 - 0.5) for k in range(len(vals))]
        ax.scatter(jitter, vals, s=9, alpha=0.55, color=C_DFA,
                   edgecolors="none", zorder=3)
        m = sorted(vals)[len(vals) // 2]
        ax.plot([x - 0.28, x + 0.28], [m, m], color=C_NOSBS, linewidth=2.0, zorder=4)
        ax.annotate("%.1f" % m, (x + 0.30, m), fontsize=7, color=C_NOSBS,
                    va="center")
    ax.axhline(1.0, color="#888888", linewidth=0.9, linestyle=":", zorder=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([lb for lb, _ in series], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def fig_run100_boundary():
    """Where symmetry breaking pays: nodes fall everywhere, time only sometimes."""
    _style()
    present = _run100_present()
    nodes, times = [], []
    for f, _full, short in present:
        nr, tr = [], []
        for _i, d, b in _pairs(_load(RESULTS / f)):
            if not (_finished(d) and _finished(b)):
                continue
            nd, nb = _f(d, "n_nodes"), _f(b, "n_nodes")
            if nd and nb and nd > 0:
                nr.append(nb / nd)
            td, tb = _f(d, "time"), _f(b, "time")
            if td and tb and td > 0:
                tr.append(tb / td)
        nodes.append((short, nr))
        times.append((short, tr))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    _ratio_strip(axes[0], nodes, "search-tree reduction", "Nodes (no-SBS / DFA)")
    _ratio_strip(axes[1], times, "speedup", "Time (no-SBS / DFA)")
    fig.tight_layout()
    _save(fig, "fig_run100_boundary")


def fig_run100_scale():
    """Reduction against bare tree size.

    Spearman rho is +0.44 on WCT and +0.32 on the full model, so the gain rises
    with the tree only as a tendency. On makespan (+0.04) and the infeasible
    deadline (-0.20) it does not rise at all, which is the same boundary the
    strip plot shows from the other side.
    """
    _style()
    present = _run100_present()
    marks = ["o", "^", "s", "D"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    colors = [C_DFA, C_THIRD, C_NOSBS, C_AUTO]
    for (f, full, _short), mk, col in zip(present, marks, colors):
        xs, ys = [], []
        for _i, d, b in _pairs(_load(RESULTS / f)):
            if not (_finished(d) and _finished(b)):
                continue
            nd, nb = _f(d, "n_nodes"), _f(b, "n_nodes")
            # a tree of a handful of nodes closed at the root; the ratio there
            # reports presolve noise, not symmetry removed
            if nd and nb and nd > 0 and nb >= 10:
                xs.append(nb)
                ys.append(nb / nd)
        ax.scatter(xs, ys, s=14, marker=mk, alpha=0.7, color=col,
                   edgecolors="none", label=full, zorder=3)
    ax.axhline(1.0, color="#888888", linewidth=0.9, linestyle=":", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("no-SBS search-tree size (nodes)")
    ax.set_ylabel("search-tree reduction")
    ax.legend(frameon=False, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, 1.20), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig_run100_scale")


# ---------------------------------------------------------------------------
# Symmetry removed against symmetry remaining, on the enumerable instances
# ---------------------------------------------------------------------------

def fig_symmetry():
    """What the DFA removes, and what it leaves.

    The two panels stop at different sizes because the two enumerations do.
    Sequence counts come from subset enumeration and reach twenty operations;
    trace counts need every accepted word enumerated and reach fourteen.
    """
    _style()
    seq = {r["instance"]: r for r in _load(RESULTS / "symmetry_synth30.csv")}
    tr = {r["instance"]: r for r in _load(RESULTS / "traces_synth30.csv")}

    xb, yb, xd, yd = [], [], [], []
    xa, ya, xt, yt = [], [], [], []
    for k, r in sorted(seq.items(), key=lambda kv: int(kv[0].split("_")[-1])):
        n = _f(r, "n_ops")
        nb, nd = _f(r, "n_seq_no_sbs"), _f(r, "n_seq_dfa")
        if nb and nd:
            xb.append(n); yb.append(nb)
            xd.append(n); yd.append(nd)
        t = tr.get(k)
        if t and _f(t, "n_traces"):
            xa.append(n); ya.append(_f(t, "n_accepted"))
            xt.append(n); yt.append(_f(t, "n_traces"))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    ax.scatter(xb, yb, s=24, marker="o", color=C_NOSBS, edgecolors="none",
               label="precedence-feasible", zorder=3)
    ax.scatter(xd, yd, s=24, marker="^", color=C_DFA, edgecolors="none",
               label="accepted by the DFA", zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(6.4, 20.6)
    ax.set_xlabel("operations $|W|$")
    ax.set_ylabel("slot sequences")
    ax.set_title("Removed: feasible against accepted")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[1]
    ax.scatter(xa, ya, s=24, marker="^", color=C_DFA, edgecolors="none",
               label="accepted by the DFA", zorder=3)
    ax.scatter(xt, yt, s=24, marker="s", color=C_THIRD, edgecolors="none",
               label="traces (distinct schedules)", zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(6.4, 14.6)
    ax.set_xlabel("operations $|W|$")
    ax.set_ylabel("slot sequences")
    ax.set_title("Remaining: accepted against traces")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    fig.tight_layout()
    _save(fig, "fig_symmetry")


def main():
    _style()
    fig_jobshop()
    fig_jobshop_symmetry()
    fig_jobshop_symmetry_makespan()
    fig_crude_perf()
    fig_crude_nodes()
    fig_size_symmetry()
    fig_run100_closure()
    fig_run100_perf()
    fig_run100_boundary()
    fig_run100_scale()
    fig_symmetry()
    print("done.")


if __name__ == "__main__":
    main()
