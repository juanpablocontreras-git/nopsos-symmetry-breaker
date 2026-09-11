# A general automatic symmetry breaker for priority-slot process scheduling

Replication code, instances and results for the paper *A General Automatic
Symmetry Breaker for Process Scheduling with Continuous-Time Representation*.

A priority-slot (Single Operation Sequencing) scheduling formulation, NOPSOS,
whose combinatorial symmetries are removed by a symmetry breaker built
automatically as a deterministic finite automaton from the precedence and
non-overlapping relations of the problem. The automaton's accepted language is
embedded in the mixed-integer program through a unit-flow formulation. Models
are solved with AMPL + Gurobi.

## Setup

```powershell
python -m pip install -r requirements.txt
$env:AMPL_LICENSE_UUID="your-license-uuid"
```

Reproducing the experiments needs AMPL and Gurobi licences. Regenerating the
figures from the bundled result CSVs needs only Python and matplotlib.

## Reproducing the paper

Every experiment runs through one driver. `--objective` picks the experiment and
`--instances` the collection.

| Paper item | Command | Result file |
|---|---|---|
| DFA vs RE on the four refineries | `python run_experiments.py --objective gross_margin --instances mouret --config re dfa --time-limit 600` | `dfa_vs_re_size.csv` |
| Symmetry removed and remaining | `python run_experiments.py --objective symmetry --instances suiteA` | `symmetry_synth30.csv`, `traces_synth30.csv` |
| Solver symmetry detection | `python run_experiments.py --objective wct --instances suiteA --config dfa nosbs-auto nosbs-off --time-limit 600` | `wct_600.csv` |
| Weighted completion time | `python run_experiments.py --objective wct --instances suiteB --config dfa nosbs-off --time-limit 600` | `run100_wct.csv` |
| Makespan | `python run_experiments.py --objective makespan --instances suiteB --config dfa nosbs-off --time-limit 600` | `run100_mks.csv` |
| Infeasible deadlines | `python run_experiments.py --objective feasibility --instances suiteB --config dfa nosbs-off --time-limit 600` | `run100_inf.csv` |
| Gross margin, full model | `python run_experiments.py --objective gross_margin --instances suiteB --config dfa nosbs-off --time-limit 600` | `run100_full.csv` |
| Variable-free encoding | `python run_experiments.py --objective makespan --instances suiteA --config dfa pairwise nosbs-off --time-limit 600` | `pairwise_wct_600.csv` |

Result files live in `results/`. Those committed here are the ones the paper's
tables and figures were computed from, so the numbers can be checked without
re-running the solver. Raw solver logs are not tracked; they are regenerable.

Figures:

```powershell
python -m src.plots                  # writes figures/
$env:NOPSOS_FIGDIR="path\to\images"  # or straight into a manuscript tree
```

Main options:

- `--objective {gross_margin, makespan, wct, feasibility, jobshop, openshop, size, symmetry}`
- `--instances {mouret, suiteA, suiteB, crude, jobshop, openshop, all, <comma-separated list>}`
- `--config` — `dfa` (the automatic DFA), `re` (the handcrafted regular
  expression), `pairwise` (the variable-free encoding of the same language),
  `nosbs-auto` / `nosbs-off` (no breaker, solver symmetry detection on / off),
  `classical` (job-shop only: a disjunctive MILP baseline)
- `--time-limit` (seconds), `--mip-gap`, `--seeds`, `--out <csv>`
- `--nlp` — `gross_margin` only: run the MILP→NLP two-step
- `--running-example` — print the worked example and exit

## Layout

- `run_experiments.py` — unified experiment driver.
- `models/crude_oil.mod` — priority-slot core + symmetry breaker + operational
  layer (gross-margin objective).
- `models/scheduling.mod` — sequencing core (makespan / WCT / feasibility).
  Job shop reuses this model unchanged.
- `models/jobshop_classical.mod` — classical disjunctive job-shop MILP baseline.
- `models/nopsos.mod` — reference core model for the worked example.
- `src/dfa_builder.py` — the linear extension and the automaton construction.
  Reads only the two relations; nothing in it is application-specific.
- `src/sos_re.py` — the regular-expression automaton of Mouret et al., the
  literature benchmark.
- `src/crude_oil.py`, `src/crude_oil_ampl.py` — the crude-oil operational layer.
- `src/instance_generator.py` — the seeded instance generator.
- `src/plots.py` — figures.
- `results/` — aggregated result CSVs.
- `figures/` — generated figures.

## Instances

| Collection | Files | What it is |
|---|---|---|
| `mouret` | `problem{1,2,3,4}_mouret.json` | the four published refineries |
| `suiteA` | `synthd_01..30.json` | Suite A, 30 instances, 7 to 26 operations |
| `suiteB` | `synth100_001..100.json` | Suite B, 100 instances, 6 to 26 operations |
| `crude` | `crude_01..10.json` | an earlier 10-instance crude-oil suite |
| `jobshop` | `jobshop_01..10.json` | job-shop instances, not reported in the paper |
| — | `wang_ex1..6.json` | instances from Wang et al., not reported in the paper |
| — | `instance_template.json` | annotated schema reference |

Instances are numbered in increasing order of the size parameter `s`.
Processing times and completion-time weights for the sequencing core are drawn
deterministically from the instance name, so those runs are reproducible
without the values being stored in the files.

Further synthetic instances can be generated (written to `instances/`, not
tracked):

```powershell
python -m src.instance_generator 30           # crude-oil synthetic pool
python -m src.instance_generator jobshop 10   # job-shop suite
```

### A note on the published refinery data

The four refineries are used as published, with one correction. On refinery 2
the second property of crudes D, E and F is taken as 0.0333, 0.023 and 0.0133
rather than 0.333, 0.23 and 0.133. The published figures place each charging
tank's initial content an order of magnitude outside the specification of the
mix it must deliver, which makes the model infeasible once the bilinear
composition constraint is enforced. The corrected values lie inside their
windows and reproduce the reported optimum. The paper's appendix discusses this.

### Job shop: an additional application, not in the paper

Job-shop instances map onto the same two relations with no operational layer:
routing chains become the precedence graph and each machine a non-overlapping
clique. This path is kept because it exercises the claim that the construction
reads only the two relations — nothing in `src/dfa_builder.py` changes between
crude oil and job shop. It is deliberately not part of the paper: a
priority-slot model is not the right tool for a job shop, since a disjunctive
formulation has one binary per machine-sharing pair, no slots, and therefore
none of the symmetry studied here.

```powershell
python run_experiments.py --objective jobshop --instances jobshop --config dfa pairwise nosbs-off
```

## Citing

If you use this code, please cite the paper.

## License

MIT. See [LICENSE](LICENSE).
