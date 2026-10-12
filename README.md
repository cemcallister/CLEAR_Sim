# CLEAR simulation code

This package regenerates every simulated figure, table and quoted number in the
paper *"Extending Our Understanding of the Dynamic, State- and Context-Dependent
Nature of ADHD: An Integrative Control-Theoretic Model"* (McAllister & Sonuga-Barke).

## What this is

This is the paper's minimal, unfitted implementation of the CLEAR model. Every
quantity is in arbitrary units and nothing is fitted to data. The simulations
show what the model produces; no number here is a claim about a particular
person, a clinical measurement, or a duration in real time. The code is
deliberately small and direct so that it can be read alongside the model
equations in Sections 3 and 5 of the paper.

Running it reproduces the figures exactly as they appear in the manuscript, the
robustness table (Table S1), and the numbers the text and captions print.

## Requirements and installation

- **Python 3.12, 3.13 or 3.14** (tested on 3.12.11). The pinned NumPy 2.5.3 and
  SciPy 1.18.1 need Python 3.12 or later. With these versions and Matplotlib
  3.10.9 the figures are pixel-identical to the published PDFs.
- `requirements-lock.txt` pins every package in the environment used for the
  paper, including the libraries that NumPy, SciPy and Matplotlib depend on.
  `requirements.txt` pins only those three and lets pip choose the rest.

It is safest to work in a *virtual environment*, which keeps these packages
separate from the rest of your computer. In a terminal, from this folder:

```
python3 -m venv venv              # needs Python 3.12 or later: check with python3 --version
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements-lock.txt
```

(You only need to do this once. `source venv/bin/activate` each time you open a
new terminal.)

With conda, `environment.yml` provides Python 3.12:

```
conda env create -f environment.yml
conda activate clear-sim
pip install -r requirements-lock.txt
```

## How to run

To regenerate everything and run all the checks:

```
python run_all.py
```

This writes all output into a new `output/` folder (figures as PDF and PNG, the
Table S1 scan as `scan_robust.npz` and `scan_robust.log`, and the
`<stem>_numbers.json` files), then checks the regenerated numbers against the
reference copies in `expected/`, runs `paper_numbers.py`, and runs the
independent checks. It prints a summary and a per-step timing table.

**Approximate runtimes** (a 2020-era laptop, Python 3.12): the nine figures take
two to three minutes all together; Table S1 takes about two to three minutes (it
is a scan of 100 parameter combinations over 100 periods each); the checks take
about a minute. The whole run is roughly five to six minutes.

You can also run a single figure on its own, for example:

```
python figures/fig3_hysteresis.py
python tables/table_S1_robustness.py
python paper_numbers.py
```

Each figure script writes its own PDF, PNG and `<stem>_numbers.json` into
`output/`. The output location can be changed by setting the environment
variable `CLEAR_FIG_OUT` to a folder of your choice.

## What each script makes

The figure files keep the stems they had while the paper was being written.
**These stems deliberately do not match the figure numbers in the paper** (the
figures were reordered late on). The manuscript's LaTeX source includes each
figure by its stem, so the stems must not be renamed. The mapping is:

| Script | Output stem | In the paper | Section |
|---|---|---|---|
| `figures/fig1_boundary.py` | `fig1_boundary` | Figure 1 | 3 (schematic) |
| `figures/fig2_parameter_map.py` | `fig2_parameter_map` | Figure 2 | 3.7 |
| `figures/fig3_hysteresis.py` | `fig3_hysteresis` | Figure 3 | 3.7 |
| `figures/fig4_proximity.py` | `fig4_proximity` | Figure 4 | 5.3 |
| `figures/fig6_burnout.py` | `fig6_burnout` | **Figure 5** (`fig:burnout`) | 5.4 |
| `figures/fig5_drive.py` | `fig5_drive` | **Figure 6** (`fig:drive`) | 5.5 |
| `figures/figS3_hysteresis_loop.py` | `figS3_hysteresis_loop` | Figure S1 | 3.7 / S1 |
| `figures/figS4_proximity_stats.py` | `figS4_proximity_stats` | Figure S2 | 5.3 / S2 |
| `figures/figS5_drive_sweep.py` | `figS5_drive_sweep` | Figure S3 | 5.5 / S3 |
| `tables/table_S1_robustness.py` | `scan_robust` | Table S1 | 5.4 / S1 |

Note the two swaps: `fig6_burnout` is **Figure 5** and `fig5_drive` is **Figure
6** in the paper.

The Table S1 scan sweeps 100 parameter combinations, so it uses a vectorised
batch version of the slow dynamics (Equations 13 and 14) in
`tables/burnout_core_v1nu.py`. This batch version is checked against the
single-trajectory `clear/burnout_core.py` used by Figure 5, by
`checks/check_burnout_equivalence.py` (see *Checks*, below); the two agree to
floating-point precision.

The shared model code is in `clear/`:

- `clear/clear_core.py` — the fast loop of Section 3.7 (effective capacity, the
  control authority *k*, engagement *s*, and the load and affect dynamics).
  `REF` holds the reference parameter set and `folds(p)` returns the collapse and
  recovery thresholds.
- `clear/burnout_core.py` — the slow capacity-ceiling layer of Section 3.1
  (Equations 13 and 14), used by Figure 5 (`fig6_burnout`).
- `clear/style.py` — the shared plotting style (see `FIGURE_SPEC.md`).

## Reference values

The **reference parameter set** (the person used in Figures 3 to 6, and the
centre of the maps in Figure 2) is:

> anticipated reward *R* = 2, affect gain *g* = 2.5, engagement exponent *n* = 4,
> maximum capacity *C* = 1, effort *E* = 1, crossover *k\** = 1, passive leaks
> *c0L* = *c0A* = 0.1, affect clearance modulation *κ* = 1. The slow affect state
> and the slow capacity ceiling are off unless a figure says otherwise.

For Figure 5 (burnout) the slow layer uses η0 = 1/400, δ = 1.2×10⁻³, θ = 0.1,
κ = 1×10⁻³ and ν = 1/2000.

The two **stochastic** figures use fixed random seeds, so they reproduce exactly:

- **Figure 4** (`fig4_proximity`): noise σ = 0.02, step dt = 0.02. The figure
  itself uses **seed 1**; the six-seed range quoted in Section 5.3 uses seeds
  **1, 2, 3, 7, 11, 23**.
- **Figure S2** (`figS4_proximity_stats`): σ = 0.02, dt = 0.02, four seeds per
  cell. The functioning branch uses seeds **11, 111, 211, 311** and the
  collapsed branch **12, 112, 212, 312** (base seeds 11 and 12, plus 100 per
  repeat).

All other figures and Table S1 are deterministic.

## Checks

Three kinds of check are included, and `run_all.py` gates on all of them.

- **`paper_numbers.py`** recomputes every implementation number the paper prints
  — the collapse and recovery thresholds, the 3.7 decomposition, the Figure 4
  collapse demands, the Figure S2 responses, the burnout settled ceilings, the
  break-even and collapse prices of the drive figures, and the Table S1 counts —
  and compares each with the paper at the precision the paper prints it. Every
  comparison is a strict PASS or FAIL at that precision; the paper text and the
  code agree, so they all pass.

- **The equivalence check** `checks/check_burnout_equivalence.py` confirms that
  the vectorised batch harness used for Table S1
  (`tables/burnout_core_v1nu.py`) and the single-trajectory slow layer used for
  Figure 5 (`clear/burnout_core.py`) implement the same Equations 13 and 14: it
  runs the reference burnout scenario through both and checks that the capacity
  ceiling agrees period by period to floating-point precision.

- **The independent re-implementation checks** in `checks/` re-implement the
  model a second time, from the equations in the paper's prose rather than from
  `clear/`, and with a different integrator (SciPy's adaptive LSODA, or a
  fixed-step RK4, in place of the figures' forward Euler). Agreement between two
  independent implementations is evidence that the numbers are properties of the
  model, not of one piece of code. `indep_check_fig5.py`, `indep_check_fig6.py`,
  `indep_check_figS5.py` and `indep_check_robust.py` each print PASS when they
  agree with the regenerated numbers to their stated tolerances.

## Licence and how to cite

This code is released under the **MIT Licence** (see `LICENSE`), copyright 2026
Craig E. McAllister.

Please cite both the paper and this code. Citation metadata for the software is
in `CITATION.cff`. In short:

> McAllister, C. E., & Sonuga-Barke, E. J. S. (2026). *Extending our
> understanding of the dynamic, state- and context-dependent nature of ADHD: An
> integrative control-theoretic model* [Manuscript submitted for publication].

> McAllister, C. E. (2026). *CLEAR: Simulation code for an integrative
> control-theoretic model of ADHD* (Version 1.0.0) [Computer software].
