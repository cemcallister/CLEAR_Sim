"""
run_all.py -- regenerate every CLEAR figure, table and number, then check them.

What it does, in order:
  1. regenerates the nine figures and Table S1 into output/ (PDF + PNG for the
     figures, scan_robust.npz + scan_robust.log for the table);
  2. compares each regenerated output/<stem>_numbers.json with the reference in
     expected/ (deterministic values to floating-point, the two stochastic
     figures exactly, because their seeds are fixed);
  3. runs paper_numbers.py (every number the paper prints);
  4. runs the independent re-implementation checks.

It prints a summary and exits non-zero if anything that should pass fails.

Everything is written under output/ (overridable with the CLEAR_FIG_OUT
environment variable). output/ is disposable and git-ignored.

Approximate runtimes on a 2020-era laptop (Python 3.12): the figures together
about two to three minutes, Table S1 about two minutes, the checks about a
minute; the whole run is roughly five to six minutes.
"""
import os, sys, time, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CLEAR_FIG_OUT", os.path.join(HERE, "output"))
EXP = os.path.join(HERE, "expected")
PY = sys.executable
os.makedirs(OUT, exist_ok=True)

ENV = dict(os.environ, CLEAR_FIG_OUT=OUT)

FIGURES = [
    ("fig1_boundary", "Figure 1"),
    ("fig2_parameter_map", "Figure 2"),
    ("fig3_hysteresis", "Figure 3"),
    ("fig4_proximity", "Figure 4"),
    ("fig6_burnout", "Figure 5 (fig:burnout)"),
    ("fig5_drive", "Figure 6 (fig:drive)"),
    ("figS3_hysteresis_loop", "Figure S1"),
    ("figS4_proximity_stats", "Figure S2"),
    ("figS5_drive_sweep", "Figure S3"),
]

# stems that write a <stem>_numbers.json (fig1 is a schematic and writes none)
NUMBERS_STEMS = ["fig2", "fig3", "fig4", "fig5", "fig6", "figS3", "figS4", "figS5"]


def run(label, args, logpath=None):
    """Run a subprocess; return (ok, seconds). If logpath, tee stdout to it."""
    print(f"  - {label} ...", end="", flush=True)
    t0 = time.time()
    proc = subprocess.run([PY] + args, cwd=HERE, env=ENV,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    out = proc.stdout.decode("utf-8", "replace")
    if logpath:
        open(logpath, "w").write(out)
    ok = proc.returncode == 0
    print(f" {'ok' if ok else 'FAILED'} ({dt:.1f}s)")
    if not ok:
        print("    --- output ---")
        print("    " + "\n    ".join(out.strip().splitlines()[-25:]))
    return ok, dt, out


def close(a, b, rtol=1e-6, atol=1e-9):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return False
        return all(close(a[k], b[k], rtol, atol) for k in a)
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return False
        return all(close(x, y, rtol, atol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a != a and b != b:   # both NaN
            return True
        if a == b:              # exact, and handles +/- inf
            return True
        return abs(a - b) <= atol + rtol * abs(b)
    return a == b


def tail_is_pass(out):
    lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
    return bool(lines) and lines[-1] == "PASS"


def main():
    t_start = time.time()
    fails = []
    times = {}

    print("[1/4] Regenerating figures and Table S1 into", OUT)
    for stem, paper in FIGURES:
        ok, dt, _ = run(f"{stem}  ->  {paper}", [os.path.join("figures", stem + ".py")])
        times[stem] = dt
        if not ok:
            fails.append(stem)
    ok, dt, _ = run("Table S1 robustness scan  ->  Table S1 (slow, ~3 min)",
                    [os.path.join("tables", "table_S1_robustness.py")],
                    logpath=os.path.join(OUT, "scan_robust.log"))
    times["table_S1"] = dt
    if not ok:
        fails.append("table_S1_robustness")

    print("\n[2/4] Checking regenerated numbers against expected/")
    for stem in NUMBERS_STEMS:
        op = os.path.join(OUT, stem + "_numbers.json")
        ep = os.path.join(EXP, stem + "_numbers.json")
        try:
            match = close(json.load(open(op)), json.load(open(ep)))
        except Exception as ex:          # noqa: BLE001
            match = False
            print(f"  - {stem}_numbers.json: ERROR {ex}")
            fails.append(f"numbers:{stem}")
            continue
        print(f"  - {stem}_numbers.json: {'matches expected' if match else 'DIFFERS from expected'}")
        if not match:
            fails.append(f"numbers:{stem}")

    print("\n[3/4] paper_numbers.py")
    t0 = time.time()
    proc = subprocess.run([PY, "paper_numbers.py"], cwd=HERE, env=ENV)
    times["paper_numbers"] = time.time() - t0
    if proc.returncode != 0:
        fails.append("paper_numbers")

    print("\n[4/4] Independent checks")
    gating = [
        ("check_burnout_equivalence", "Table S1 harness == clear/burnout_core.py"),
        ("indep_check_fig5", "Figure 6 drive arms"),
        ("indep_check_fig6", "Figure 5 burnout"),
        ("indep_check_figS5", "Figure S3 drive sweep"),
        ("indep_check_robust", "Table S1 robustness scan"),
    ]
    for stem, what in gating:
        ok, dt, out = run(f"{stem}  ({what})", [os.path.join("checks", stem + ".py")])
        times[stem] = dt
        if not (ok and tail_is_pass(out)):
            fails.append(stem)

    total = time.time() - t_start
    print("\n" + "=" * 70)
    print("TIMINGS (seconds)")
    for k in ["fig1_boundary", "fig2_parameter_map", "fig3_hysteresis", "fig4_proximity",
              "fig6_burnout", "fig5_drive", "figS3_hysteresis_loop", "figS4_proximity_stats",
              "figS5_drive_sweep", "table_S1", "paper_numbers", "check_burnout_equivalence",
              "indep_check_fig5", "indep_check_fig6", "indep_check_figS5", "indep_check_robust"]:
        if k in times:
            print(f"  {k:28s} {times[k]:7.1f}")
    print(f"  {'TOTAL':28s} {total:7.1f}")
    print("=" * 70)
    if fails:
        print("RESULT: FAIL -", ", ".join(fails))
        return 1
    print("RESULT: PASS - all figures, numbers and gating checks reproduced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
