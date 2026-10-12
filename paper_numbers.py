"""
paper_numbers.py -- every implementation number the CLEAR paper prints, checked
against the code.

Each check recomputes a quantity (from clear_core, or from a regenerated
<stem>_numbers.json, or from the Table S1 scan) and compares it with the value
the paper prints, at the precision the paper prints it. It reports PASS or FAIL
with both values, and a final count.

Every check is a strict PASS or FAIL at the paper's printed precision; there are
no tolerated "known" rounding items (the paper text was reconciled with the code
in revision 12m, so the values now agree).

Run after run_all.py has regenerated output/ (so the JSON files and
output/scan_robust.npz exist). It also runs stand-alone: it will fall back to
expected/ for the JSON files, and skip the Table S1 counts if the scan .npz is
absent.

Numbers are in arbitrary units; nothing here is a claim about a person.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "clear"))
import clear_core as cc

OUT = os.environ.get("CLEAR_FIG_OUT", os.path.join(HERE, "output"))
EXP = os.path.join(HERE, "expected")


def load_numbers(stem):
    """The regenerated numbers if present, else the reference values."""
    for d in (OUT, EXP):
        p = os.path.join(d, stem + "_numbers.json")
        if os.path.exists(p):
            return json.load(open(p)), d
    raise FileNotFoundError(stem)


# ---------------------------------------------------------------- comparison
_results = []     # (ok, label, code_val, paper_val, detail)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def check(label, code, paper, tol=None, prec=None):
    """Compare at the paper's printed precision (prec decimals) or within an
    absolute tolerance (tol). Exactly one of tol/prec should be given; default
    is exact equality."""
    if tol is not None:
        ok = abs(code - paper) <= tol
        detail = f"|Delta|={abs(code - paper):.4g} <= {tol}"
    elif prec is not None:
        ok = round(code, prec) == round(paper, prec)
        detail = f"rounded to {prec} dp: {round(code, prec)} vs {round(paper, prec)}"
    else:
        ok = code == paper
        detail = ""
    _results.append((ok, label, code, paper, detail))
    return ok


# ---------------------------------------------------------------- 3.7 decomposition
REF = dict(cc.REF)
u_up, u_rec = cc.folds(REF)
# functioning state at (just below) the collapse threshold
e = max(cc.stable_states(u_up * 0.999999, REF, N=6001), key=lambda q: q["s"])
L_thr, A_thr = e["L"], e["A"]
ceff_thr = (REF["C"] - L_thr) / REF["C"]
k_thr = cc.k_of(L_thr, A_thr, 0.0, REF)

# 3.7, the regimes
check("3.7 two-state region, lower edge (recovery)", u_rec, 0.0085, prec=4)
check("3.7 two-state region, upper edge (collapse)", u_up, 0.144, prec=3)
check("3.7 window factor (u_up/u_rec ~ seventeen)", u_up / u_rec, 17.0, prec=0)
check("3.7 load at the collapse threshold (0.16)", L_thr, 0.16, prec=2)

# 3.7, the decomposition (joint runaway)
check("3.7 effective capacity at collapse threshold (0.84)", ceff_thr, 0.84, prec=2)
check("3.7 affect at collapse threshold (0.23)", A_thr, 0.23, prec=2)
check("3.7 control authority k at collapse threshold (1.36)", k_thr, 1.36, prec=2)

u_up_g0, u_rec_g0 = cc.folds(dict(REF, g=0.0))
check("3.7 g=0, two states lower edge (0.09)", u_rec_g0, 0.09, prec=2)
check("3.7 g=0, two states upper edge (0.31)", u_up_g0, 0.31, prec=2)

# capacity not eroded by load == effective capacity held at 1 (the C -> infinity limit)
u_up_noerode, _ = cc.folds(dict(REF, C=1e6), lo=0.001, hi=2.0)
check("3.7 capacity not eroded, collapse only at 0.87", u_up_noerode, 0.87, prec=2)

# affect loop lowers the recovery threshold from 0.09 (g=0) to 0.0085 (reference)
check("3.7 affect loop lowers recovery threshold: g=0 value (0.09)", u_rec_g0, 0.09, prec=2)
check("3.7 affect loop lowers recovery threshold: reference value (0.0085)", u_rec, 0.0085, prec=4)

# 3.7 and Figure S1, the steepness
r_n1 = (lambda ab: ab[0] / ab[1])(cc.folds(dict(REF, n=1)))
r_n4 = (lambda ab: ab[0] / ab[1])(cc.folds(dict(REF, n=4)))
check("3.7/S1 threshold ratio at n=1 (1.24)", r_n1, 1.24, prec=2)
check("3.7/S1 threshold ratio at n=4 (17)", r_n4, 17.0, prec=0)


def _first_n(predicate, ns):
    return next((float(n) for n in ns if predicate(float(n))), None)


_ns = np.round(np.arange(1.0, 2.51, 0.01), 2)


def _coexist(n):
    a, b = cc.folds(dict(REF, n=n))
    if not (a and b):
        return False
    for u in np.linspace(b, a, 40):
        ss = cc.stable_states(u, dict(REF, n=n), N=2001)
        if any(x["s"] > 0.5 for x in ss) and any(x["s"] < 0.5 for x in ss):
            return True
    return False


def _up_to_threshold(n):
    a, b = cc.folds(dict(REF, n=n))
    if not a:
        return False
    ss = cc.stable_states(a * 0.999, dict(REF, n=n), N=4001)
    return any(x["s"] > 0.5 for x in ss) and any(x["s"] < 0.5 for x in ss)


n_coexist = _first_n(_coexist, _ns)
n_upto = _first_n(_up_to_threshold, _ns)
check("3.7/S1 functioning coexists with collapsed from n = 1.1", n_coexist, 1.1, prec=1)
check("3.7/S1 functioning up to the collapse threshold from n = 1.7", n_upto, 1.7, prec=1)

# 5.5 dose asymmetry
check("5.5 collapse threshold at R=2 (0.144)", cc.folds(dict(REF, R=2.0))[0], 0.144, prec=3)
check("5.5 collapse threshold at R=3 (0.35)", cc.folds(dict(REF, R=3.0))[0], 0.35, prec=2)
check("5.5 passive-clearance level c0*C (0.10)", REF["c0L"] * REF["C"], 0.10, prec=2)

# ---------------------------------------------------------------- Figure 4 (5.3)
f4, _ = load_numbers("fig4")
check("Fig4 collapse demand, seed 1 (0.138)", f4["u_at_collapse"], 0.138, prec=3)
seed_u = [v["u_at_collapse"] for v in f4["seeds"].values()]
check("Fig4 collapse demand, lowest across six seeds (0.129)", min(seed_u), 0.129, prec=3)
check("Fig4 collapse demand, highest across six seeds (0.144)", max(seed_u), 0.144, prec=3)

# ---------------------------------------------------------------- Figure S2
fS4, _ = load_numbers("figS4")


def _nearest(d, target):
    k = min(d, key=lambda kk: abs(float(kk) - target))
    return d[k]


check("FigS2 response at demand 0.14 (0.81)", _nearest(fS4["sens_upper_load"], 0.14), 0.81, prec=2)
check("FigS2 response at demand 0.0092 (0.93)", _nearest(fS4["sens_lower_load"], 0.0092), 0.93, prec=2)

# ---------------------------------------------------------------- Figure 5 (5.4): burnout
f6, _ = load_numbers("fig6")
slow = f6["slow"]
check("Fig5 slow parameter eta0 = 1/400", slow["eta"], 1.0 / 400.0, tol=1e-12)
check("Fig5 slow parameter delta = 1.2e-3", slow["delta"], 1.2e-3, tol=1e-12)
check("Fig5 slow parameter theta = 0.1", slow["theta"], 0.1, tol=1e-12)
check("Fig5 slow parameter kappa = 1e-3", slow["kappa"], 1.0e-3, tol=1e-12)
check("Fig5 slow parameter nu = 1/2000", slow["nu"], 1.0 / 2000.0, tol=1e-12)
check("Fig5 settled ceiling, peak 0.18 typical (0.77)",
      f6["rows"]["a"]["settled"]["C_after_relief"], 0.77, prec=2)
check("Fig5 settled ceiling, peak 0.25 typical (0.57)",
      f6["rows"]["b"]["settled"]["C_after_relief"], 0.57, prec=2)
check("Fig5 settled ceiling, peak 0.18 half rate (0.42)",
      f6["rows"]["c"]["settled"]["C_after_relief"], 0.42, prec=2)

# ---------------------------------------------------------------- Figure 6 and S3 (5.5): drive
fS5, _ = load_numbers("figS5")
lv = fS5["levels"]
be_paper = {"far from the boundary": 0.57, "nearer": 0.59, "beside it": 0.65}
cp_paper = {"far from the boundary": 1.56, "nearer": 1.13, "beside it": 0.78}
for name, bev in be_paper.items():
    check(f"Fig6/S3 break-even price, {name} ({bev})", lv[name]["break_even_closed_form"], bev, prec=2)
for name, cpv in cp_paper.items():
    check(f"FigS3 collapse price, {name} ({cpv})", lv[name]["collapse_price_grid"], cpv, prec=2)
check("Fig6 drive with no affective price raises long-run engagement to 0.98",
      min(lv[n]["s_drive_no_price"] for n in lv), 0.98, prec=2)

# ---------------------------------------------------------------- Table S1
npz = os.path.join(OUT, "scan_robust.npz")
if os.path.exists(npz):
    d = np.load(npz)
    C_set, share, out = d["C_set"], d["share"], d["out"]
    tolC, tolS = 1e-4, 0.005
    a, b, c, dd = (C_set[..., i] for i in range(4))
    sa, sb, sc, sd = (share[..., i] for i in range(4))
    demand_ok = (c < b - tolC) & (b < a - tolC) & (sc <= sb + tolS) & (sb <= sa + tolS)
    repair_ok = (dd < a - tolC) & (sd <= sa + tolS)
    settles = np.all(out == 1, axis=-1)
    check("TableS1 demand direction holds (100/100)", int(demand_ok.sum()), 100)
    check("TableS1 repair direction holds (100/100)", int(repair_ok.sum()), 100)
    check("TableS1 scenarios level off with rest restoring (97/100)", int(settles.sum()), 97)
    # reference combination (delta x1, kappa x1, theta 0.10, peak 0.18 typical) within 0.01 of Figure 5
    i = int(np.argmin(abs(d["DF"] - 1.0))); j = int(np.argmin(abs(d["KF"] - 1.0)))
    kk = int(np.argmin(abs(d["TH"] - 0.10)))
    C_ref = float(C_set[i, j, kk, 0])
    check("TableS1 reference combination within 0.01 of Figure 5 ceiling",
          abs(C_ref - f6["rows"]["a"]["settled"]["C_after_relief"]), 0.0, tol=0.01)
else:
    print("NOTE: output/scan_robust.npz not found; skipping Table S1 counts "
          "(run tables/table_S1_robustness.py, or run_all.py, first).\n")

# ---------------------------------------------------------------- report
print("=" * 78)
print("paper_numbers.py -- CLEAR implementation numbers vs the paper")
print("=" * 78)
nfail = 0
for ok, label, code, paper, detail in _results:
    tag = "PASS" if ok else "FAIL"
    if not ok:
        nfail += 1
    line = f"[{tag}] {label}\n        code={_fmt(code)}  paper={_fmt(paper)}"
    if detail:
        line += f"   ({detail})"
    print(line)
print("-" * 78)
npass = sum(1 for r in _results if r[0])
print(f"{npass}/{len(_results)} checks PASS; {nfail} FAIL (strict, at the paper's printed precision).")
sys.exit(1 if nfail else 0)
