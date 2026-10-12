"""Table S1: robustness of Figure 5's direction claims to the three slow
parameters that Figure 5 holds fixed.

Varied: delta (ceiling erosion rate) and kappa (repair-rate wear) from a
quarter to four times their reference values, theta (occupancy set point)
from 0.05 to 0.5. Held: nu = 1/2000 and everything else as in Figure 5.
Scenarios per combination: peak demand 0.18, 0.20, 0.25 with typical repair,
and peak 0.18 with repair at half the typical rate. 100 periods of 600 + 150.

Checks per combination:
  demand  : more demand gives a lower settled ceiling and no later collapse
  repair  : poorer repair gives a lower settled ceiling and no later collapse
  settles : every scenario levels off with rest still restoring engagement
Outputs scan_robust.npz, scan_robust.log (printed summary)."""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "clear"))
_OUT = os.environ.get("CLEAR_FIG_OUT", os.path.join(_HERE, "..", "output"))
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import burnout_core_v1nu as bn

DF = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
KF = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
TH = np.array([0.05, 0.10, 0.20, 0.50])
SCEN = [(0.18, 1.0), (0.20, 1.0), (0.25, 1.0), (0.18, 0.5)]
NU = 1.0 / 2000.0
N_PER = 100

if __name__ == "__main__":
    t0 = time.time()
    peak = np.array([s[0] for s in SCEN])[None, None, None, :]
    fe = np.array([s[1] for s in SCEN])[None, None, None, :]
    rec, _ = bn.run_batch(0.4 * peak, peak, fe / 400.0, NU, N_PER,
                          delta=1.2e-3 * DF[:, None, None, None], kappa=1.0e-3 * KF[None, :, None, None],
                          theta=TH[None, None, :, None])
    out, first_fail = bn.classify(rec)
    C_set = rec["C_after_hol"][-1]
    share = np.where(np.isnan(rec["coll_frac"][-1]), 1.0, rec["coll_frac"][-1])
    np.savez(os.path.join(_OUT, "scan_robust.npz"), DF=DF, KF=KF, TH=TH, SCEN=np.array(SCEN), NU=NU,
             out=out, first_fail=first_fail, C_set=C_set, share=share,
             C_after_hol=rec["C_after_hol"], coll_frac=rec["coll_frac"], s_after_hol=rec["s_after_hol"])
    tolC, tolS = 1e-4, 0.005
    a, b, c, d = (C_set[..., i] for i in range(4))
    sa, sb, sc, sd = (share[..., i] for i in range(4))
    demand_ok = (c < b - tolC) & (b < a - tolC) & (sc <= sb + tolS) & (sb <= sa + tolS)
    repair_ok = (d < a - tolC) & (sd <= sa + tolS)
    settles = np.all(out == 1, axis=-1)
    n = demand_ok.size
    print(f"done in {time.time() - t0:.0f}s; {n} parameter combinations x {len(SCEN)} scenarios")
    print(f"demand direction holds in {int(demand_ok.sum())}/{n}")
    print(f"repair direction holds in {int(repair_ok.sum())}/{n}")
    print(f"all scenarios level off with rest restoring in {int(settles.sum())}/{n}")
    code = {0: "never collapses", 1: "levels off", 2: "still falling at period 100", 3: "rest stops restoring"}
    for i, df in enumerate(DF):
        for j, kf in enumerate(KF):
            for k, th in enumerate(TH):
                flags = []
                if not demand_ok[i, j, k]: flags.append("DEMAND")
                if not repair_ok[i, j, k]: flags.append("REPAIR")
                if not settles[i, j, k]:
                    flags.append("OUTCOME " + ", ".join(f"{SCEN[q]}: {code[int(out[i, j, k, q])]}"
                                                         + (f" from period {int(first_fail[i, j, k, q])}" if out[i, j, k, q] == 3 else "")
                                                         for q in range(len(SCEN)) if out[i, j, k, q] != 1))
                if flags:
                    print(f"  delta x{df:g}, kappa x{kf:g}, theta {th:g}: " + "; ".join(flags))
    print("\nsettled ceiling [peak 0.18, 0.20, 0.25 typical repair | 0.18 half repair], functioning share in brackets")
    for i, df in enumerate(DF):
        for j, kf in enumerate(KF):
            row = []
            for k, th in enumerate(TH):
                row.append(f"th{th:g}: " + " ".join(f"{C_set[i, j, k, q]:.2f}({share[i, j, k, q]:.2f})" for q in range(4)))
            print(f"d x{df:<4g} k x{kf:<4g} | " + " | ".join(row))
