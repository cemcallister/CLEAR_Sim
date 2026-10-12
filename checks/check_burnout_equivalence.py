"""
Equivalence check: the Table S1 batch harness vs the Figure 5 slow layer.

Table S1 (tables/table_S1_robustness.py) scans 100 parameter combinations, so
it uses tables/burnout_core_v1nu.py -- a vectorised, batch implementation of the
slow capacity-ceiling dynamics (Equations 13 and 14). Figure 5
(figures/fig6_burnout.py) uses clear/burnout_core.py, a single-trajectory
implementation of the same equations.

This check confirms the two implement the SAME equations: it runs the reference
burnout scenario (peak demand 0.18, typical restoration rate, nu = 1/2000) through
both and compares the capacity ceiling after each relief, period by period. They
must agree to floating-point precision (the two share clear_core.rhs, the same
forward-Euler step and the same clamps, so the only possible difference is
floating-point summation order).

Prints PASS/FAIL and exits non-zero on failure, so run_all.py can gate on it.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "clear"))
sys.path.insert(0, os.path.join(HERE, "..", "tables"))
import clear_core as cc
import burnout_core as bc            # Figure 5: single trajectory
import burnout_core_v1nu as bn       # Table S1: vectorised batch

TOL = 1e-12                          # floating-point agreement
N_PER = 15
DT = 0.02
PEAK, FE = 0.18, 1.0                 # reference scenario: peak 0.18, typical rate
DELTA, KAPPA, THETA, NU = 1.2e-3, 1.0e-3, 0.10, 1.0 / 2000.0
T_TERM, T_HOL = 600.0, 150.0


def core_trajectory():
    """Per-relief ceiling from clear/burnout_core.py (one trajectory)."""
    sp = dict(bc.SLOW, eta=FE * bc.ETA_TYPICAL, nu=NU, delta=DELTA, kappa=KAPPA, theta=THETA)
    P = dict(cc.REF)
    u_fn, period = bc.term_holiday(0.4 * PEAK, PEAK, T_TERM, T_HOL)
    rec = bc.integrate((bc.U_BASAL / 1.1, 0.0, 0.0, 1.0), N_PER * period, DT, P, u_fn,
                       slow_p=sp, slow_affect=False, record_every=1)
    t, C = rec["t"], rec["C"]
    out = []
    for k in range(N_PER):
        mh = (t >= k * period + T_TERM) & (t < (k + 1) * period)
        out.append(float(C[mh][-1]))
    return out


def batch_trajectory():
    """Per-relief ceiling from tables/burnout_core_v1nu.py (batch of one cell)."""
    rec, _ = bn.run_batch(np.array([0.4 * PEAK]), np.array([PEAK]), np.array([FE / 400.0]),
                          NU, N_PER, delta=DELTA, kappa=KAPPA, theta=THETA)
    return [float(rec["C_after_hol"][k, 0]) for k in range(N_PER)]


if __name__ == "__main__":
    core = core_trajectory()
    batch = batch_trajectory()
    worst = 0.0
    print("period   burnout_core (Fig 5)   burnout_core_v1nu (Table S1)        |diff|")
    for k in range(N_PER):
        d = abs(core[k] - batch[k])
        worst = max(worst, d)
        print(f"{k + 1:5d}   {core[k]:.12f}         {batch[k]:.12f}        {d:.2e}")
    print(f"\nmax |difference| in capacity after relief over {N_PER} periods: {worst:.2e}")
    ok = worst <= TOL
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
