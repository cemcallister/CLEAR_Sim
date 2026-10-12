"""Independent check of the robustness scan at its corners: RK4 at dt = 0.05
(the scan uses forward Euler at 0.02), fast loop from indep_fast_core, slow
layer written from the burnout_core docstring. Compares the ceiling after
relief and the functioning share at period 40 with scan_robust.npz."""
import numpy as np, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from indep_slow_harness import run


def slow(L, A, s, C, eta, u, par):
    occ = 1.0 - np.clip(C - L, 0.0, None) / C
    over = np.clip(occ - par["theta"], 0.0, None)
    dC = eta * (1.0 - C) - par["delta"] * over * C
    deta = par["nu"] * (par["eta_base"] - eta) - par["kappa"] * over * eta
    return dC, deta


CORNERS = [(4.0, 4.0, 0.05), (0.25, 0.25, 0.5), (4.0, 0.25, 0.5), (0.25, 4.0, 0.05)]
SCEN_IDX = [2, 3]   # (0.25, typical) and (0.18, half) in scan_robust.SCEN
P_CHECK = 40

if __name__ == "__main__":
    d = np.load(os.environ.get("SCAN_ROBUST", os.path.join(_HERE, "..", "output", "scan_robust.npz")))
    DF, KF, TH, SCEN = d["DF"], d["KF"], d["TH"], d["SCEN"]
    cells = [(c, q) for c in CORNERS for q in SCEN_IDX]
    df = np.array([c[0] for c, q in cells]); kf = np.array([c[1] for c, q in cells]); th = np.array([c[2] for c, q in cells])
    peak = np.array([SCEN[q][0] for c, q in cells]); fe = np.array([SCEN[q][1] for c, q in cells])
    par = dict(u0=0.4 * peak, u1=peak, Td=600.0, Tr=150.0, u_relief=0.003, eta0=fe / 400.0, eta_base=fe / 400.0,
               nu=np.full(len(cells), 1 / 2000.0), delta=1.2e-3 * df, kappa=1.0e-3 * kf, theta=th, Cmin=0.02)
    n = len(cells)
    X0 = np.array([np.full(n, 0.003 / 1.1), np.zeros(n), np.ones(n), fe / 400.0])
    t, R = run(slow, par, P_CHECK, dt=0.05, X0=X0, record_every=10)
    C, s = R[:, 2, :], R[:, 4, :]
    worstC = worstS = 0.0
    for b, (c, q) in enumerate(cells):
        i = int(np.argmin(abs(DF - c[0]))); j = int(np.argmin(abs(KF - c[1]))); k = int(np.argmin(abs(TH - c[2])))
        p = P_CHECK - 1
        m = (t >= p * 750) & (t < p * 750 + 600); tt = t[m] - p * 750; ss = s[m, b]
        share = float(tt[np.argmax(ss < 0.5)] / 600) if (ss < 0.5).any() else 1.0
        mh = (t >= (p + 1) * 750 - 1.0) & (t < (p + 1) * 750)
        Ci = float(C[mh, b][-1])
        Cm = float(d["C_after_hol"][p, i, j, k, q]); sm = d["coll_frac"][p, i, j, k, q]
        sm = 1.0 if np.isnan(sm) else float(sm)
        worstC = max(worstC, abs(Ci - Cm)); worstS = max(worstS, abs(share - sm))
        print(f"delta x{c[0]:g} kappa x{c[1]:g} theta {c[2]:g} scenario {tuple(SCEN[q])}: "
              f"C {Ci:.4f} vs {Cm:.4f}, functioning {share:.3f} vs {sm:.3f}")
    print(f"largest differences: ceiling {worstC:.4f}, functioning share {worstS:.4f}")
    print("PASS" if worstC < 0.005 and worstS < 0.01 else "FAIL")
