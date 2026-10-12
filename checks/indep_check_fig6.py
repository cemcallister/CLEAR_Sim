"""Independent check of fig6_numbers.json (Figure 5 in the paper). The fast loop comes from
indep_fast_core (written from the paper's equations in Section 3.7), the slow
layer is written from the burnout_core docstring, and the integrator is RK4 at
dt = 0.05 (the figure uses forward Euler at 0.02). Compares, per period, the
ceiling after relief, the share of the period spent functioning before
collapse, and the repair rate relative to typical, against fig6_numbers.json.
Reads fig6_numbers.json from output/ (or the folder named in CLEAR_FIG_OUT)."""
import numpy as np, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_OUT = os.environ.get("CLEAR_FIG_OUT", os.path.join(HERE, "..", "output"))
from indep_slow_harness import run

ETA_T = 1.0 / 400.0


def slow(L, A, s, C, eta, u, par):
    occ = 1.0 - np.clip(C - L, 0.0, None) / C
    over = np.clip(occ - 0.10, 0.0, None)
    dC = eta * (1.0 - C) - 1.2e-3 * over * C
    deta = (1.0 / 2000.0) * (par["eta_base"] - eta) - 1.0e-3 * over * eta
    return dC, deta


ROWS = [("a", 0.18, 1.0), ("b", 0.25, 1.0), ("c", 0.18, 0.5)]
N_PER, T_TERM, PERIOD = 15, 600.0, 750.0

if __name__ == "__main__":
    fig = json.load(open(os.path.join(_OUT, "fig6_numbers.json")))
    peak = np.array([r[1] for r in ROWS]); fe = np.array([r[2] for r in ROWS])
    par = dict(u0=0.4 * peak, u1=peak, Td=T_TERM, Tr=PERIOD - T_TERM, u_relief=0.003,
               eta0=fe * ETA_T, eta_base=fe * ETA_T, Cmin=0.02)
    X0 = np.array([np.full(3, 0.003 / 1.1), np.zeros(3), np.ones(3), fe * ETA_T])
    t, R = run(slow, par, N_PER, dt=0.05, X0=X0, record_every=4)
    C, eta, s = R[:, 2, :], R[:, 3, :], R[:, 4, :]
    worst = dict(C=0.0, share=0.0, repair=0.0)
    lines = []
    for b, (lab, _, _) in enumerate(ROWS):
        ref = fig["rows"][lab]["periods"]
        for k in range(N_PER):
            m = (t >= k * PERIOD) & (t < k * PERIOD + T_TERM)
            tt = t[m] - k * PERIOD; ss = s[m, b]
            share = float(tt[np.argmax(ss < 0.5)] / T_TERM) if (ss < 0.5).any() else 1.0
            mh = (t >= (k + 1) * PERIOD - 0.25) & (t < (k + 1) * PERIOD)
            C_end = float(C[mh, b][-1]); rep = float(eta[mh, b][-1] / ETA_T)
            dC = abs(C_end - ref[k]["C_after_relief"]); dS = abs(share - ref[k]["functioning_share"])
            dR = abs(rep - ref[k]["repair_rel_typical"])
            worst["C"] = max(worst["C"], dC); worst["share"] = max(worst["share"], dS); worst["repair"] = max(worst["repair"], dR)
            lines.append(f"{lab} period {k + 1:2d}: C {C_end:.3f} vs {ref[k]['C_after_relief']:.3f} | "
                         f"functioning {share:.3f} vs {ref[k]['functioning_share']:.3f} | "
                         f"repair {rep:.3f} vs {ref[k]['repair_rel_typical']:.3f}")
    print("\n".join(lines))
    print(f"\nlargest differences over all rows and periods: ceiling {worst['C']:.4f}, "
          f"functioning share {worst['share']:.4f}, repair rate {worst['repair']:.4f}")
    ok = worst["C"] < 0.005 and worst["share"] < 0.01 and worst["repair"] < 0.005
    print("PASS" if ok else "FAIL")
