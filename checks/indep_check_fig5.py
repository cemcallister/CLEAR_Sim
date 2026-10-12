"""
Independent check of fig5_numbers.json (Figure 6 in the paper).

Written without clear_core: its own right-hand side from the equations in the
paper's Section 3.7, its own starting states (integrated to rest from zero load
at fixed demand, which lands on the functioning branch), and an adaptive
stiff solver (scipy LSODA, tight tolerances) in place of the figure's fixed-step
Euler. Passes if every checked quantity agrees within the tolerances below. The starting
state and the immediate peak agree to rounding because both are algebraic at the
same fixed point; the transients are where the two integrators differ.
"""
import json, os
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.environ.get("CLEAR_FIG_OUT", os.path.join(HERE, "..", "output"))
_json = [os.path.join(_OUT, "fig5_numbers.json"), os.path.join(HERE, "..", "expected", "fig5_numbers.json")]
REF = json.load(open(next(p for p in _json if os.path.exists(p))))   # regenerated output, or the reference values

R, E, C, N, KSTAR = 2.0, 1.0, 1.0, 4, 1.0
G, C0L, C0A, KAPPA, GATE = 2.5, 0.10, 0.10, 1.0, 0.3
RV, BETA = 1.0, 1.0


def engagement(L, A, Rtot, Eff):
    k = Rtot * max(C - L, 0.0) / C / (Eff + A)
    kn = max(k, 0.0) ** N
    return kn / (KSTAR ** N + kn)


def authority(L, A, Rtot, Eff):
    return Rtot * max(C - L, 0.0) / C / (Eff + A)


def make_rhs(u, Rtot, Eff, beta_rv):
    def f(t, y):
        L, A = max(y[0], 0.0), max(y[1], 0.0)
        s = engagement(L, A, Rtot, Eff)
        dL = u - (C0L + s) * L
        dA = G * (1 - s) * L / (L + GATE) - (C0A + KAPPA * s) * A + beta_rv * s
        return [dL, dA]
    return f


def rest_state(u):
    sol = solve_ivp(make_rhs(u, R, E, 0.0), (0, 4000), [0.0, 0.0], method="LSODA", rtol=1e-10, atol=1e-12)
    L, A = sol.y[:, -1]
    return L, A, engagement(L, A, R, E)


def after(u, kind, T=800.0):
    L0, A0, s0 = rest_state(u)
    if kind == "drive":
        f = make_rhs(u, R + RV, E, BETA * RV)
    else:
        Ep = R * (E + A0) / (R + RV) - A0          # same immediate gain in k as the drive arm
        f = make_rhs(u, R, Ep, 0.0)
        eff = Ep
    sol = solve_ivp(f, (0, T), [L0, A0], method="LSODA", rtol=1e-9, atol=1e-11, dense_output=True, max_step=0.05)
    tt = np.linspace(0, T, 160001)
    Y = sol.sol(tt)
    Rt, Et = (R + RV, E) if kind == "drive" else (R, eff)
    s = np.array([engagement(max(a, 0), max(b, 0), Rt, Et) for a, b in zip(Y[0], Y[1])])
    i50 = np.searchsorted(tt, 50.0)
    below = np.where(s < 0.5)[0]
    Lend, Aend = max(Y[0][-1], 0.0), max(Y[1][-1], 0.0)
    return dict(s_start=s0, s_peak=s[tt <= 20].max(), s_at_50=s[i50], s_long_run=s[-1],
                A_long_run=Aend, t_below_half=(tt[below[0]] if len(below) else None),
                k_start=authority(L0, A0, R, E), k_immediate=authority(L0, A0, Rt, Et),
                k_long_run=authority(Lend, Aend, Rt, Et))


if __name__ == "__main__":
    tol = dict(s_start=2e-3, s_peak=5e-3, s_at_50=5e-3, s_long_run=5e-3, A_long_run=0.05, t_below_half=0.2,
               k_start=2e-3, k_immediate=2e-3, k_long_run=5e-3)
    cases = [("a", "drive", REF["a"]["u"], REF["a"]["drive"]), ("a", "friction", REF["a"]["u"], REF["a"]["friction"])]
    cases += [("b", name, v["u"], v) for name, v in REF["b"].items()]
    worst, ok = {}, True
    for panel, name, u, ref in cases:
        kind = "friction" if name == "friction" else "drive"
        got = after(u, kind)
        line = []
        for q, t in tol.items():
            a, b = got[q], ref[q]
            if a is None or b is None:
                good = (a is None) == (b is None); d = 0.0 if good else np.inf
            else:
                d = abs(a - b); good = d <= t
            worst[q] = max(worst.get(q, 0.0), d); ok &= good
            line.append(f"{q} {a if a is None else round(a, 4)} vs {b if b is None else round(b, 4)}{'' if good else ' FAIL'}")
        print(f"({panel}) {name:22s} u={u:.4f}: " + "; ".join(line))
    print("largest differences:", ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    print("PASS" if ok else "FAIL")
