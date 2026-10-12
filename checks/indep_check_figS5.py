"""
Independent check of figS5_numbers.json (Figure S3 in the paper).

Written without clear_core: its own right-hand side, starting states found by
integrating to rest from zero load, and scipy LSODA in place of the figure's
fixed-step Euler. It checks the change in long-run engagement at four prices,
the drive-without-price and friction-reduced levels, and locates the break-even
and collapse prices by bisection, which must fall inside the 0.01 grid cell the
figure reports. Needs scipy.
"""
import json, os
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.environ.get("CLEAR_FIG_OUT", os.path.join(HERE, "..", "output"))
_json = [os.path.join(_OUT, "figS5_numbers.json"), os.path.join(HERE, "..", "expected", "figS5_numbers.json")]
REF = json.load(open(next(p for p in _json if os.path.exists(p))))

R, E, C, N = 2.0, 1.0, 1.0, 4
G, C0L, C0A, KAPPA, GATE = 2.5, 0.10, 0.10, 1.0, 0.3
RV, T = 1.0, 800.0


def engagement(L, A, Rtot, Eff):
    k = max(Rtot * max(C - L, 0.0) / C / (Eff + A), 0.0)
    return k ** N / (1.0 + k ** N)


def settle(u, Rtot, Eff, beta_rv, y0, t_end):
    def f(t, y):
        L, A = max(y[0], 0.0), max(y[1], 0.0)
        s = engagement(L, A, Rtot, Eff)
        return [u - (C0L + s) * L, G * (1 - s) * L / (L + GATE) - (C0A + KAPPA * s) * A + beta_rv * s]
    sol = solve_ivp(f, (0, t_end), y0, method="LSODA", rtol=1e-9, atol=1e-11)
    L, A = max(sol.y[0, -1], 0.0), max(sol.y[1, -1], 0.0)
    return L, A, engagement(L, A, Rtot, Eff)


def bisect(fn, lo, hi, it=16):
    """fn(lo) True, fn(hi) False; returns the crossing."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if fn(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    ok, lines = True, []
    for name, ref in REF["levels"].items():
        u = ref["u"]
        L0, A0, s0 = settle(u, R, E, 0.0, [0.0, 0.0], 4000.0)
        s_end = lambda b: settle(u, R + RV, E, b * RV, [L0, A0], T)[2]
        change = {q: s_end(float(q)) - s0 for q in ref["change_at_price"]}
        Ep = R * (E + A0) / (R + RV) - A0
        s_fr = settle(u, R, Ep, 0.0, [L0, A0], T)[2]
        be = bisect(lambda b: s_end(b) - s0 >= 0.0, 0.3, 1.0)
        cp = bisect(lambda b: s_end(b) >= 0.5, be, 2.0)
        checks = [("s_control", s0, ref["s_control"], 2e-3), ("s_friction_reduced", s_fr, ref["s_friction_reduced"], 5e-3),
                  ("s_drive_no_price", s_end(0.0), ref["s_drive_no_price"], 5e-3)]
        checks += [(f"change at {q}", change[q], ref["change_at_price"][q], 5e-3) for q in ref["change_at_price"]]
        for q, a, b, tol in checks:
            good = abs(a - b) <= tol; ok &= good
            lines.append(f"{name:22s} {q:20s} {a:8.4f} vs {b:8.4f}{'' if good else '  FAIL'}")
        for q, x, g in (("break-even", be, ref["break_even_grid"]), ("collapse price", cp, ref["collapse_price_grid"])):
            good = (g - 0.01 - 1e-9) < x <= (g + 1e-6); ok &= good
            lines.append(f"{name:22s} {q:20s} {x:8.4f} in grid cell ({g - 0.01:.2f}, {g:.2f}]{'' if good else '  FAIL'}")
        lines.append(f"{name:22s} {'closed-form break-even':20s} {ref['break_even_closed_form']:8.4f} (bisection {be:.4f})")
    print("\n".join(lines))
    print("PASS" if ok else "FAIL")
