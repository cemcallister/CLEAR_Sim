"""
Slow capacity dynamics on top of the fast loop (clear_core): the capacity
ceiling of Section 3.1 and Equations 13 and 14, used by Figure 5 in the paper
(figures/fig6_burnout.py).

The ceiling is "restored" by the recovery processes of Section 3.1 at the
restoration rate eta; "recovery" is kept for engagement returning after
collapse.

    occ     = 1 - C_eff                        (fraction of the ceiling taken by load)
    over    = (occ - theta)+
    dC/dt   = eta (C_base - C) - delta * over * C
    deta/dt = nu (eta_base - eta) - kappa * over * eta

Restoration of the ceiling is first order toward its baseline at the
restoration rate eta. Erosion is driven by occupancy of the ceiling above a set point
theta and is proportional to the current ceiling. The restoration rate is
worn by the same overload and returns toward its baseline at rate nu, so a long
enough withdrawal from demand restores both fully. Poorer restoration, the
route through sleep or diet, is a lower baseline eta_base.

Explicit floors: basal demand u_basal > 0 and a capacity floor C_min.
Numerics: forward Euler.
"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clear_core as cc

SLOW = dict(eta=1.0 / 400.0, delta=1.2e-3, theta=0.10, C_base=1.0, C_min=0.02,
            kappa=1.0e-3, nu=1.0 / 2000.0)
U_BASAL = 0.003
ETA_TYPICAL = SLOW["eta"]


def occupancy(L, C):
    return 1.0 - max(C - L, 0.0) / C


def integrate(y0, T, dt, p, u_fn, slow_p=SLOW, slow_affect=False, record_every=1):
    """Euler integration of (L, A, Ab, C, eta). u_fn(t) gives demand; p is the
    fast parameter set (its 'C' entry is overridden by the state). slow_p['eta']
    is the person's baseline restoration rate, which is also the starting value."""
    L, A, Ab, C = map(float, y0)
    eta_base = slow_p["eta"]
    eta = eta_base
    n = int(round(T / dt))
    m = n // record_every
    rec = {k: np.zeros(m) for k in ("t", "L", "A", "C", "s", "u", "occ", "eta")}
    pp = dict(p)
    j = 0
    for i in range(n):
        t = i * dt
        u = u_fn(t)
        pp["C"] = C
        dL, dA, dAb, s = cc.rhs(L, A, Ab, u, pp, slow=slow_affect)
        occ = occupancy(L, C)
        over = max(occ - slow_p["theta"], 0.0)
        dC = eta * (slow_p["C_base"] - C) - slow_p["delta"] * over * C
        deta = slow_p["nu"] * (eta_base - eta) - slow_p["kappa"] * over * eta
        if i % record_every == 0 and j < m:
            rec["t"][j], rec["L"][j], rec["A"][j], rec["C"][j] = t, L, A, C
            rec["s"][j], rec["u"][j], rec["occ"][j], rec["eta"][j] = s, u, occ, eta
            j += 1
        L = max(L + dt * dL, 0.0); A = max(A + dt * dA, 0.0); Ab = max(Ab + dt * dAb, 0.0)
        C = min(max(C + dt * dC, slow_p["C_min"]), slow_p["C_base"])
        eta = max(eta + dt * deta, 0.0)
    return rec


def term_holiday(u_lo, u_hi, T_term, T_hol, u_hol=U_BASAL):
    """Demand rising linearly through each period of demand from u_lo to u_hi
    (work accumulating toward its end), basal demand through each relief."""
    period = T_term + T_hol
    def u_fn(t):
        x = t % period
        return u_lo + (u_hi - u_lo) * x / T_term if x < T_term else u_hol
    return u_fn, period


def threshold_table(p, Cs):
    """Collapse and recovery thresholds as functions of the ceiling (fast loop,
    slow affect layer off), for interpolation in log-log."""
    ups, recs = [], []
    for C in Cs:
        a, b = cc.folds(dict(p, C=float(C)))
        ups.append(a if a else np.nan); recs.append(b if b else np.nan)
    return np.array(ups), np.array(recs)


def interp_loglog(x, xs, ys):
    return np.exp(np.interp(np.log(x), np.log(xs), np.log(ys)))
