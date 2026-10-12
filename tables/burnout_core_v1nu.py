"""
Batch version of the slow ceiling layer (Equations 13 and 14) for the Table S1
scan: the same equations as clear/burnout_core.py, vectorised over parameter
cells (checks/check_burnout_equivalence.py confirms that the two agree). The
repair (restoration) rate is worn by overload and returns slowly toward its
baseline.

    occ   = 1 - C_eff                      (fraction of the ceiling taken by load)
    over  = (occ - theta)+
    dC/dt = eta (C_base - C) - delta * over * C
    deta/dt = nu (eta_base - eta) - kappa * over * eta

Numerics as clear/burnout_core.py: forward Euler at dt = 0.02, the same clamps,
initial state and demand schedule (a linear ramp through each period of demand,
then basal relief). Per-period summaries are recorded on the fly, so long runs
need no full time series.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "clear"))
import numpy as np
import clear_core as cc

U_BASAL = 0.003
BASE = dict(eta_base=1.0 / 400.0, delta=1.2e-3, theta=0.10, C_base=1.0, C_min=0.02,
            kappa=1.0e-3, nu=0.0)
T_TERM, T_HOL = 600.0, 150.0


def run_batch(u_lo, u_hi, eta_base, nu, n_periods, delta=BASE["delta"], kappa=BASE["kappa"],
              theta=BASE["theta"], T_term=T_TERM, T_hol=T_HOL, u_hol=U_BASAL, dt=0.02,
              keep_series=None, series_every=25):
    """All of u_lo, u_hi, eta_base, nu, delta, kappa, theta may be arrays of one shape.
    keep_series: optional boolean mask of cells whose time series to keep."""
    shape = np.broadcast(np.asarray(u_lo), np.asarray(u_hi), np.asarray(eta_base),
                         np.asarray(nu), np.asarray(delta), np.asarray(kappa), np.asarray(theta)).shape
    if shape == ():
        shape = (1,)
    B = lambda x: np.broadcast_to(np.asarray(x, dtype=float), shape).copy()
    u_lo, u_hi, eta_base, nu, delta, kappa, theta = map(B, (u_lo, u_hi, eta_base, nu, delta, kappa, theta))
    p = dict(cc.REF)
    L = np.full(shape, U_BASAL / 1.1); A = np.zeros(shape); Ab = np.zeros(shape)
    C = np.ones(shape); eta = eta_base.copy()
    period = T_term + T_hol
    n_term = int(round(T_term / dt)); n_per = int(round(period / dt))
    n = n_periods * n_per
    rec = dict(C_term_start=np.zeros((n_periods,) + shape), coll_frac=np.full((n_periods,) + shape, np.nan),
               C_term_end=np.zeros((n_periods,) + shape), C_after_hol=np.zeros((n_periods,) + shape),
               s_after_hol=np.zeros((n_periods,) + shape), eta_after_hol=np.zeros((n_periods,) + shape),
               s_min_term=np.ones((n_periods,) + shape))
    series = None
    if keep_series is not None:
        m = np.asarray(keep_series)
        series = dict(t=[], s=[], C=[], eta=[], u=[], L=[])
    for i in range(n):
        k, j = divmod(i, n_per)
        t = i * dt
        x = j * dt
        if j < n_term:
            u = u_lo + (u_hi - u_lo) * x / T_term
        else:
            u = np.full(shape, u_hol)
        if j == 0:
            rec["C_term_start"][k] = C
        p["C"] = C
        dL, dA, dAb, s = cc.rhs(L, A, Ab, u, p, slow=False)
        if j < n_term:
            first = np.isnan(rec["coll_frac"][k]) & (s < 0.5)
            rec["coll_frac"][k][first] = x / T_term
            rec["s_min_term"][k] = np.minimum(rec["s_min_term"][k], s)
        if j == n_term - 1:
            rec["C_term_end"][k] = C
        if j == n_per - 1:
            rec["C_after_hol"][k] = C; rec["s_after_hol"][k] = s; rec["eta_after_hol"][k] = eta
        occ = 1.0 - np.maximum(C - L, 0.0) / C
        over = np.maximum(occ - theta, 0.0)
        dC = eta * (BASE["C_base"] - C) - delta * over * C
        deta = nu * (eta_base - eta) - kappa * over * eta
        if series is not None and i % series_every == 0:
            series["t"].append(t); series["s"].append(s[m]); series["C"].append(C[m].copy())
            series["eta"].append((eta / eta_base)[m]); series["u"].append(np.broadcast_to(u, shape)[m].copy())
            series["L"].append(L[m].copy())
        L = np.maximum(L + dt * dL, 0.0); A = np.maximum(A + dt * dA, 0.0)
        C = np.minimum(np.maximum(C + dt * dC, BASE["C_min"]), BASE["C_base"])
        eta = np.maximum(eta + dt * deta, 0.0)
    if series is not None:
        series = {kk: np.array(v) for kk, v in series.items()}
    return rec, series


def classify(rec, tol=0.005, last=5):
    """Long-run outcome per cell.
    0 never collapses; 1 settles (relief restores, ceiling steady);
    2 still falling at the end (relief restores); 3 relief fails (burnout)."""
    coll = ~np.all(np.isnan(rec["coll_frac"]), axis=0)
    fails = np.any(rec["s_after_hol"] < 0.5, axis=0)
    Cs = rec["C_term_start"]
    drift = Cs[-1] - Cs[-1 - last]
    out = np.where(~coll, 0, np.where(fails, 3, np.where(np.abs(drift) < tol, 1, 2)))
    first_fail = np.where(fails, np.argmax(rec["s_after_hol"] < 0.5, axis=0) + 1, 0)
    return out, first_fail
