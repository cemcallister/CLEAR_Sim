"""Vectorised fixed-step RK4 for the fast loop plus a slow ceiling layer.
Written independently of clear/. The slow-layer equations are passed in as a
function, so each check supplies the form it tests.

Demand schedule: periods of length Td in which u ramps linearly u0 -> u1,
followed by relief of length Tr at u_relief.
"""
import numpy as np
from indep_fast_core import REF


def demand(t, u0, u1, Td, Tr, u_relief):
    T = Td + Tr
    ph = np.mod(t, T)
    ramp = u0 + (u1 - u0) * ph / Td
    return np.where(ph < Td, ramp, u_relief)


def make_rhs(slow, p=REF):
    """slow(L, A, s, C, eta, u, par) -> (dC, deta)."""
    def f(t, X, par):
        L, A, C, eta = X
        u = demand(t, par["u0"], par["u1"], par["Td"], par["Tr"], par["u_relief"])
        head = np.clip(C - L, 0.0, None) / C
        k = p["R"] / (p["E"] + A) * head
        kn = k ** p["n"]
        s = kn / (1.0 + kn)
        dL = u - (p["c0L"] + s) * L
        dA = p["g"] * (1.0 - s) * L / (L + p["Lh"]) - (p["c0A"] + p["kap"] * s) * A
        dC, de = slow(L, A, s, C, eta, u, par)
        return np.array([dL, dA, dC, de]), s, u
    return f


def run(slow, par, n_periods, dt=0.05, X0=None, record_every=10):
    """par values may be arrays (batch). Returns time grid and records."""
    f = make_rhs(slow)
    T = par["Td"] + par["Tr"]
    T = float(np.max(T))
    n = int(round(n_periods * T / dt))
    B = np.broadcast(*[np.asarray(v) for v in par.values()]).shape
    if X0 is None:
        X = np.array([np.full(B, 0.05), np.full(B, 0.05),
                      np.full(B, 1.0), np.asarray(par["eta0"]) * np.ones(B)])
    else:
        X = X0
    rec_t, rec = [], []
    t = 0.0
    for i in range(n):
        k1, s, u = f(t, X, par)
        k2, _, _ = f(t + dt / 2, X + dt / 2 * k1, par)
        k3, _, _ = f(t + dt / 2, X + dt / 2 * k2, par)
        k4, _, _ = f(t + dt, X + dt * k3, par)
        X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        X[2] = np.maximum(X[2], par.get("Cmin", 1e-3))
        t += dt
        if i % record_every == 0:
            rec_t.append(t)
            rec.append(np.vstack([X, s[None] if np.ndim(s) else [s], u * np.ones(B)[None]]))
    return np.array(rec_t), np.array(rec)
