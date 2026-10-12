"""
CLEAR fast loop: one implementation used by every main-text figure.

Model (all dimensionless; reference set REF below):

    C_eff = (C - L)^+ / C                      paper eq:ceff, absolute load units
    k     = R * C_eff / (E + A + A_b)          paper eq:gain with a slow affect state
    s     = k^n / (k*^n + k^n)                 paper dominance weight at n = 1, Hill above
    dL/dt = u - (c0L + s) L                    engagement-gated clearance + passive leak
    dA/dt = g (1 - s) f(L) - (c0A + kappa s) A + beta_v R_v s
            f(L) = L / (L + 0.3)               demand gate
    dA_b/dt = eps (alpha (1 - s) f(L) - A_b)   slow layer (off unless stated)

The last term of dA/dt is the avoidance-drive coupling used only in the drive
figure (beta_v = 0 everywhere else): drive R_v enters R, and generates affect
in proportion to the engagement it buys.

Equilibria are found by a scalar reduction in s (no multi-start solver):
L(s) = u/(c0L+s), A(s) = [g(1-s)f(L) + beta_v R_v s]/(c0A + kappa s), and s
must equal s_of(k(L(s), A(s))). Roots are sign changes of an increasing map,
so none can be missed at the grid resolution and none can be spurious.
Stability from the 2x2 Jacobian of the fast (L, A) system.
"""
import numpy as np

REF = dict(R=2.0, kstar=1.0, n=4, kappa=1.0, c0L=0.10, c0A=0.10, g=2.5,
           eps=0.02, alpha=2.0, E=1.0, C=1.0, Rv=0.0, beta_v=0.0)

FDEM_K = 0.3


def s_of(k, p):
    k = np.maximum(np.asarray(k, dtype=float), 0.0)
    kn = k ** p["n"]
    return kn / (p["kstar"] ** p["n"] + kn)


def k_of(L, A, Ab, p):
    ceff = np.maximum(p["C"] - L, 0.0) / p["C"]
    return (p["R"] + p["Rv"]) * ceff / (p["E"] + A + Ab)


def rhs(L, A, Ab, u, p, slow=False):
    k = k_of(L, A, Ab, p)
    s = s_of(k, p)
    f = L / (L + FDEM_K)
    dL = u - (p["c0L"] + s) * L
    dA = p["g"] * (1.0 - s) * f - (p["c0A"] + p["kappa"] * s) * A + p["beta_v"] * p["Rv"] * s
    dAb = p["eps"] * (p["alpha"] * (1.0 - s) * f - Ab) if slow else 0.0 * Ab
    return dL, dA, dAb, s


# ---------------------------------------------------------------- equilibria

def _A_of_s(s, L, p):
    f = L / (L + FDEM_K)
    return (p["g"] * (1.0 - s) * f + p["beta_v"] * p["Rv"] * s) / (p["c0A"] + p["kappa"] * s)


def equilibria(u, p, N=4001, Ab=0.0):
    ss = np.linspace(0.0, 1.0 - 1e-9, N)
    L = u / (p["c0L"] + ss)
    A = _A_of_s(ss, L, p)
    ceff = np.maximum(p["C"] - L, 0.0) / p["C"]
    k = (p["R"] + p["Rv"]) * ceff / (p["E"] + A + Ab)
    G = s_of(k, p) - ss
    roots = []
    if G[0] == 0.0:                      # boundary root: L(0) >= C, s = 0 exactly
        roots.append(0.0)
    for i in range(N - 1):
        if G[i] * G[i + 1] < 0:
            lo, hi = ss[i], ss[i + 1]
            def Gf(s):
                Ls = u / (p["c0L"] + s)
                As = _A_of_s(s, Ls, p)
                kk = (p["R"] + p["Rv"]) * max(p["C"] - Ls, 0.0) / p["C"] / (p["E"] + As + Ab)
                return float(s_of(kk, p)) - s
            glo = Gf(lo)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                gm = Gf(mid)
                if gm * glo <= 0:
                    hi = mid
                else:
                    lo, glo = mid, gm
            roots.append(0.5 * (lo + hi))
    out = []
    for s in roots:
        Ls = u / (p["c0L"] + s)
        As = _A_of_s(s, Ls, p)
        h = 1e-7
        f0 = np.array(rhs(Ls, As, Ab, u, p)[:2], dtype=float)
        J = np.zeros((2, 2))
        for j, (dl, da) in enumerate([(h, 0.0), (0.0, h)]):
            f1 = np.array(rhs(Ls + dl, As + da, Ab, u, p)[:2], dtype=float)
            J[:, j] = (f1 - f0) / h
        ev = np.linalg.eigvals(J)
        stable = bool(np.all(ev.real < 0)) if s > 0 else True   # boundary root is attracting
        out.append(dict(s=float(s), L=float(Ls), A=float(As), stable=stable, ev=ev))
    return out


def stable_states(u, p, **kw):
    return [e for e in equilibria(u, p, **kw) if e["stable"]]


def branches(us, p, **kw):
    """Upper and lower stable branches (NaN where absent)."""
    up = np.full(len(us), np.nan); lo = np.full(len(us), np.nan)
    upL = np.full(len(us), np.nan); loL = np.full(len(us), np.nan)
    upA = np.full(len(us), np.nan); loA = np.full(len(us), np.nan)
    for i, u in enumerate(us):
        st = stable_states(u, p, **kw)
        if not st:
            continue
        hi = max(st, key=lambda e: e["s"]); lw = min(st, key=lambda e: e["s"])
        up[i], upL[i], upA[i] = hi["s"], hi["L"], hi["A"]
        if len(st) >= 2:
            lo[i], loL[i], loA[i] = lw["s"], lw["L"], lw["A"]
    return dict(up=up, lo=lo, upL=upL, loL=loL, upA=upA, loA=loA)


def n_stable(u, p, **kw):
    return len(stable_states(u, p, **kw))


def folds(p, lo=0.001, hi=1.2, coarse=600, tol=1e-5, **kw):
    """(u_up, u_rec): upper and lower edges of the window with >= 2 stable
    states, refined by bisection. Returns (None, None) if no window."""
    us = np.linspace(lo, hi, coarse)
    flag = np.array([n_stable(u, p, **kw) >= 2 for u in us])
    if not flag.any():
        return None, None
    i_hi = np.where(flag)[0].max(); i_lo = np.where(flag)[0].min()
    def refine(a, b, want_true_at_a):
        # bisect between a (flag True) and b (flag False)
        for _ in range(40):
            m = 0.5 * (a + b)
            if n_stable(m, p, **kw) >= 2:
                a = m
            else:
                b = m
            if abs(b - a) < tol:
                break
        return a
    u_up = refine(us[i_hi], us[min(i_hi + 1, coarse - 1)], True) if i_hi < coarse - 1 else us[i_hi]
    u_rec = refine(us[i_lo], us[max(i_lo - 1, 0)], True) if i_lo > 0 else us[i_lo]
    return float(u_up), float(u_rec)


# ---------------------------------------------------------------- integration

def integrate(y0, T, dt, p, u=None, u_fn=None, slow=False, sigma=0.0, seed=None,
              p_fn=None, record=True):
    """Euler (Maruyama if sigma > 0) integration of one trajectory.
    u_fn(t) overrides u; p_fn(t) returns a parameter dict override at t."""
    rng = np.random.default_rng(seed)
    n = int(round(T / dt))
    L, A, Ab = float(y0[0]), float(y0[1]), float(y0[2]) if len(y0) > 2 else 0.0
    t_arr = np.arange(n) * dt
    rec = dict(t=t_arr, L=np.zeros(n), A=np.zeros(n), Ab=np.zeros(n), s=np.zeros(n), u=np.zeros(n))
    sq = np.sqrt(dt)
    for i in range(n):
        t = t_arr[i]
        pp = dict(p, **p_fn(t)) if p_fn is not None else p
        uu = u_fn(t) if u_fn is not None else u
        dL, dA, dAb, s = rhs(L, A, Ab, uu, pp, slow=slow)
        L = L + dt * dL; A = A + dt * dA; Ab = Ab + dt * dAb
        if sigma > 0:
            L += sigma * sq * rng.standard_normal(); A += sigma * sq * rng.standard_normal()
        L = max(L, 0.0); A = max(A, 0.0)
        if record:
            rec["L"][i], rec["A"][i], rec["Ab"][i], rec["s"][i], rec["u"][i] = L, A, Ab, s, uu
    if record:
        return rec
    return L, A, Ab


def integrate_vec(us, init, T, dt, p, sigma=0.0, seed=None, transient=0.0, keep=False,
                  Rs=None, Cs=None):
    """Vectorised over a demand grid (and optionally per-cell R or C).
    Returns final (L, A) and, if keep, the s time series after the transient."""
    rng = np.random.default_rng(seed)
    us = np.asarray(us, dtype=float)
    L = np.full(us.shape, init[0], dtype=float); A = np.full(us.shape, init[1], dtype=float)
    Ab = np.zeros(us.shape)
    pp = dict(p)
    pp["R"] = np.asarray(Rs, dtype=float) if Rs is not None else p["R"]
    pp["C"] = np.asarray(Cs, dtype=float) if Cs is not None else p["C"]
    n = int(round(T / dt)); n0 = int(round(transient / dt)); sq = np.sqrt(dt)
    series = np.zeros((n - n0, len(us))) if keep else None
    for i in range(n):
        dL, dA, _, s = rhs(L, A, Ab, us, pp, slow=False)
        L = L + dt * dL; A = A + dt * dA
        if sigma > 0:
            L = L + sigma * sq * rng.standard_normal(us.shape)
            A = A + sigma * sq * rng.standard_normal(us.shape)
        L = np.maximum(L, 0.0); A = np.maximum(A, 0.0)
        if keep and i >= n0:
            series[i - n0] = s
    return L, A, series


FUN_INIT, COL_INIT = (0.05, 0.05), (1.4, 15.0)


def diverge(LA_a, LA_b, p, tol_s=0.02, tol_L=0.05):
    """Operational bistability: two histories end in different states."""
    sa = s_of(k_of(LA_a[0], LA_a[1], 0.0, p), p); sb = s_of(k_of(LA_b[0], LA_b[1], 0.0, p), p)
    return (np.abs(LA_a[0] - LA_b[0]) > tol_L) | (np.abs(sa - sb) > tol_s)
