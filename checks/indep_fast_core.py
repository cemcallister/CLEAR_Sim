"""Independent implementation of the CLEAR fast loop (reference set), written
from the equations in the figure prose (Section 3.7), not from clear_core.py.

    k   = R/(E + A) * (C - L)+ / C
    s   = k^n / (1 + k^n)
    dL  = u - (c0L + s) L
    dA  = g (1 - s) L/(L + 0.3) - (c0A + kap s) A

Reference: R=2, g=2.5, n=4, E=1, c0L=c0A=0.1, kap=1. Slow affect layer off.
Used as the second implementation for the burnout scan checks.
"""
import numpy as np

REF = dict(R=2.0, g=2.5, n=4.0, E=1.0, c0L=0.1, c0A=0.1, kap=1.0, Lh=0.3)


def s_of(L, A, C, p=REF):
    head = np.clip(C - L, 0.0, None) / C
    k = p["R"] / (p["E"] + A) * head
    kn = k ** p["n"]
    return kn / (1.0 + kn)


def rhs(L, A, u, C, p=REF):
    s = s_of(L, A, C, p)
    dL = u - (p["c0L"] + s) * L
    dA = p["g"] * (1.0 - s) * L / (L + p["Lh"]) - (p["c0A"] + p["kap"] * s) * A
    return dL, dA, s


def residual(s, u, C, p=REF):
    """Scalar reduction: equilibrium iff residual(s) = 0."""
    L = u / (p["c0L"] + s)
    A = p["g"] * (1.0 - s) * (L / (L + p["Lh"])) / (p["c0A"] + p["kap"] * s)
    return s_of(L, A, C, p) - s


def n_equilibria(u, C, p=REF, m=4001):
    s = np.linspace(0.0, 1.0, m)
    r = residual(s, u, C, p)
    # count sign changes plus exact zeros at s=0 (collapsed state with C_eff=0)
    sc = np.sum(np.sign(r[:-1]) * np.sign(r[1:]) < 0)
    if r[0] == 0.0:
        sc += 1
    return int(sc)


def thresholds(C, p=REF, lo=1e-4, hi=2.0, tol=1e-6):
    """Collapse threshold u_up (largest u with a functioning state) and
    recovery threshold u_rec (smallest u with a collapsed state)."""
    def has_functioning(u):
        s = np.linspace(0.0, 1.0, 4001)
        r = residual(s, u, C, p)
        idx = np.where(np.sign(r[:-1]) * np.sign(r[1:]) < 0)[0]
        # functioning = a stable root with k > 1, i.e. s > 0.5; the upper root
        return any(s[i] > 0.5 for i in idx)

    def has_collapsed(u):
        s = np.linspace(0.0, 1.0, 4001)
        r = residual(s, u, C, p)
        if r[0] == 0.0:
            return True
        idx = np.where(np.sign(r[:-1]) * np.sign(r[1:]) < 0)[0]
        # the collapsed state is the lowest stable root: residual goes + to -
        for i in idx:
            if r[i] > 0 and r[i + 1] < 0 and s[i] < 0.5:
                return True
        return False

    # u_up: bisection on has_functioning (True below, False above)
    a, b = lo, hi
    while b - a > tol:
        m = 0.5 * (a + b)
        if has_functioning(m):
            a = m
        else:
            b = m
    u_up = 0.5 * (a + b)
    # u_rec: bisection on has_collapsed (False below, True above)
    a, b = lo, hi
    while b - a > tol:
        m = 0.5 * (a + b)
        if has_collapsed(m):
            b = m
        else:
            a = m
    u_rec = 0.5 * (a + b)
    return u_up, u_rec


if __name__ == "__main__":
    for C in (1.0, 0.5, 0.2, 0.1):
        up, rec = thresholds(C)
        print(f"C={C:4.2f}  u_up={up:.4f}  u_rec={rec:.4f}")
