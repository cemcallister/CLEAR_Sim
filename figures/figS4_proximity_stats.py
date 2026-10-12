"""
Figure S2 in the paper (stem figS4_proximity_stats): the parameter-space form of Figure 4. Fluctuation and sensitivity against demand.

Fluctuation (a, b): the noisy system held at fixed demand on each branch
(sigma = 0.02 on load and affect, Euler-Maruyama, dt = 0.02, 400 t.u. discarded,
2400 t.u. scored, four seeds per cell, median over the seeds that stayed on the
branch). Variance and lag-25 (0.5 t.u.) autocorrelation of engagement. Cells in
which noise moved any seed off its branch are omitted; (b) shows the functioning
branch only.

Sensitivity (c): the deterministic change in engagement produced by a
proportional shift of demand (+/-10%, larger response shown) applied at each
demand on each branch and integrated to equilibrium. The R shift (+5%) is
computed and written to the JSON but not plotted.
Reference set R = 2, g = 2.5, n = 4, slow layer off.
A row of three panels, matching Figure S1. The JSON also holds the
collapsed-branch statistics and responses.
"""
import numpy as np, sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clear"))
import clear_core as cc
import style as st
import matplotlib.pyplot as plt

P = dict(cc.REF)
_U_UP, _U_REC = cc.folds(P)
# 16 log-spaced demand values, plus one value 8% above the recovery threshold so
# that the collapsed branch has a cell whose -10% demand shift crosses recovery,
# as the top functioning cell's +10% shift crosses collapse.
us = np.sort(np.append(np.geomspace(0.003, 0.14, 16), 1.08 * _U_REC))
SIG, DT, T_TR, T_SC, LAG = 0.02, 0.02, 400.0, 2400.0, 25


NSEED = 4


def fluct(init, seed, expect_hi):
    """Median over NSEED seeds of the runs that stayed on their branch; a cell is
    marked switched if any seed left the expected branch (during the transient
    or the scored window) or flickered between branches."""
    vars_, acs, sw = [], [], []
    for r in range(NSEED):
        L, A, series = cc.integrate_vec(us, init, T_TR + T_SC, DT, P, sigma=SIG, seed=seed + 100 * r,
                                        transient=T_TR, keep=True)
        var = series.var(axis=0)
        x = series - series.mean(axis=0)
        num = (x[:-LAG] * x[LAG:]).mean(axis=0); den = x.var(axis=0)
        ac = np.where(den > 1e-14, num / np.maximum(den, 1e-14), np.nan)
        hi_frac = (series > 0.5).mean(axis=0)
        s_here = (hi_frac > 0.02) & (hi_frac < 0.98)
        # a run that ended on the other branch entirely also counts as switched
        end_hi = series[-5:].mean(axis=0) > 0.5
        s_here |= (end_hi != expect_hi)
        vars_.append(np.where(s_here, np.nan, var)); acs.append(np.where(s_here, np.nan, ac)); sw.append(s_here)
    vars_ = np.array(vars_); acs = np.array(acs); sw = np.array(sw)
    with np.errstate(all="ignore"):
        var_med = np.nanmedian(vars_, axis=0); ac_med = np.nanmedian(acs, axis=0)
    switched = sw.any(axis=0)
    # cells where every seed switched carry the raw (contaminated) statistic so they still plot
    allsw = sw.all(axis=0)
    var_med = np.where(allsw, np.array(vars_).shape[0] and np.nan, var_med)
    return var_med, ac_med, switched, None


def sens_from_state(branchL, branchA, du_frac=0.0, dR_frac=0.0, T=500.0, dt=0.01):
    """Response of engagement to a proportional shift: demand by du_frac, R by dR_frac."""
    ok = ~np.isnan(branchL)
    L = np.where(ok, branchL, 0.0).copy(); A = np.where(ok, branchA, 0.0).copy()
    s0 = cc.s_of(cc.k_of(L, A, 0.0, P), P)
    pp = dict(P, R=P["R"] * (1.0 + dR_frac))
    uu = us * (1.0 + du_frac)
    for _ in range(int(T / dt)):
        dL, dA, _, _ = cc.rhs(L, A, 0.0, uu, pp)
        L = np.maximum(L + dt * dL, 0.0); A = np.maximum(A + dt * dA, 0.0)
    s1 = cc.s_of(cc.k_of(L, A, 0.0, pp), pp)
    d = s1 - s0
    d[~ok] = np.nan
    return d


if __name__ == "__main__":
    u_up, u_rec = cc.folds(P)
    br = cc.branches(us, P, N=3001)
    var_u, ac_u, sw_u, m_u = fluct(cc.FUN_INIT, seed=11, expect_hi=True)
    var_l, ac_l, sw_l, m_l = fluct(cc.COL_INIT, seed=12, expect_hi=False)
    # the lower-branch run is only meaningful where a lower branch exists (u >= u_rec);
    # below u_rec the collapsed init simply recovers to the upper branch
    lower_exists = ~np.isnan(br["lo"])
    upper_exists = ~np.isnan(br["up"]) & (us <= u_up)

    FR_U, FR_R = 0.10, 0.05
    d_up_load = np.nanmax(np.abs(np.vstack([sens_from_state(br["upL"], br["upA"], du_frac=-FR_U),
                                             sens_from_state(br["upL"], br["upA"], du_frac=+FR_U)])), axis=0)
    d_lo_load = np.nanmax(np.abs(np.vstack([sens_from_state(br["loL"], br["loA"], du_frac=-FR_U),
                                             sens_from_state(br["loL"], br["loA"], du_frac=+FR_U)])), axis=0)
    d_up_R = np.abs(sens_from_state(br["upL"], br["upA"], dR_frac=FR_R))
    d_lo_R = np.abs(sens_from_state(br["loL"], br["loA"], dR_frac=FR_R))

    numbers = dict(u_up=u_up, u_rec=u_rec,
                   var_upper=dict(zip(map(float, np.round(us, 4)), map(float, var_u))),
                   ac_upper=dict(zip(map(float, np.round(us, 4)), map(float, ac_u))),
                   sens_upper_load=dict(zip(map(float, np.round(us, 4)), map(float, d_up_load))),
                   sens_upper_R=dict(zip(map(float, np.round(us, 4)), map(float, d_up_R))),
                   switched_upper=[float(u) for u in us[sw_u]], switched_lower=[float(u) for u in us[sw_l & lower_exists]])
    key = lambda arr, mask: {float(u): float(v) for u, v, m in zip(np.round(us, 4), arr, mask) if m}
    numbers.update(var_lower=key(var_l, lower_exists & ~sw_l),
                   sens_lower_load=key(d_lo_load, lower_exists), sens_lower_R=key(d_lo_R, lower_exists))
    json.dump(numbers, open(os.path.join(st.OUT, "figS4_numbers.json"), "w"), indent=1, default=float)

    fig, axes = plt.subplots(1, 3, figsize=(st.DOUBLE, st.ROW), sharex=True)
    from matplotlib.lines import Line2D

    def line(ax, y, mask, color, marker):
        m = mask & ~np.isnan(y)
        ax.plot(us[m], y[m], marker, color=color, ms=4, lw=1.4, ls="-")

    ax = axes[0]; st.grid(ax)
    line(ax, var_u * 1e4, upper_exists & ~sw_u, st.BLUE, "o")
    line(ax, var_l * 1e4, lower_exists & ~sw_l, st.BLUE_D, "s")
    ax.set_ylabel("Variance of $s$ ($\\times 10^{-4}$)"); ax.set_ylim(0, 2.5)
    ax.legend([Line2D([], [], color=st.BLUE, marker="o", ms=4, lw=1.4), Line2D([], [], color=st.BLUE_D, marker="s", ms=4, lw=1.4)],
              ["functioning state", "collapsed state"], loc="upper left", bbox_to_anchor=(0.02, 0.9), fontsize=st.ANN, handlelength=1.6)
    st.thresholds(ax, u_up, u_rec, label=True)
    st.demand_axis(ax, lo=0.0028, hi=0.16)
    st.letter(ax, "a")

    ax = axes[1]; st.grid(ax)
    line(ax, ac_u, upper_exists & ~sw_u & (var_u > 1e-6), st.BLUE, "o")
    ax.set_ylabel("Autocorrelation of $s$"); ax.set_ylim(0, 1.0)
    st.thresholds(ax, u_up, u_rec)
    st.demand_axis(ax, lo=0.0028, hi=0.16)
    st.letter(ax, "b")

    ax = axes[2]; st.grid(ax)
    line(ax, d_up_load, upper_exists, st.BLUE, "o")
    line(ax, d_lo_load, lower_exists, st.BLUE_D, "s")
    ax.set_ylabel("Response, $|\\Delta s|$"); ax.set_ylim(0, 1.0)
    st.thresholds(ax, u_up, u_rec)
    st.demand_axis(ax, lo=0.0028, hi=0.16)
    st.letter(ax, "c")
    fig.tight_layout(w_pad=1.6)
    png = st.save(fig, "figS4_proximity_stats")
    print(png)
    print("upper-branch var min/max:", np.nanmin(var_u[upper_exists]), np.nanmax(var_u[upper_exists & ~sw_u]))
    print("upper AC min/max:", np.nanmin(ac_u[upper_exists]), np.nanmax(ac_u[upper_exists & ~sw_u]))
    print("switched upper cells:", us[upper_exists & sw_u]); print("switched lower cells:", us[lower_exists & sw_l])
    print("sens upper load:", np.round(d_up_load[upper_exists], 4)); print("sens upper R:", np.round(d_up_R[upper_exists], 4))
    print("sens lower load:", np.round(d_lo_load[lower_exists], 4)); print("sens lower R:", np.round(d_lo_R[lower_exists], 5))
