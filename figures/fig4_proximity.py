"""
Figure 4. What approaching the boundary looks like, drawn against time.

One time axis, full width: demand is raised very slowly toward the collapse
threshold while small random perturbations act on load and affect.
  (a) demand and engagement: the jitter grows and slows before the collapse
  (b) variance of engagement in a sliding window
  (c) lag-0.5 autocorrelation of engagement in the same window
The response to the same small push at three distances is in Figure S2c (both
branches), and the intervention version in Figure 6b.
Reference set R = 2, g = 2.5, n = 4, slow layer off. sigma = 0.02, dt = 0.02.
"""
import numpy as np, sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clear"))
import clear_core as cc
import style as st
import matplotlib.pyplot as plt

P = dict(cc.REF)
DT, SIG, SEED = 0.02, 0.02, 1          # seed 1 is the median of six by collapse time
SEEDS_CHECK = (1, 2, 3, 7, 11, 23)
T_RAMP = 4000.0
U0, U1 = 0.02, 0.17
WIN, LAG = 200.0, 25          # window in time units; lag in steps (0.5 t.u.)


def u_ramp(t):
    return U0 + (U1 - U0) * min(t, T_RAMP) / T_RAMP


def rolling_stats(s, dt, win, lag):
    n = int(win / dt)
    N = len(s)
    var = np.full(N, np.nan); ac = np.full(N, np.nan)
    tt = np.arange(n, dtype=float)
    for i in range(n, N, 25):
        x = s[i - n:i]
        x = x - np.polyval(np.polyfit(tt, x, 1), tt)      # remove the within-window trend
        v = x.var()
        var[i] = v
        ac[i] = (x[:-lag] * x[lag:]).mean() / v if v > 1e-14 else np.nan
    return var, ac


if __name__ == "__main__":
    u_up, u_rec = cc.folds(P)
    y0 = cc.stable_states(U0, P, N=3001); e0 = max(y0, key=lambda q: q["s"])
    r = cc.integrate((e0["L"], e0["A"], 0.0), T_RAMP + 200.0, DT, P, u_fn=u_ramp, sigma=SIG, seed=SEED)
    t, s, u = r["t"], r["s"], r["u"]
    var, ac = rolling_stats(s, DT, WIN, LAG)
    t_cross = float(t[np.argmax(u >= u_up)])
    t_coll = float(t[np.argmax(s < 0.5)])
    T_END = t_coll - 20.0                       # last window ends before the plunge itself
    ok = np.isfinite(var) & (t <= T_END)
    numbers = dict(u_up=u_up, seed=SEED, t_threshold_crossed=t_cross, t_collapse=t_coll, u_at_collapse=float(u_ramp(t_coll)),
                   var_first=float(var[ok][0]), var_last=float(var[ok][-1]),
                   ac_first=float(ac[ok][0]), ac_last=float(ac[ok][-1]), seeds={})
    # the same run under other seeds: where the collapse falls and how the statistics move
    for sd in SEEDS_CHECK:
        rr = cc.integrate((e0["L"], e0["A"], 0.0), T_RAMP + 200.0, DT, P, u_fn=u_ramp, sigma=SIG, seed=sd)
        tc = float(rr["t"][np.argmax(rr["s"] < 0.5)])
        vv, aa = rolling_stats(rr["s"], DT, WIN, LAG)
        kk = np.isfinite(vv) & (rr["t"] <= tc - 20.0)
        numbers["seeds"][sd] = dict(t_collapse=tc, u_at_collapse=float(u_ramp(tc)),
                                    var_first=float(vv[kk][0]), var_last=float(vv[kk][-1]),
                                    ac_first=float(aa[kk][0]), ac_last=float(aa[kk][-1]))

    fig = plt.figure(figsize=(st.DOUBLE, 4.4))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.85, 0.85, 0.85], hspace=0.30)
    top = outer[0].subgridspec(2, 1, height_ratios=[0.6, 1.25], hspace=0.12)      # (a): demand over engagement
    axu = fig.add_subplot(top[0]); axs = fig.add_subplot(top[1], sharex=axu)
    axv = fig.add_subplot(outer[1], sharex=axu); axa = fig.add_subplot(outer[2], sharex=axu)

    # ---------------- (a) demand strip and engagement under a slow ramp with noise
    st.grid(axu)
    axu.plot(t, u, color=st.INK3, lw=1.2)
    axu.axhline(u_up, color=st.INK3, lw=0.7, ls=(0, (1, 2)))
    axu.set_ylim(0, 0.2); axu.set_yticks([0, 0.1, 0.2]); axu.set_ylabel("Demand, $u$", labelpad=2)
    axu.annotate("collapse threshold", xy=(10, u_up), xytext=(0, 2), textcoords="offset points",
                 fontsize=st.ANN, color=st.INK2, ha="left", va="bottom")
    st.grid(axs)
    axs.plot(t, s, color=st.BLUE, lw=0.8)
    axs.set_ylim(-0.03, 1.03); axs.set_ylabel("Engagement, $s$")
    for ax in (axu, axs, axv, axa):
        ax.axvline(t_coll, color=st.INK3, lw=0.7, ls=(0, (1, 2)))
    axs.annotate("jitter grows\nand slows", xy=(t_coll - 40, 0.45), xytext=(-6, 0), textcoords="offset points",
                 fontsize=st.ANN, color=st.INK2, ha="right", va="center")
    for ax in (axu, axs, axv):
        plt.setp(ax.get_xticklabels(), visible=False)
    st.letter(axu, "a", dx=-42)

    # ---------------- (b) rolling variance, (c) rolling autocorrelation, windows before the threshold
    m = np.isfinite(var) & (t <= T_END)
    st.grid(axv); axv.plot(t[m], var[m] * 1e4, color=st.BLUE, lw=1.4)
    axv.set_ylabel("Variance\n($\\times 10^{-4}$)"); axv.set_ylim(0, None)
    st.letter(axv, "b", dx=-42)
    st.grid(axa); axa.plot(t[m], ac[m], color=st.BLUE, lw=1.4)
    axa.set_ylabel("Auto-\ncorrelation"); axa.set_ylim(0, 1); axa.set_yticks([0, 0.5, 1])
    axa.set_xlabel("Time"); axa.set_xlim(0, T_RAMP + 200.0)
    st.letter(axa, "c", dx=-42)

    fig.align_ylabels([axu, axs, axv, axa])
    png = st.save(fig, "fig4_proximity")
    json.dump(numbers, open(os.path.join(st.OUT, "fig4_numbers.json"), "w"), indent=1, default=float)
    print(json.dumps(numbers, indent=1, default=float)); print(png)
