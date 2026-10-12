"""
Figure 3. Hysteresis as a person would live it, drawn against time.

Every panel has the same structure: the demand a person is under (top strip)
and their engagement (below), against time.

(a) Switch-like engagement response (n = 4, used in all figures). Demand is
    raised slowly and then lowered slowly. Engagement holds until demand
    crosses the collapse threshold, drops to zero, and returns only after
    demand has fallen below the recovery threshold. Orange points: the same
    demand, 0.05, on the way up (functioning) and down (collapsed).
(b) The same demand history under the goal weight of Equation (8),
    s = k/(1+k), which is n = 1. The two thresholds lie close together.
(c) Recovery after a short and a long collapse. Demand is held past the
    collapse threshold for 5 or 320 time units, then relieved to 0.005.
    The longer collapse returns later. Slow affect state on.
Each transition lags the moment demand crosses its threshold (open circles in
the strips of (a) and (b)), because near a threshold the state that is being
left changes slowly. The dotted verticals mark the transitions (steepest change).

Layout: (a) and (b) side by side, (c) full width with the whole hold visible,
threshold crossings marked. The demand strip of (c) is on a log axis so relief
below the recovery threshold shows; the strips of (a) and (b) stay linear,
since a log axis would turn their ramps into domes.
Reference set R = 2, g = 2.5, C = 1, E = 1.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clear"))
import clear_core as cc
import style as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

P = dict(cc.REF)
DT = 0.01
T_RISE, T_FALL, T_HOLD = 600.0, 600.0, 300.0
U_LO, U_HI = 0.003, 0.20
T_TOT = T_RISE + T_FALL + T_HOLD
REF_LS = (0, (1, 2))
C_X0 = -365.0            # start of the time axis in (c), so the whole 320-unit hold shows


def u_tri(t):
    if t <= T_RISE:
        return U_LO + (U_HI - U_LO) * t / T_RISE
    if t <= T_RISE + T_FALL:
        return U_HI - (U_HI - U_LO) * (t - T_RISE) / T_FALL
    return U_LO


def t_of_u(u, leg):
    if leg == "up":
        return T_RISE * (u - U_LO) / (U_HI - U_LO)
    return T_RISE + T_FALL * (U_HI - u) / (U_HI - U_LO)


def run_tri(p):
    y0 = (U_LO / (p["c0L"] + 1.0), 0.0, 0.0)
    return cc.integrate(y0, T_TOT, DT, p, u_fn=u_tri)


def run_hold(T_hold, p, u_base=0.10, u_pulse=0.50, u_rel=0.005, T_after=200.0, slow=True):
    """Settle at u_base, hold at u_pulse for T_hold, relieve to u_rel. Returns the
    record with time counted from relief (negative during the hold)."""
    y = (u_base / (p["c0L"] + 1.0), 0.0, 0.0)
    y = cc.integrate(y, 200.0, DT, p, u=u_base, slow=slow, record=False)
    base = cc.integrate(y, 45.0, DT, p, u=u_base, slow=slow)
    y = (base["L"][-1], base["A"][-1], base["Ab"][-1])
    hold = cc.integrate(y, T_hold, DT, p, u=u_pulse, slow=slow)
    y = (hold["L"][-1], hold["A"][-1], hold["Ab"][-1])
    rel = cc.integrate(y, T_after, DT, p, u=u_rel, slow=slow)
    t = np.concatenate([base["t"] - 45.0 - T_hold, hold["t"] - T_hold, rel["t"]])
    s = np.concatenate([base["s"], hold["s"], rel["s"]]); u = np.concatenate([base["u"], hold["u"], rel["u"]])
    return dict(t=t, s=s, u=u, t_rec=float(rel["t"][np.argmax(rel["s"] > 0.5)]))


def extend_base(r, t0):
    """The run sits at rest on its base demand before the recorded window; draw
    that rest state back to t0 so the short and long runs share one time axis."""
    if r["t"][0] <= t0:
        return r
    tt = np.arange(t0, r["t"][0], 1.0)
    return dict(r, t=np.concatenate([tt, r["t"]]), s=np.concatenate([np.full(tt.size, r["s"][0]), r["s"]]),
                u=np.concatenate([np.full(tt.size, r["u"][0]), r["u"]]))


def events(r):
    """Times of the collapse (steepest fall of engagement) and the recovery
    (steepest rise after it). Slope-based so that it also works for n = 1,
    whose functioning branch sits below one half near the fold."""
    ds = np.gradient(r["s"], r["t"])
    ic = int(np.argmin(ds)); tc = float(r["t"][ic])
    ir = ic + int(np.argmax(ds[ic:])); tr = float(r["t"][ir])
    return tc, tr


def crossings(u_up_, u_rec_):
    return t_of_u(u_up_, "up"), t_of_u(u_rec_, "down")


def log_demand(ax, lo=0.002, hi=0.7, ticks=None):
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    ticks = ticks or [t for t in st.U_TICKS if lo <= t <= hi]
    ax.set_yticks(ticks); ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("Demand, $u$", labelpad=2)


def note(ax, x, y, text, ha="right", va="bottom", dx=0, dy=2, color=None):
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points", fontsize=st.ANN,
                color=color or st.INK2, ha=ha, va=va)


def strip_and_trace(ax_top, ax_bot, r, u_up_, u_rec_, col, title, ls="-"):
    ax_top.plot(r["t"], r["u"], color=st.INK3, lw=1.2)
    ax_top.set_ylim(0, 0.22); ax_top.set_yticks([0, 0.1, 0.2]); ax_top.set_ylabel("Demand, $u$", labelpad=2)
    for uu in (u_up_, u_rec_):
        ax_top.axhline(uu, color=st.INK3, lw=0.7, ls=REF_LS)
    xu, xr = crossings(u_up_, u_rec_)
    ax_top.plot([xu, xr], [u_up_, u_rec_], "o", ms=4, mfc="white", mec=st.INK2, mew=0.9, zorder=5)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    st.grid(ax_top); st.grid(ax_bot)
    ax_bot.plot(r["t"], r["s"], color=col, lw=1.6, ls=ls)
    ax_bot.set_ylim(-0.03, 1.08); ax_bot.set_yticks([0, 0.5, 1.0]); ax_bot.set_xlim(0, T_TOT)
    ax_bot.set_xlabel("Time")
    tc, tr = events(r)
    for ax in (ax_top, ax_bot):
        ax.axvline(tc, color=st.INK3, lw=0.7, ls=REF_LS)
        ax.axvline(tr, color=st.INK3, lw=0.7, ls=REF_LS)
    ax_top.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                    fontsize=8, color=st.INK, ha="left", va="bottom")
    return tc, tr


if __name__ == "__main__":
    u_up, u_rec = cc.folds(P)
    u_up1, u_rec1 = cc.folds(dict(P, n=1))
    r4 = run_tri(P)
    r1 = run_tri(dict(P, n=1))
    short = run_hold(5.0, P); long_ = run_hold(320.0, P)
    tc4, tr4 = events(r4); tc1, tr1 = events(r1)
    x4, xr4 = crossings(u_up, u_rec); x1, xr1 = crossings(u_up1, u_rec1)
    numbers = dict(u_up=u_up, u_rec=u_rec, u_up_n1=u_up1, u_rec_n1=u_rec1,
                   t_collapse=tc4, u_at_collapse=u_tri(tc4), t_recover=tr4, u_at_recover=u_tri(tr4),
                   t_collapse_n1=tc1, u_at_collapse_n1=u_tri(tc1), t_recover_n1=tr1, u_at_recover_n1=u_tri(tr1),
                   t_rec_short=short["t_rec"], t_rec_long=long_["t_rec"],
                   t_cross_collapse=x4, t_cross_recovery=xr4, t_cross_collapse_n1=x1, t_cross_recovery_n1=xr1,
                   lag_collapse=tc4 - x4, lag_recovery=tr4 - xr4, lag_collapse_n1=tc1 - x1, lag_recovery_n1=tr1 - xr1)

    fig = plt.figure(figsize=(st.DOUBLE, 4.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.42)
    top = outer[0].subgridspec(1, 2, wspace=0.12)
    ga = top[0].subgridspec(2, 1, height_ratios=[0.5, 1.0], hspace=0.10)
    gb = top[1].subgridspec(2, 1, height_ratios=[0.5, 1.0], hspace=0.10)
    gc = outer[1].subgridspec(2, 1, height_ratios=[0.5, 1.0], hspace=0.10)
    axd4 = fig.add_subplot(ga[0]); axs4 = fig.add_subplot(ga[1], sharex=axd4)
    axd1 = fig.add_subplot(gb[0], sharex=axd4, sharey=axd4); axs1 = fig.add_subplot(gb[1], sharex=axd4, sharey=axs4)
    axdc = fig.add_subplot(gc[0]); axsc = fig.add_subplot(gc[1], sharex=axdc)

    # ---------------- (a) switch-like response, n = 4
    tc, tr = strip_and_trace(axd4, axs4, r4, u_up, u_rec, st.BLUE, "switch-like engagement ($n$ = 4)")
    note(axd4, T_TOT - 4, u_up, "collapse")
    note(axd4, T_TOT - 4, u_rec, "recovery")
    u_same = 0.05
    t1, t2 = t_of_u(u_same, "up"), t_of_u(u_same, "down")
    s1 = float(np.interp(t1, r4["t"], r4["s"])); s2 = float(np.interp(t2, r4["t"], r4["s"]))
    axs4.plot([t1, t2], [s1, s2], "o", color=st.ORANGE, ms=4.5, zorder=5)
    axd4.plot([t1, t2], [u_same, u_same], "o", color=st.ORANGE, ms=3.5, zorder=5)
    axd4.hlines(u_same, t1, t2, color=st.ORANGE, lw=0.7, ls=REF_LS, zorder=4)
    axd4.annotate("same demand", xy=(T_RISE, u_same), xytext=(0, 1), textcoords="offset points", fontsize=st.ANN,
                 color=st.INK2, ha="center", va="bottom", zorder=6,
                 bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    note(axs4, t1 + 55.0, s1, "functioning", ha="center", va="top", dy=-7)
    note(axs4, t2, s2, "collapsed", ha="center", va="bottom", dy=7)
    axs4.set_ylabel("Engagement, $s$")
    st.letter(axd4, "a", dx=-38)

    # ---------------- (b) the goal weight of Equation (8), n = 1
    tc, tr = strip_and_trace(axd1, axs1, r1, u_up1, u_rec1, st.AQUA, "goal weight of Equation (8), $s = k/(1+k)$ ($n$ = 1)", ls=(0, (4, 2)))
    note(axd1, T_TOT - 4, u_up1, "collapse")
    note(axd1, T_TOT - 4, u_rec1, "recovery", va="top", dy=-2)
    plt.setp(axd1.get_yticklabels(), visible=False); plt.setp(axs1.get_yticklabels(), visible=False)
    axd1.set_ylabel(""); st.letter(axd1, "b", dx=-14)

    # ---------------- (c) short against long collapse, time counted from relief
    cols = {"short": st.BLUE, "long": st.BLUE_D}
    for r, key in ((extend_base(long_, C_X0), "long"), (extend_base(short, C_X0), "short")):
        axdc.plot(r["t"], r["u"], color=cols[key], lw=1.2)
        axsc.plot(r["t"], r["s"], color=cols[key], lw=1.6)
    axdc.set_xlim(C_X0, 160); log_demand(axdc, hi=0.9, ticks=[0.01, 0.1])
    for uu in (u_up, u_rec):
        axdc.axhline(uu, color=st.INK3, lw=0.7, ls=REF_LS)
    note(axdc, 158, u_up, "collapse")
    note(axdc, 158, u_rec, "recovery")
    note(axdc, -316, 0.27, "held 320", ha="left", va="center", dy=0)
    note(axdc, -9, 0.27, "held 5", ha="right", va="center", dy=0)
    plt.setp(axdc.get_xticklabels(), visible=False)
    st.grid(axdc); st.grid(axsc)
    for ax in (axdc, axsc):
        ax.axvline(0, color=st.INK3, lw=0.7, ls=REF_LS)
    axsc.set_ylim(-0.03, 1.08); axsc.set_yticks([0, 0.5, 1.0])
    axsc.set_xlabel("Time since demand relieved"); axsc.set_ylabel("Engagement, $s$")
    axsc.annotate("", xy=(long_["t_rec"], 0.5), xytext=(short["t_rec"], 0.5),
                  arrowprops=dict(arrowstyle="-|>", color=st.INK2, lw=0.8, mutation_scale=8))
    note(axsc, short["t_rec"], 0.5, "%.0f time units\nlater" % (long_["t_rec"] - short["t_rec"]), dx=-6, dy=0, va="center")
    axdc.annotate("after a short and a long collapse", xy=(0, 1), xycoords="axes fraction", xytext=(0, 5),
                  textcoords="offset points", fontsize=8, color=st.INK, ha="left", va="bottom")
    st.letter(axdc, "c", dx=-38)
    fig.align_ylabels([axd4, axs4, axdc, axsc])

    png = st.save(fig, "fig3_hysteresis")
    with open(os.path.join(st.OUT, "fig3_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    print(json.dumps(numbers, indent=1, default=float)); print(png)
