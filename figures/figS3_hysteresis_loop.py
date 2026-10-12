"""
Figure S1 in the paper (stem figS3_hysteresis_loop): the parameter-space form of Figure 3. Hysteresis loop against demand, duration-scaled recovery, and loop width against n.

(a) Equilibrium engagement against demand: the two stable branches and the
    saddle between them, with a slow up-and-down ramp overlaid. Collapse at the
    upper fold, recovery only at the lower fold.
(b) Time from relief (demand cut to 0.005) to recovery (s crossing 0.5) against
    time held past the boundary, with and without the slow affect state.
    Returning demand to its pre-episode value of 0.10 never recovers (annotated).
(c) Width of the loop (collapse fold / recovery fold) against the engagement
    exponent n. n = 1 is the goal weight of Equation (8).
Reference set: R = 2, g = 2.5, C = 1, n = 4; slow layer off except in (b).
"""
import numpy as np, sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clear"))
import clear_core as cc
import style as st
import matplotlib
import matplotlib.pyplot as plt

P = dict(cc.REF)


def branches_full(us, p):
    up, lo, sad = [np.full(len(us), np.nan) for _ in range(3)]
    for i, u in enumerate(us):
        eqs = cc.equilibria(u, p, N=3001)
        stab = [e for e in eqs if e["stable"]]; uns = [e for e in eqs if not e["stable"]]
        if stab:
            up[i] = max(e["s"] for e in stab)
            if len(stab) >= 2:
                lo[i] = min(e["s"] for e in stab)
        if uns:
            sad[i] = uns[0]["s"]
    return up, lo, sad


def ramp(p, us, y0, T=400.0, dt=0.01):
    L, A, Ab = y0
    out = []
    for u in us:
        L, A, Ab = cc.integrate((L, A, Ab), T, dt, p, u=u, record=False)
        out.append(float(cc.s_of(cc.k_of(L, A, Ab, p), p)))
    return np.array(out), (L, A, Ab)


def recovery_time(T_hold, slow, p, u_base=0.10, u_pulse=0.50, u_rec=0.005, dt=0.01):
    y = (u_base / (p["c0L"] + 1.0), 0.0, 0.0)
    y = cc.integrate(y, 200.0, dt, p, u=u_base, slow=slow, record=False)
    y = cc.integrate(y, T_hold, dt, p, u=u_pulse, slow=slow, record=False)
    rec = cc.integrate(y, 3000.0, dt, p, u=u_rec, slow=slow)
    idx = np.argmax(rec["s"] > 0.5)
    return float(rec["t"][idx]) if (rec["s"] > 0.5).any() else np.inf


if __name__ == "__main__":
    numbers = {}
    # ---------------- (a)
    us = np.geomspace(0.002, 0.4, 260)
    up, lo, sad = branches_full(us, P)
    u_up, u_rec = cc.folds(P)
    numbers["u_up"], numbers["u_rec"], numbers["ratio"] = u_up, u_rec, u_up / u_rec
    us_r = np.geomspace(0.002, 0.4, 120)
    s_upr, ylast = ramp(P, us_r, (0.02, 0.0, 0.0))
    s_dnr, _ = ramp(P, us_r[::-1], ylast)

    # ---------------- (b)
    Ts = np.array([5, 10, 20, 40, 80, 160, 320])
    rs = [recovery_time(T, True, P) for T in Ts]
    rf = [recovery_time(T, False, P) for T in Ts]
    r_base = recovery_time(80, True, P, u_rec=0.10)
    numbers["duration_slow"] = dict(zip(map(int, Ts), rs)); numbers["duration_fast"] = dict(zip(map(int, Ts), rf))
    numbers["return_to_baseline_recovery"] = r_base

    # ---------------- (c)
    ns = [1, 1.5, 2, 3, 4, 6, 8]
    ratios = []
    for n in ns:
        a, b = cc.folds(dict(P, n=n))
        ratios.append(a / b if a and b else np.nan)
    numbers["ratio_vs_n"] = dict(zip(map(str, ns), ratios))

    fig, axes = plt.subplots(1, 3, figsize=(st.DOUBLE, st.ROW), gridspec_kw=dict(width_ratios=[1.3, 1.0, 0.75]))
    from matplotlib.lines import Line2D
    ax = axes[0]
    st.grid(ax)
    ax.plot(us_r, s_upr, color=st.BLUE, lw=1.6)
    ax.plot(us_r[::-1], s_dnr, color=st.ORANGE, lw=1.6, ls=(0, (4, 2)))
    for xs, ys, frac, col in ((us_r, s_upr, 0.55, st.BLUE), (us_r[::-1], s_dnr, 0.6, st.ORANGE)):
        i = int(frac * len(xs))
        ax.annotate("", xy=(xs[i + 2], ys[i + 2]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=10))
    st.thresholds(ax, u_up, u_rec, label=True)
    st.demand_axis(ax, lo=0.002, hi=0.4)
    ax.set_ylabel("Engagement, $s$"); ax.set_ylim(-0.03, 1.0)
    handles = [Line2D([], [], color=st.BLUE, lw=1.6), Line2D([], [], color=st.ORANGE, lw=1.6, ls=(0, (4, 2)))]
    labels_a = ["demand rising", "demand falling"]
    # the empty middle of the loop, clear of both vertical drops
    ax.legend(handles, labels_a, loc="center", bbox_to_anchor=(0.54, 0.45), fontsize=st.ANN, handlelength=1.8)
    st.letter(ax, "a")

    ax = axes[1]
    st.grid(ax)
    ax.plot(Ts, rs, "o-", color=st.BLUE, label="with slow affect state")
    ax.plot(Ts, rf, "s-", color=st.AQUA, label="fast states only")
    ax.set_xscale("log"); ax.set_xticks([5, 20, 80, 320]); ax.set_xticklabels(["5", "20", "80", "320"])
    ax.minorticks_off()
    ax.set_xlabel("Time held past the boundary"); ax.set_ylabel("Time from relief to recovery")
    ax.set_ylim(0, 130)
    ax.legend(loc="lower right", fontsize=8)
    st.letter(ax, "b")

    ax = axes[2]
    st.grid(ax)
    ax.plot(ns, ratios, "o-", color=st.BLUE)
    ax.set_yscale("log"); ax.set_ylim(0.9, 120)
    ax.set_yticks([1, 3, 10, 30, 100]); ax.set_yticklabels(["1", "3", "10", "30", "100"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Exponent, $n$"); ax.set_ylabel("Loop width, $u_{up}/u_{rec}$")
    ax.set_xticks([1, 2, 4, 6, 8])
    ax.annotate("Equation (8)", xy=(1, ratios[0]), xytext=(2.8, 1.25), fontsize=st.ANN, color=st.INK2,
                va="center", arrowprops=dict(arrowstyle="-", color=st.INK3, lw=0.6))
    ax.annotate("reference", xy=(4, ratios[4]), xytext=(5.0, 8.0), fontsize=st.ANN, color=st.INK2,
                va="center", arrowprops=dict(arrowstyle="-", color=st.INK3, lw=0.6))
    st.letter(ax, "c")

    fig.tight_layout(w_pad=1.8)
    png = st.save(fig, "figS3_hysteresis_loop")
    json.dump(numbers, open(os.path.join(st.OUT, "figS3_numbers.json"), "w"), indent=1, default=float)
    print(json.dumps(numbers, indent=1, default=float)); print(png)
