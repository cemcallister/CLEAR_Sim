"""
Figure 6 in the paper (stem fig5_drive). Pushing harder against making it
easier (Prediction 5.5), drawn against time since the intervention.
  - a row for control authority k (the gain) shows the starting condition and
    the identical immediate boost,
  - affect is on a log axis, so the modest rises far from the boundary show,
  - panel (b) gives the same push at three distances from the boundary, which
    ties the figure to the ADHD direction of Figure 2.
The sweep over the affective price is Figure S3 (figures/figS5_drive_sweep.py).

Arms, each from the functioning state at demand u (the model is clear_core):
  no change   nothing changes
  drive       avoidance drive R_v = 1 is added to R (anticipated reward raised
              by half) and generates affect in proportion to the engagement it
              buys, dA/dt += beta_v R_v s, with beta_v = 1 throughout
  friction    effort E is lowered so that the immediate gain in k equals the
              drive arm's, with no affective cost, E' = R (E + A0)/(R + R_v) - A0
(a) the three arms near the boundary, u = 0.95 u_up
(b) the drive arm at u = 0.35, 0.60 and 0.95 u_up. The middle distance is 0.60
    because at 0.70 the price beta_v = 1 sits within half a per cent of that
    case's own collapse price. The run beside the boundary is the drive arm of (a).
Rows: control authority k, engagement s, affect A (log axis).
Reference set R = 2, g = 2.5, n = 4, C = 1, E = 1, slow layer off.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clear"))
import clear_core as cc
import style as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

P = dict(cc.REF)
RV, BETA = 1.0, 1.0
T_ON, DT = 100.0, 0.01
T_SHOW = (-5.0, 50.0)          # window drawn, in time since the intervention
T_LONG = 800.0                 # run length after the intervention, for the long-run numbers

GREY, BLUE = st.INK3, st.BLUE
RAMP = ["#eb6834", "#c4501f", "#8a3313"]   # far, nearer, beside: one hue, darker nearer the boundary
DRIVE = RAMP[2]                            # the drive arm of (a) is the run beside the boundary in (b)
FRACS = [(0.35, "far from the boundary"), (0.60, "nearer"), (0.95, "beside it")]
REF_LS = (0, (1, 2))


def control_state(u):
    e = max(cc.stable_states(u, P, N=3001), key=lambda q: q["s"])
    return e["L"], e["A"], e["s"]


def friction_E(A0):
    return P["R"] * (P["E"] + A0) / (P["R"] + RV) - A0


def after_params(kind, A0):
    if kind == "drive":
        return dict(P, Rv=RV, beta_v=BETA)
    if kind == "friction":
        return dict(P, E=friction_E(A0))
    return dict(P)


def run_arm(u, kind):
    L0, A0, s0 = control_state(u)
    after = after_params(kind, A0)
    pf = None if kind == "control" else (lambda t: ({q: after[q] for q in ("Rv", "beta_v", "E")} if t >= T_ON else {}))
    r = cc.integrate((L0, A0, 0.0), T_ON + T_LONG, DT, P, u=u, p_fn=pf)
    r["tt"] = r["t"] - T_ON
    # control authority with the parameters in force at each moment
    r["k"] = np.where(r["tt"] >= 0, cc.k_of(r["L"], r["A"], 0.0, after), cc.k_of(r["L"], r["A"], 0.0, P))
    r["k_immediate"] = float(cc.k_of(L0, A0, 0.0, after))
    return r, L0, A0, s0


def summary(r, L0, A0, s0):
    m = r["tt"] >= 0
    s, A, tt = r["s"][m], r["A"][m], r["tt"][m]
    i50 = int(round(50.0 / DT))
    below = np.where(s < 0.5)[0]
    return dict(s_start=float(s0), A_start=float(A0), k_start=float(cc.k_of(L0, A0, 0.0, P)),
                k_immediate=r["k_immediate"], k_long_run=float(r["k"][-1]),
                s_peak=float(s[: int(20 / DT)].max()), s_at_50=float(s[i50]), s_long_run=float(s[-1]),
                A_at_50=float(A[i50]), A_long_run=float(A[-1]), change_long_run=float(s[-1] - s0),
                t_below_half=(float(tt[below[0]]) if len(below) else None))


def window(r):
    m = (r["tt"] >= T_SHOW[0]) & (r["tt"] <= T_SHOW[1])
    return r["tt"][m], r["k"][m], r["s"][m], r["A"][m]


def end_label(ax, y, text, va):
    dy = 2 if va == "bottom" else -2
    ax.annotate(text, xy=(T_SHOW[1], y), xytext=(-1, dy), textcoords="offset points",
                fontsize=st.ANN, color=st.INK2, ha="right", va=va)


if __name__ == "__main__":
    u_up, u_rec = cc.folds(P)
    u_near = 0.95 * u_up
    Lf, Af, sf = control_state(u_up * 0.99999)
    numbers = dict(u_up=u_up, u_rec=u_rec, RV=RV, beta_v=BETA, window=T_SHOW, t_long=T_LONG,
                   k_at_collapse_threshold=float(cc.k_of(Lf, Af, 0.0, P)), k_unloaded=P["R"] / P["E"])

    fig = plt.figure(figsize=(st.DOUBLE, 4.9))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 0.95, 0.6], hspace=0.13, wspace=0.26)
    a_k = fig.add_subplot(gs[0, 0]); a_s = fig.add_subplot(gs[1, 0], sharex=a_k); a_A = fig.add_subplot(gs[2, 0], sharex=a_k)
    b_k = fig.add_subplot(gs[0, 1], sharex=a_k, sharey=a_k)
    b_s = fig.add_subplot(gs[1, 1], sharex=a_k, sharey=a_s)
    b_A = fig.add_subplot(gs[2, 1], sharex=a_k, sharey=a_A)
    rows_a, rows_b = (a_k, a_s, a_A), (b_k, b_s, b_A)

    # ---------------- (a) three arms from the same state beside the boundary
    arms = {kind: run_arm(u_near, kind) for kind in ("control", "friction", "drive")}
    cols = dict(control=GREY, friction=BLUE, drive=DRIVE)
    for kind in ("control", "friction", "drive"):
        tt, k, s, A = window(arms[kind][0])
        post = tt >= 0.0
        for ax, y in zip(rows_a, (k, s, A)):
            if kind == "control":
                ax.plot(tt, y, color=cols[kind], zorder=2)
            else:
                ax.plot(tt[post], y[post], color=cols[kind], zorder=3)
    numbers["a"] = dict(u=u_near, E_friction=friction_E(arms["control"][2]),
                        **{kind: summary(*arms[kind]) for kind in arms})
    end_label(a_s, numbers["a"]["friction"]["s_long_run"], "making it easier (friction reduced)", "bottom")
    end_label(a_s, numbers["a"]["control"]["s_start"], "no change", "top")
    end_label(a_s, 0.0, "pushing harder (avoidance drive)", "bottom")
    k_imm = numbers["a"]["drive"]["k_immediate"]
    a_k.annotate("same immediate boost", xy=(0.4, k_imm), xytext=(5.0, 1.95), textcoords="data",
                 fontsize=st.ANN, color=st.INK2, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=st.INK2, lw=0.6, mutation_scale=6, shrinkA=2, shrinkB=2))
    a_k.annotate("pushing harder against making it easier", xy=(0, 1), xycoords="axes fraction",
                 xytext=(0, 5), textcoords="offset points", fontsize=8, ha="left", va="bottom")

    # ---------------- (b) the same push at three distances from the boundary
    numbers["b"] = {}
    for (frac, name), col in zip(FRACS, RAMP):
        u = frac * u_up
        res = run_arm(u, "drive")
        tt, k, s, A = window(res[0])
        numbers["b"][name] = dict(u=u, frac_of_u_up=frac, **summary(*res))
        b_s.hlines(numbers["b"][name]["s_start"], 0.0, T_SHOW[1], color=col, lw=0.7, ls=REF_LS, zorder=1)
        for ax, y in zip(rows_b, (k, s, A)):
            ax.plot(tt, y, color=col, zorder=3)
    # the two upper end levels sit too close for end labels, so (b) takes a legend in the empty lower right
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c) for c in RAMP] + [Line2D([], [], color=RAMP[1], lw=0.7, ls=REF_LS)]
    b_s.legend(handles, [n for _, n in FRACS] + ["level without the push"], loc="center right",
               bbox_to_anchor=(1.0, 0.38), fontsize=st.ANN, handlelength=1.8, labelcolor=st.INK2)
    b_k.annotate("the same push, nearer and nearer the boundary", xy=(0, 1), xycoords="axes fraction",
                 xytext=(0, 5), textcoords="offset points", fontsize=8, ha="left", va="bottom")

    # ---------------- axes
    for ax in rows_a + rows_b:
        ax.axvline(0.0, color=st.INK3, lw=0.7, ls=REF_LS, zorder=1)
        st.grid(ax)
    a_k.set_xlim(*T_SHOW)
    for ax in (a_k, b_k):
        ax.axhline(1.0, color=st.INK3, lw=0.7, ls=REF_LS, zorder=1)
    a_k.annotate("$k$ = 1", xy=(T_SHOW[1], 1.0), xytext=(-1, 2), textcoords="offset points",
                 fontsize=st.ANN, color=st.INK2, ha="right", va="bottom")
    a_k.set_ylim(-0.08, 3.25); a_k.set_yticks([0, 1, 2, 3])
    a_s.set_ylim(-0.03, 1.10); a_s.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a_A.set_yscale("log"); a_A.set_ylim(0.005, 60)
    a_A.set_yticks([0.01, 0.1, 1, 10]); a_A.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a_A.yaxis.set_minor_formatter(mticker.NullFormatter())
    for ax in (a_k, a_s, b_k, b_s):
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in (b_k, b_s, b_A):
        plt.setp(ax.get_yticklabels(), visible=False)
    for ax in (a_A, b_A):
        ax.set_xlabel("Time since the intervention")
    a_k.set_ylabel("Control authority, $k$"); a_s.set_ylabel("Engagement, $s$"); a_A.set_ylabel("Affect, $A$")
    fig.align_ylabels([a_k, a_s, a_A])
    st.letter(a_k, "a"); st.letter(b_k, "b", dx=-14)

    png = st.save(fig, "fig5_drive")
    with open(os.path.join(st.OUT, "fig5_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    print(json.dumps({q: numbers[q] for q in ("k_at_collapse_threshold", "k_unloaded")}, default=float))
    for panel in ("a", "b"):
        for name, v in numbers[panel].items():
            if isinstance(v, dict):
                print(panel, name, {q: (round(v[q], 3) if isinstance(v[q], float) else v[q]) for q in
                                    ("k_start", "k_immediate", "k_long_run", "s_start", "s_long_run", "A_start", "A_long_run")})
    print(png)
