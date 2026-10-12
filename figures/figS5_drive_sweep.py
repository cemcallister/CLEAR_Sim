"""
Figure S3 in the paper (stem figS5_drive_sweep). The affective price of
avoidance drive, swept at the three distances from the boundary of Figure 6b.
The long-run outcome is drawn as the change from no push, so one line marks
"no effect".

Drive as in Figure 6 (stem fig5_drive): R_v = 1 added to R, affect generated at beta_v R_v s.
For each distance (u = 0.35, 0.60, 0.95 u_up) and price beta_v, the run starts
at the functioning state and is read 800 time units after the push.
Reference set R = 2, g = 2.5, n = 4, C = 1, E = 1, slow layer off.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clear"))
import clear_core as cc
import style as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P = dict(cc.REF)
RV = 1.0
T_AFTER, DT = 800.0, 0.01
BETAS = np.round(np.arange(0.0, 2.0 + 1e-9, 0.01), 2)
RAMP = ["#eb6834", "#c4501f", "#8a3313"]          # as Figure 6b, darker nearer the boundary
FRACS = [(0.35, "far from the boundary"), (0.60, "nearer"), (0.95, "beside it")]
BETA_FIG5 = 1.0


def control_state(u):
    e = max(cc.stable_states(u, P, N=3001), key=lambda q: q["s"])
    return e["L"], e["A"], e["s"]


def friction_E(A0):
    return P["R"] * (P["E"] + A0) / (P["R"] + RV) - A0


def break_even_closed_form(A0, s0):
    """Exact break-even: at this price the pushed system settles at the unpushed
    state, because the affect the push generates exactly cancels the authority it
    adds there, (E + A0)(c0A + kappa s0)/(R s0). The check confirms it by bisection."""
    return (P["E"] + A0) * (P["c0A"] + P["kappa"] * s0) / (P["R"] * s0)


def long_run(u, betas):
    """Long-run engagement under drive at each price, vectorised over the prices."""
    L0, A0, s0 = control_state(u)
    betas = np.asarray(betas, dtype=float)
    L = np.full(betas.shape, L0); A = np.full(betas.shape, A0)
    pp = dict(P, Rv=RV, beta_v=betas)
    for _ in range(int(round(T_AFTER / DT))):
        dL, dA, _, s = cc.rhs(L, A, 0.0, u, pp)
        L = np.maximum(L + DT * dL, 0.0); A = np.maximum(A + DT * dA, 0.0)
    return cc.s_of(cc.k_of(L, A, 0.0, pp), pp), (L0, A0, s0)


def friction_long_run(u):
    L0, A0, s0 = control_state(u)
    r = cc.integrate((L0, A0, 0.0), T_AFTER, DT, dict(P, E=friction_E(A0)), u=u, record=False)
    return float(cc.s_of(cc.k_of(r[0], r[1], 0.0, dict(P, E=friction_E(A0))), P))


if __name__ == "__main__":
    u_up, u_rec = cc.folds(P)
    numbers = dict(u_up=u_up, RV=RV, t_after=T_AFTER, beta_step=0.01, levels={})

    fig, ax = plt.subplots(figsize=(st.SINGLE, 2.7))
    st.grid(ax)
    ax.axhline(0.0, color=st.INK3, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.axvline(BETA_FIG5, color=st.INK3, lw=0.7, ls=(0, (1, 2)), zorder=1)
    handles = []
    for (frac, name), col in zip(FRACS, RAMP):
        u = frac * u_up
        s_end, (L0, A0, s0) = long_run(u, BETAS)
        d = s_end - s0
        ax.plot(BETAS, d, color=col, zorder=3)
        handles.append(Line2D([], [], color=col))
        below = np.where(d < 0)[0]
        coll = np.where(s_end < 0.5)[0]
        pick = lambda b: float(d[int(round(b / 0.01))])
        numbers["levels"][name] = dict(
            u=float(u), frac_of_u_up=frac, s_control=float(s0), A_control=float(A0),
            s_friction_reduced=friction_long_run(u), s_drive_no_price=float(s_end[0]),
            change_at_price={"0": pick(0.0), "0.3": pick(0.3), "0.65": pick(0.65), "1": pick(1.0)},
            break_even_closed_form=float(break_even_closed_form(A0, s0)),
            break_even_grid=(float(BETAS[below[0]]) if len(below) else None),
            collapse_price_grid=(float(BETAS[coll[0]]) if len(coll) else None))
    ax.set_xlim(0, 2.0); ax.set_ylim(-1.0, 0.25)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0]); ax.set_yticks([-1.0, -0.75, -0.5, -0.25, 0, 0.25])
    ax.set_xlabel("Affect per unit of engagement bought, $\\beta_v$")
    ax.set_ylabel("Change in long-run engagement")
    ax.annotate("no push", xy=(2.0, 0.0), xytext=(-1, 2), textcoords="offset points",
                fontsize=st.ANN, color=st.INK2, ha="right", va="bottom")
    ax.annotate("Figure 6 price", xy=(BETA_FIG5, 1), xycoords=("data", "axes fraction"), xytext=(3, -2),
                textcoords="offset points", fontsize=st.ANN, color=st.INK2, ha="left", va="top")
    fig.tight_layout()
    fig.legend(handles, [n for _, n in FRACS], loc="upper center", bbox_to_anchor=(0.55, 0.0), ncol=3,
               fontsize=st.ANN, handlelength=1.6, columnspacing=1.2)
    png = st.save(fig, "figS5_drive_sweep")
    with open(os.path.join(st.OUT, "figS5_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    for name, v in numbers["levels"].items():
        print(name, {q: (round(x, 3) if isinstance(x, float) else x) for q, x in v.items() if q != "change_at_price"},
              {q: round(x, 3) for q, x in v["change_at_price"].items()})
    print(png)
