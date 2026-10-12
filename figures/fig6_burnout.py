"""
Figure 5 in the paper (stem fig6_burnout). Burnout: repeated collapse lowers
the capacity ceiling, and demand and restoration both set how far.

The fast loop of Section 3.7 with the slow ceiling of clear/burnout_core.py
(Equations 13 and 14), in which the restoration rate is worn by overload and
returns slowly toward its baseline. One person throughout. Demand rises through
each period (work accumulating toward its end) and drops to basal through each
relief. Rows differ from (a) in one thing each:

(a) peak demand 0.18, typical restoration rate
(b) peak demand 0.25, typical restoration rate              (the demand route)
(c) peak demand 0.18, restoration rate half the typical    (the restoration route)

Each row: demand and the collapse threshold, which falls as the ceiling falls;
then engagement, the ceiling and the restoration rate (relative to the typical rate).
"Restoration" follows Section 3.1 (capacity is restored by recovery processes);
"recovery" is kept for engagement returning after collapse.
Reference set R = 2, g = 2.5, n = 4, slow affect layer off.
"""
import numpy as np, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clear"))
import clear_core as cc
import burnout_core as bc
import style as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

P = dict(cc.REF)
DT = 0.02
T_TERM, T_HOL = 600.0, 150.0
PERIOD = T_TERM + T_HOL
N_PER = 15
LO_FRAC = 0.4          # each period's demand starts at 0.4 x its peak
REC_EVERY = 10         # record every 0.2 time units for plotting and summaries
SHADE = "#f2f1ee"
ROWS = [  # (label, peak demand, restoration rate as a fraction of typical, title)
    ("a", 0.18, 1.0, "Peak demand 0.18, typical restoration rate"),
    ("b", 0.25, 1.0, "Peak demand 0.25, typical restoration rate"),
    ("c", 0.18, 0.5, "Peak demand 0.18, restoration rate half the typical"),
]


def per_period(rec, eta_base):
    t, s, C, eta = rec["t"], rec["s"], rec["C"], rec["eta"]
    rows = []
    for k in range(N_PER):
        m = (t >= k * PERIOD) & (t < k * PERIOD + T_TERM)
        tt = t[m] - k * PERIOD; ss = s[m]
        tc = float(tt[np.argmax(ss < 0.5)]) if (ss < 0.5).any() else None
        mh = (t >= k * PERIOD + T_TERM) & (t < (k + 1) * PERIOD)
        rows.append(dict(period=k + 1,
                         C_start=float(C[m][0]),
                         functioning_share=(tc / T_TERM if tc is not None else 1.0),
                         C_min_in_period=float(C[m].min()),
                         C_after_relief=float(C[mh][-1]),
                         s_after_relief=float(s[mh][-1]),
                         repair_rel_typical=float(eta[mh][-1] / bc.ETA_TYPICAL),
                         repair_rel_own_baseline=float(eta[mh][-1] / eta_base)))
    return rows


if __name__ == "__main__":
    Cs = np.geomspace(0.2, 1.0, 18)
    ups, recs = bc.threshold_table(P, Cs)
    numbers = dict(slow=dict(bc.SLOW), T_term=T_TERM, T_relief=T_HOL, n_periods=N_PER,
                   demand_start_fraction_of_peak=LO_FRAC, u_basal=bc.U_BASAL, dt=DT,
                   threshold_table=dict(C=Cs.tolist(), u_up=ups.tolist(), u_rec=recs.tolist()),
                   rows={})

    fig = plt.figure(figsize=(st.DOUBLE, 8.2))
    outer = fig.add_gridspec(3, 1, hspace=0.46)
    T_tot = N_PER * PERIOD
    for r, (lab, peak, fe, title) in enumerate(ROWS):
        sp = dict(bc.SLOW, eta=fe * bc.ETA_TYPICAL)
        u_fn, period = bc.term_holiday(LO_FRAC * peak, peak, T_TERM, T_HOL)
        rec = bc.integrate((bc.U_BASAL / 1.1, 0.0, 0.0, 1.0), T_tot, DT, P, u_fn, slow_p=sp,
                           slow_affect=False, record_every=REC_EVERY)
        rows = per_period(rec, sp["eta"])
        last = rows[-1]
        settle = next(k for k in range(N_PER) if abs(rows[k]["C_after_relief"] - last["C_after_relief"]) < 0.01) + 1
        numbers["rows"][lab] = dict(peak=peak, repair_fraction_of_typical=fe, periods=rows,
                                    settled=dict(C_after_relief=last["C_after_relief"],
                                                 functioning_share=last["functioning_share"],
                                                 repair_rel_typical=last["repair_rel_typical"],
                                                 within_0p01_by_period=settle),
                                    C_min_overall=float(rec["C"].min()))
        thr = bc.interp_loglog(np.clip(rec["C"], Cs[0], Cs[-1]), Cs, ups)
        x = rec["t"] / PERIOD      # time in periods

        g = outer[r].subgridspec(2, 1, height_ratios=[0.42, 1.0], hspace=0.12)
        axd = fig.add_subplot(g[0]); axm = fig.add_subplot(g[1], sharex=axd)
        for ax in (axd, axm):
            for k in range(N_PER):
                ax.axvspan(k, k + T_TERM / PERIOD, color=SHADE, lw=0, zorder=0)
            st.grid(ax)
        axd.plot(x, rec["u"], color=st.INK3, lw=1.0)
        axd.plot(x, thr, color=st.INK, lw=0.8, ls=(0, (1, 2)))
        axd.set_ylim(0, 0.27); axd.set_yticks([0, 0.1, 0.2]); axd.set_ylabel("Demand")
        plt.setp(axd.get_xticklabels(), visible=False)
        axm.plot(x, rec["s"], color=st.BLUE, lw=1.0)
        axm.plot(x, rec["C"], color=st.BLUE_D, lw=1.6)
        axm.plot(x, rec["eta"] / bc.ETA_TYPICAL, color=st.AQUA, lw=1.4)
        axm.set_ylim(-0.03, 1.05); axm.set_xlim(0, N_PER)
        axm.set_xticks(range(0, N_PER + 1, 5))
        axm.set_ylabel("Engagement,\nceiling,\nrestoration rate")
        if r == len(ROWS) - 1:
            axm.set_xlabel("Time, in periods of demand and relief")
        axd.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                     fontsize=8, color=st.INK, ha="left", va="bottom")
        st.letter(axd, lab, dx=-40)
        fig.align_ylabels([axd, axm])

    handles = [Line2D([], [], color=st.INK3, lw=1.0), Line2D([], [], color=st.BLUE, lw=1.0),
               Line2D([], [], color=st.INK, lw=0.8, ls=(0, (1, 2))), Line2D([], [], color=st.BLUE_D, lw=1.6),
               Patch(fc=SHADE, ec="none"), Line2D([], [], color=st.AQUA, lw=1.4)]
    labels = ["demand", "engagement, $s$",
              "collapse threshold (falls with the ceiling)", "capacity ceiling, $C$",
              "period of demand", "restoration rate, relative to typical"]
    st.legend_below(fig, handles, labels, ncol=3, y=0.06)
    png = st.save(fig, "fig6_burnout")
    json.dump(numbers, open(os.path.join(st.OUT, "fig6_numbers.json"), "w"), indent=1, default=float)
    for lab in "abc":
        rr = numbers["rows"][lab]
        print(lab, "settled", {k: round(v, 3) if isinstance(v, float) else v for k, v in rr["settled"].items()},
              "C_min", round(rr["C_min_overall"], 3))
        print("   per period (C after relief, functioning share):",
              [(p["period"], round(p["C_after_relief"], 3), round(p["functioning_share"], 2)) for p in rr["periods"]])
    print(png)
