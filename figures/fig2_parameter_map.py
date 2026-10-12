"""
Figure 2. Where the boundary sits, by person parameter.

Three panels share the demand axis u. Each varies one of the parameters the
paper hypothesises for ADHD, holding the others at the reference set:
  (a) affect feedback gain g            (reactivity of affect to regulatory failure)
  (b) anticipated reward R             (scales with anticipatory transfer rho_a
                                         at fixed appraised value)
  (c) maximum capacity C_max (capacity held at it; absolute load units; C_eff = (C-L)+/C)
Cells are classified by counting stable equilibria (scalar reduction, no
multistart): one functioning state (s > 1/2, i.e. k > 1), two states (bistable), or one collapsed
state. The reference person is R = 2, g = 2.5, C = 1, n = 4.

The reference-person label reads Figs 3 to 6, which use the same person, and
sits on a light box so it reads over the dark region.
"""
import numpy as np, sys, json
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "clear"))
import clear_core as cc
import style as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

NU, NP = 110, 90
us = np.geomspace(0.003, 0.5, NU)

AXES = [
    ("g", np.linspace(0.0, 6.0, NP), "Affect feedback gain, $g$", "up", 2.5),
    ("R", np.linspace(0.5, 5.0, NP), "Anticipated reward, $R$", "down", 2.0),
    ("C", np.geomspace(0.05, 1.5, NP), "Maximum capacity, $C_{\\mathrm{max}}$ (log)", "down", 1.0),   # log spacing so the near-zero region shows
]
LOG_AXES = {"C"}


def classify(u, p):
    st_ = cc.stable_states(u, p, N=1501)
    if len(st_) >= 2:
        return 1
    if not st_:
        return 2
    # single state: functioning if the goal dominates there, s > 1/2, which is k > k* = 1
    # for every n (the paper's own crossover); collapsed otherwise
    return 0 if st_[0]["s"] > 0.5 else 2


def compute():
    maps = {}
    for key, vals, _, _, _ in AXES:
        Z = np.zeros((NP, NU), dtype=int)
        for i, v in enumerate(vals):
            p = dict(cc.REF, **{key: float(v)})
            for j, u in enumerate(us):
                Z[i, j] = classify(u, p)
        maps[key] = Z
    return maps


if __name__ == "__main__":
    maps = compute()
    np.savez(st.OUT + "/fig2_maps.npz", us=us, **{k: maps[k] for k in maps},
             **{k + "_vals": v for k, v, *_ in AXES})

    cmap = ListedColormap([st.REGION["functioning"], st.REGION["bistable"], st.REGION["collapsed"]])
    fig, axes = plt.subplots(1, 3, figsize=(st.DOUBLE, st.ROW))
    for ax, (key, vals, label, direction, ref), lab in zip(axes, AXES, "abc"):
        Z = maps[key]
        ax.pcolormesh(us, vals, Z, cmap=cmap, vmin=-0.5, vmax=2.5, shading="nearest", rasterized=True)
        st.demand_axis(ax, lo=us[0], hi=us[-1])
        if key in LOG_AXES:
            ax.set_yscale("log"); ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(vals[0], vals[-1])
        ax.axhline(ref, color=st.INK, lw=0.7, ls=(0, (2, 2)))
        ax.annotate("reference person (Figs 3 to 6)", xy=(us[-1], ref), xytext=(-3, 2), textcoords="offset points",
                    fontsize=st.ANN, color=st.INK, ha="right", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
        # arrow in the hypothesised ADHD direction, placed by axis fraction so it works on log axes too
        tr = (lambda f: vals[0] * (vals[-1] / vals[0]) ** f) if key in LOG_AXES else (lambda f: vals[0] + f * (vals[-1] - vals[0]))
        f0, f1 = (0.58, 0.86) if direction == "up" else (0.30, 0.06)
        y0, y1 = tr(f0), tr(f1)
        xa = 0.3
        ax.annotate("", xy=(xa, y1), xytext=(xa, y0),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=0.9, mutation_scale=8))
        ax.annotate("hypothesised\nADHD direction", xy=(xa, tr(0.5 * (f0 + f1))), xytext=(-5, 0), textcoords="offset points",
                    fontsize=st.ANN, color="white", ha="right", va="center")
        ax.set_ylabel(label)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(True); ax.spines[sp].set_color(st.INK2)
        st.letter(ax, lab)
    handles = [Patch(fc=st.REGION["functioning"], label="only a functioning state exists"),
               Patch(fc=st.REGION["bistable"], label="both states exist; the path decides which"),
               Patch(fc=st.REGION["collapsed"], label="only a collapsed state exists")]
    fig.tight_layout(w_pad=1.6)
    st.legend_below(fig, handles, [h.get_label() for h in handles], ncol=3, y=-0.02)
    png = st.save(fig, "fig2_parameter_map")
    # numbers for the caption
    out = {}
    for key, vals, *_ in AXES:
        Z = maps[key]
        out[key] = {"bistable_fraction": float((Z == 1).mean())}
    for key, v in [("R", 1.5), ("R", 2.0), ("R", 3.0), ("g", 1.0), ("g", 2.5), ("g", 5.0), ("C", 0.1), ("C", 0.2), ("C", 0.5), ("C", 1.0)]:
        out[f"folds_{key}={v}"] = cc.folds(dict(cc.REF, **{key: v}))
    json.dump(out, open(st.OUT + "/fig2_numbers.json", "w"), indent=1)
    print(json.dumps(out, indent=1)); print(png)
