"""
Figure 1. The regime boundary as person-environment geometry (schematic).

Drawn in the shared style of Figures 2 to 6, with collapse on the right as in
Figure 2, and worded in the paper's terms: functioning, boundary, collapse and
margin (the distance from the boundary; effective capacity is the separate
spare-capacity term).

  (a) basic geometry: a person's state, its ordinary fluctuation, the boundary
      and the margin between them
  (b) the environment shifts the boundary toward the person
  (c) people differ in how close they sit, so some cross more often
  (d) the same person, under lower and higher demand
The horizontal axis is schematic and carries no numbers.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clear"))
import style as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

X_MAX = 12.0
DARK, BAR, DOT = st.BLUE_D, st.BLUE_L, st.BLUE


def band(ax, x_b, y0, y1):
    """Collapse region to the right of the boundary, within one row."""
    ax.add_patch(Rectangle((x_b, y0), X_MAX - x_b, y1 - y0, facecolor=DARK, edgecolor="none", zorder=1))


def person(ax, x, spread, y, label=None):
    ax.plot([x - spread, x + spread], [y, y], color=BAR, lw=6, solid_capstyle="round", zorder=3)
    ax.plot(x, y, "o", color=DOT, ms=6.5, zorder=4)
    if label:
        ax.annotate(label, xy=(x, y), xytext=(0, 6), textcoords="offset points", fontsize=st.ANN,
                    color=st.INK2, ha="center", va="bottom", zorder=5)


def span_arrow(ax, x0, x1, y, text, below=True):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<|-|>", color=st.INK2, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.annotate(text, xy=(0.5 * (x0 + x1), y), xytext=(0, -3 if below else 3), textcoords="offset points",
                fontsize=st.ANN, color=st.INK2, ha="center", va="top" if below else "bottom")


def label(ax, x, y, text, ha="left", va="top", color=None):
    ax.text(x, y, text, fontsize=st.ANN, color=color or st.INK2, ha=ha, va=va, zorder=5)


def frame(ax, title, letter):
    ax.set_xlim(0, X_MAX); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                fontsize=8, ha="left", va="bottom")
    st.letter(ax, letter, dx=-14)


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 2, figsize=(st.DOUBLE, 4.4))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.04, wspace=0.12, hspace=0.30)
    (a, b), (c, d) = axs

    # ---------------- (a) basic geometry
    frame(a, "basic geometry", "a")
    band(a, 9.0, 0.12, 0.88)
    label(a, 0.2, 0.86, "functioning")
    label(a, 10.5, 0.86, "collapse", ha="center", color="white")
    a.annotate("boundary", xy=(9.0, 0.88), xytext=(0, 2), textcoords="offset points", fontsize=st.ANN,
               color=st.INK2, ha="center", va="bottom")
    person(a, 4.0, 2.0, 0.56, "person")
    span_arrow(a, 4.0, 9.0, 0.38, "margin")
    a.annotate("", xy=(11.8, 0.03), xytext=(0.2, 0.03),
               arrowprops=dict(arrowstyle="-|>", color=st.INK3, lw=0.6, mutation_scale=6))
    label(a, 8.8, 0.07, "towards collapse", ha="right", va="bottom", color=st.INK3)

    # ---------------- (b) the environment shifts the boundary
    frame(b, "environment shifts the boundary", "b")
    band(b, 9.0, 0.58, 0.94); band(b, 6.0, 0.06, 0.42)
    label(b, 0.2, 0.93, "lower demand"); label(b, 0.2, 0.41, "higher demand")
    person(b, 4.0, 2.0, 0.77); person(b, 4.0, 2.0, 0.25)
    span_arrow(b, 4.0, 9.0, 0.66, "margin")
    b.annotate("", xy=(6.0, 0.14), xytext=(4.0, 0.14),
               arrowprops=dict(arrowstyle="<|-|>", color=st.INK2, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0))
    label(b, 3.8, 0.14, "less margin", ha="right", va="center")
    b.annotate("", xy=(6.05, 0.50), xytext=(8.95, 0.50),
               arrowprops=dict(arrowstyle="-|>", color=st.INK2, lw=0.8, mutation_scale=7))
    label(b, 9.1, 0.50, "demand rises", va="center")

    # ---------------- (c) individual differences
    frame(c, "individual differences", "c")
    band(c, 7.0, 0.06, 0.94)
    c.annotate("boundary", xy=(7.0, 0.94), xytext=(0, 2), textcoords="offset points", fontsize=st.ANN,
               color=st.INK2, ha="center", va="bottom")
    for (x, sp, y, name, how) in ((2.0, 1.0, 0.76, "A", "rarely crosses"),
                                  (5.0, 2.0, 0.49, "B", "sometimes crosses"),
                                  (6.5, 2.0, 0.22, "C", "often crosses")):
        person(c, x, sp, y, name)
        label(c, 11.8, y, how, ha="right", va="center", color="white")

    # ---------------- (d) the same person under lower and higher demand
    frame(d, "person × environment", "d")
    band(d, 9.0, 0.58, 0.94); band(d, 6.0, 0.06, 0.42)
    label(d, 0.2, 0.93, "lower demand"); label(d, 0.2, 0.41, "higher demand")
    person(d, 5.0, 2.0, 0.77); person(d, 5.0, 2.0, 0.25)
    label(d, 5.0, 0.69, "stays functioning", ha="center")
    label(d, 5.9, 0.17, "crosses into collapse", ha="right")

    png = st.save(fig, "fig1_boundary")
    print(png)
