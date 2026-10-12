"""Publication style shared by the CLEAR figures. See FIGURE_SPEC.md."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.environ.get("CLEAR_FIG_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output"))
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_L, BLUE_D = "#b7d3f6", "#104281"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8985"
GRID = "#e6e5e1"
REGION = dict(functioning=BLUE_L, bistable=ORANGE, collapsed=BLUE_D)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.2,
    "figure.dpi": 100, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

SINGLE, DOUBLE = 3.4, 7.0
ROW, STACK = 2.4, 1.9
ANN = 7   # annotation point size, the minimum allowed

U_TICKS = [0.003, 0.01, 0.03, 0.1, 0.3]


def letter(ax, s, dx=-30):
    """Panel letter outside the axes, top left, same offset everywhere (dx only
    when a long two-line y label would otherwise sit under it)."""
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, 6), textcoords="offset points",
                fontsize=10, fontweight="bold", ha="left", va="bottom")


def grid(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def demand_axis(ax, lo=0.003, hi=0.4, label=True):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks([t for t in U_TICKS if lo <= t <= hi])
    ax.set_xticklabels([f"{t:g}" for t in U_TICKS if lo <= t <= hi])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if label:
        ax.set_xlabel("Demand inflow, $u$")


def thresholds(ax, u_up, u_rec, label=False):
    for x in (u_up, u_rec):
        ax.axvline(x, color=INK3, lw=0.7, ls=(0, (1, 2)), zorder=1)
    if label:
        for x, s, ha in ((u_up, "collapse", "left"), (u_rec, "recovery", "right")):
            ax.annotate(s, xy=(x, 1), xycoords=("data", "axes fraction"), xytext=(3 if ha == "left" else -3, 2),
                        textcoords="offset points", fontsize=ANN, color=INK2, ha=ha, va="bottom")


def legend_below(fig, handles, labels, ncol, y=-0.02):
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y), fontsize=8,
               handlelength=1.8, columnspacing=1.6)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    return os.path.join(OUT, name + ".png")
