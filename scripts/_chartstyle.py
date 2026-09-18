"""Single source of truth for notebook chart style: one palette, one width, one DPI, one font scale.

All chart-producing code (build_eda_artifacts.py, build_analysis_artifacts.py, and the notebook setup
cell) imports this so every figure in RESULTS.ipynb looks uniform. Import and call apply() once.
"""
import matplotlib as mpl

# palette (consistent semantic roles across the report)
GREEN = "#1d9e75"   # primary / "good"
BLUE = "#185fa5"    # secondary
PURPLE = "#6a5acd"  # tertiary (sweeps)
AMBER = "#e08a00"   # caution / unsure
RED = "#d73027"     # negative / "bad"
GREY = "#9aa0a6"    # reference / muted
SEQ = [GREEN, BLUE, PURPLE, AMBER, RED]

# one width for every figure; a small set of heights by content density
FIG_W = 7.2
H_S = 3.2     # standard single chart
H_M = 4.6     # many rows (long bar charts)
H_SQ = 6.0    # square-ish (heatmaps)
DPI = 120


def apply():
    mpl.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI, "figure.figsize": (FIG_W, H_S),
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#bdbdbd",
        "axes.grid": True, "grid.color": "#e9e9e9", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SEQ),
        # NB: no savefig.bbox="tight" — a fixed canvas keeps every saved figure the same pixel width
        # (FIG_W*DPI); tight_layout() is called per-figure to keep labels inside that canvas.
    })
