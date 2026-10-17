"""Figure: more own data -> fewer big decision errors, but more decisions come too late."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "baseline.csv")
g = df.groupby("threshold").agg(
    big=("regret_global", lambda r: 100 * (r > 0.05).mean()),
    late=("too_late", lambda r: 100 * r.mean()),
    k=("k_decide", "median"),
).sort_index(ascending=False)

INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200, facecolor=BG)
ax.set_facecolor(BG)
x = range(len(g))
ax.plot(x, g.big, color=BLUE, lw=2, marker="o", ms=7, mec=BG, mew=2, label="Large error (utility loss > 0.05)")
ax.plot(x, g.late, color=ORANGE, lw=2, marker="o", ms=7, mec=BG, mew=2, label="Too late (best time already passed)")
for xi, (b, l) in enumerate(zip(g.big, g.late)):
    hi, lo = (b, l) if b > l else (l, b)          # higher point labelled above, lower below
    ax.annotate(f"{hi:.0f}%", (xi, hi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK2)
    ax.annotate(f"{lo:.0f}%", (xi, lo), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=INK2)
ax.set_xticks(list(x), [f"{t:.0%}\n~{k:.0f} cycles" for t, k in zip(g.index, g.k)], fontsize=8, color=INK2)
ax.set_xlabel("Twin decides when capacity reaches … (more own data →)", fontsize=9, color=INK2)
ax.set_ylabel("% of decisions (82 cells × 3 runs)", fontsize=9, color=INK2)
ax.set_ylim(-6, 70)
ax.grid(axis="y", color=GRID, lw=0.8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
ax.set_title("Original battery twin: waiting for data trades big errors for late decisions",
             fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
out = ROOT / "results" / "tradeoff_baseline.png"
fig.savefig(out, facecolor=BG)
print("saved", out)
