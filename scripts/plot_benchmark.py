"""Generate benchmark_results.png from saved benchmark numbers."""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data ────────────────────────────────────────────────────────────────────

datasets = ["MNIST", "CIFAR-10"]

plain_acc  = [99.7, 90.2]
emvp_acc   = [99.7, 90.1]
agree      = [100.0, 99.9]

mean_l2    = [0.0208, 0.1141]
max_l2     = [0.0648, 0.2542]

plain_ms   = [8.86,  14.27]
emvp_ms    = [55.0,  84.0]
slowdown   = [6,     6]

# ── Style ────────────────────────────────────────────────────────────────────

PLAIN_COLOR = "#4C72B0"
EMVP_COLOR  = "#DD8452"
BG          = "#F8F8F8"
GRID        = "#E0E0E0"

fig = plt.figure(figsize=(12, 8), facecolor="white")
fig.suptitle(
    "EMVP-Protected ResNet vs Plaintext  —  1000 Test Images",
    fontsize=15, fontweight="bold", y=0.97,
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38,
                       left=0.07, right=0.97, top=0.90, bottom=0.08)

# ── Helper ───────────────────────────────────────────────────────────────────

def styled_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── 1. Accuracy bar chart ────────────────────────────────────────────────────

ax1 = fig.add_subplot(gs[0, 0])
styled_ax(ax1, "Accuracy (%)")

x = np.arange(len(datasets))
w = 0.32
b1 = ax1.bar(x - w/2, plain_acc, w, label="Plaintext", color=PLAIN_COLOR, zorder=3)
b2 = ax1.bar(x + w/2, emvp_acc,  w, label="EMVP",      color=EMVP_COLOR,  zorder=3)

ax1.set_xticks(x); ax1.set_xticklabels(datasets)
ax1.set_ylim(88, 101)
ax1.legend(fontsize=9, framealpha=0.9)

for bar, val in zip(list(b1) + list(b2), plain_acc + emvp_acc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5)


# ── 2. Prediction agreement ──────────────────────────────────────────────────

ax2 = fig.add_subplot(gs[0, 1])
styled_ax(ax2, "Prediction Agreement (%)")

bars = ax2.bar(datasets, agree, color=["#55A868", "#C44E52"], zorder=3, width=0.4)
ax2.set_ylim(99.5, 100.15)
for bar, val in zip(bars, agree):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9)


# ── 3. Logit error ───────────────────────────────────────────────────────────

ax3 = fig.add_subplot(gs[0, 2])
styled_ax(ax3, "L2 Logit Error")

x = np.arange(len(datasets))
bm = ax3.bar(x - w/2, mean_l2, w, label="Mean", color="#8172B2", zorder=3)
bx = ax3.bar(x + w/2, max_l2,  w, label="Max",  color="#CCB974", zorder=3)

ax3.set_xticks(x); ax3.set_xticklabels(datasets)
ax3.legend(fontsize=9, framealpha=0.9)

for bar, val in zip(list(bm) + list(bx), mean_l2 + max_l2):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8)


# ── 4. Speed (ms/image) ──────────────────────────────────────────────────────

ax4 = fig.add_subplot(gs[1, 0])
styled_ax(ax4, "Speed (ms / image)")

x = np.arange(len(datasets))
b1 = ax4.bar(x - w/2, plain_ms, w, label="Plaintext", color=PLAIN_COLOR, zorder=3)
b2 = ax4.bar(x + w/2, emvp_ms,  w, label="EMVP",      color=EMVP_COLOR,  zorder=3)

ax4.set_xticks(x); ax4.set_xticklabels(datasets)
ax4.set_ylabel("ms", fontsize=9)
ax4.legend(fontsize=9, framealpha=0.9)

for bar, val in zip(list(b1) + list(b2), plain_ms + emvp_ms):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.0f}", ha="center", va="bottom", fontsize=8.5)


# ── 5. Slowdown ──────────────────────────────────────────────────────────────

ax5 = fig.add_subplot(gs[1, 1])
styled_ax(ax5, "EMVP Slowdown vs Plaintext")

bars = ax5.bar(datasets, slowdown, color=EMVP_COLOR, zorder=3, width=0.4)
ax5.set_ylim(0, 9)
ax5.set_ylabel("× slower", fontsize=9)
for bar, val in zip(bars, slowdown):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f"{val}×", ha="center", va="bottom", fontsize=11, fontweight="bold")


# ── 6. Summary table ─────────────────────────────────────────────────────────

ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")

rows = [
    ["Plaintext accuracy",   "99.7%",   "90.2%"],
    ["EMVP accuracy",        "99.7%",   "90.1%"],
    ["Accuracy loss",        "0.0%",    "0.1%"],
    ["Pred agreement",       "100.0%",  "99.9%"],
    ["Mean L2 error",        "0.021",   "0.114"],
    ["Plaintext speed",      "8.9 ms",  "14.3 ms"],
    ["EMVP speed",           "55 ms",   "84 ms"],
    ["Slowdown",             "6×",      "6×"],
]
col_labels = ["Metric", "MNIST", "CIFAR-10"]

tbl = ax6.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.38)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID)
    if r == 0:
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#EEF2F5")
    else:
        cell.set_facecolor("white")

ax6.set_title("Summary", fontsize=11, fontweight="bold", pad=8)

# ── Save ─────────────────────────────────────────────────────────────────────

out = os.path.join(_ROOT, "results", "benchmark_results.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
