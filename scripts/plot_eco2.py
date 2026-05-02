# -*- coding: utf-8 -*-
"""
===========================
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import MaxNLocator

OUTPUT_PREFIX = 'Figure_Scatter'
try:
    train_df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    raise

if OBSERVED_COL not in train_df.columns:
if PREDICTED_COL not in train_df.columns:

observed = train_df[OBSERVED_COL]
predicted = train_df[PREDICTED_COL]

nse = 1 - np.sum((predicted - observed) ** 2) / np.sum((observed - np.mean(observed)) ** 2)
rmse = np.sqrt(mean_squared_error(observed, predicted))
correlation_matrix = np.corrcoef(observed, predicted)
r2 = correlation_matrix[0, 1] ** 2

print(f"RMSE = {rmse:.4f}")
print(f"NSE  = {nse:.4f}")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': 'black',
    'ps.fonttype': 42
})

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH), dpi=100)

sns.scatterplot(
    x=observed, y=predicted,
    color="darkslateblue", alpha=SCATTER_ALPHA,
    edgecolor="black",
    s=SCATTER_SIZE,
    linewidth=SCATTER_LINEWIDTH,
    label="Data Points",
    ax=ax
)

ax.plot([observed.min(), observed.max()], [observed.min(), observed.max()],
        "r--", linewidth=LINE_WIDTH, label="1:1 Line")

z = np.polyfit(observed, predicted, 1)
p = np.poly1d(z)
ax.plot(observed, p(observed), "forestgreen", linewidth=LINE_WIDTH,
        label=f"Fitted: y={z[0]:.2f}x+{z[1]:.2f}")

ax.text(0.05, 0.85, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nNSE = {nse:.3f}",
        transform=ax.transAxes, fontsize=TEXT_BOX_SIZE, color="black",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="black", linewidth=0.8))

ax.set_xlabel("Observed Runoff ($m^3$/s)", fontsize=AXIS_LABEL_SIZE, color="black")
ax.set_ylabel("Predicted Runoff ($m^3$/s)", fontsize=AXIS_LABEL_SIZE, color="black")
ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, colors="black", width=0.8, length=3)

for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(0.8)

ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax.grid(False, which='minor')
ax.grid(True, which='major', linestyle='-', color='black', alpha=1.0, linewidth=0.8)
ax.set_axisbelow(True)

lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)

ax.legend(fontsize=LEGEND_SIZE, loc="lower right", frameon=True,
          facecolor="white", edgecolor="black")

plt.tight_layout()
plt.savefig(f'{OUTPUT_PREFIX}.eps', format='eps', dpi=OUTPUT_DPI, bbox_inches='tight')

plt.savefig(f'{OUTPUT_PREFIX}.pdf', format='pdf', dpi=OUTPUT_DPI, bbox_inches='tight')

plt.savefig(f'{OUTPUT_PREFIX}.tif', format='tiff', dpi=OUTPUT_DPI, bbox_inches='tight')

plt.savefig(f'{OUTPUT_PREFIX}.png', format='png', dpi=OUTPUT_DPI, bbox_inches='tight')

plt.close()

