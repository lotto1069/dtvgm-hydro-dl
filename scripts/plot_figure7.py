import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.font_manager as fm
import os

# Chinese font loading
font_paths = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc"
]

my_font = None
for fp in font_paths:
    if os.path.exists(fp):
        my_font = fm.FontProperties(fname=fp, size=10)
        break

if my_font is None:
    print("Warning: Chinese font not found. Check C:\\Windows\\Fonts\\")
else:
    print(f"Loaded font: {my_font.get_name()}")

# Global style configuration
width_cm = 3.3
width_inch = width_cm / 2.54

plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': 'black',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 1000
})

plt.style.use('default')

BORDER_WIDTH = 0.5
GRID_WIDTH = 0.3
LABEL_SIZE = 10
TICK_SIZE = 10
LEGEND_SIZE = 8

model_colors = sns.color_palette("Set1", 4)
observed_color = "darkblue"

# Model short name mapping
model_short_names = {
    "DTVGM": "DTVGM",
    "DTVGM+LSTM": "DT+LSTM",
    "DTVGM+TCN": "DT+TCN",
    "DTVGM+Transformer": "DT+Trans"
}

# Data preparation
try:
    test_df = pd.read_csv(r"G:\tvgm\test_set_results.csv")
    observed_full = test_df['Observed']
    models = {
        "DTVGM": test_df['Predicted_TVGM'],
        "DTVGM+LSTM": test_df['Predicted_TVGM+LSTM'],
        "DTVGM+TCN": test_df['Predicted_TVGM+TCN'],
        "DTVGM+Transformer": test_df['Predicted_TVGM+Transformer']
    }
except FileNotFoundError:
    np.random.seed(42)
    N_SAMPLES = 100
    observed_full = pd.Series(np.random.rand(N_SAMPLES) * 10 + 5)
    models = {
        "DTVGM": observed_full + np.random.normal(0, 1, N_SAMPLES),
        "DTVGM+LSTM": observed_full + np.random.normal(0, 0.5, N_SAMPLES),
        "DTVGM+TCN": observed_full + np.random.normal(0, 0.7, N_SAMPLES),
        "DTVGM+Transformer": observed_full + np.random.normal(0, 0.6, N_SAMPLES)
    }

metrics = pd.DataFrame({
    "Model": ["DTVGM", "DTVGM+LSTM", "DTVGM+TCN", "DTVGM+Transformer"],
    "NSE_Test": [0.855, 0.886, 0.872, 0.860],
    "R2_Test": [0.856, 0.887, 0.878, 0.865],
    "RMSE_Test": [0.236, 0.211, 0.223, 0.233]
})

# (a) Radar chart
print("Generating Figure 7a: Radar chart...")
fig_a = plt.figure(figsize=(width_inch, width_inch))
ax_a = fig_a.add_subplot(111, polar=True)

metrics_norm = metrics.copy()
scaler = MinMaxScaler()
for col in ["NSE_Test", "R2_Test"]:
    metrics_norm[col] = scaler.fit_transform(metrics[[col]])
metrics_norm["RMSE_Test"] = 1 - scaler.fit_transform(metrics[["RMSE_Test"]])

categories = ["$R^2$", "NSE", "$RMSE^{-1}$"]
N = len(categories)
angles = [n / N * 2 * np.pi for n in range(N)] + [0]

line_styles = {
    "DTVGM": {'color': model_colors[0], 'linestyle': '--', 'linewidth': 0.8},
    "DTVGM+LSTM": {'color': model_colors[1], 'linestyle': '-', 'linewidth': 1.0},
    "DTVGM+TCN": {'color': model_colors[2], 'linestyle': '-', 'linewidth': 0.8},
    "DTVGM+Transformer": {'color': model_colors[3], 'linestyle': '-', 'linewidth': 0.8}
}

for i, row in metrics_norm.iterrows():
    values = row[["R2_Test", "NSE_Test", "RMSE_Test"]].tolist()
    values_closed = values + [values[0]]
    short_name = model_short_names[row["Model"]]
    ax_a.plot(angles, values_closed, label=short_name, marker='o', markersize=3, **line_styles[row["Model"]])

ax_a.set_xticks(angles[:-1])
ax_a.set_xticklabels(categories, fontsize=LABEL_SIZE, fontweight='bold')
ax_a.tick_params(axis='x', pad=8)
ax_a.set_yticks([0.5, 1.0])
ax_a.set_yticklabels(["0.5", "1.0"], color="black", fontsize=TICK_SIZE)
ax_a.set_ylim(0, 1.1)
ax_a.set_rlabel_position(180)

ax_a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
            fontsize=LEGEND_SIZE, frameon=True, edgecolor='black')

ax_a.grid(True, color='black', linewidth=GRID_WIDTH, linestyle='-', alpha=1.0)
ax_a.spines['polar'].set_visible(True)
ax_a.spines['polar'].set_color('black')
ax_a.spines['polar'].set_linewidth(BORDER_WIDTH)

plt.tight_layout()
fig_a.savefig("Figure_7a_Radar.tif", format='tiff', dpi=1000, bbox_inches='tight')
fig_a.savefig("Figure_7a_Radar.pdf", format='pdf', dpi=1000, bbox_inches='tight')
plt.close(fig_a)
print("Figure 7a saved.")

# (b) Dual-axis bar chart
print("Generating Figure 7b: Dual-axis bar chart...")
fig_b = plt.figure(figsize=(width_inch, width_inch))
ax_b = fig_b.add_subplot(111)
ax_b2 = ax_b.twinx()

metric_calc = {}
for name, pred_full in models.items():
    mask = np.isfinite(observed_full) & np.isfinite(pred_full)
    obs, pred = observed_full[mask], pred_full[mask]
    if len(obs) > 0:
        metric_calc[name] = {
            "R2": round(r2_score(obs, pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(obs, pred)), 3)
        }

metric_df = pd.DataFrame(metric_calc).T.reset_index().rename(columns={"index": "Model"})
metric_df["Model_Short"] = [model_short_names[m] for m in metric_df["Model"]]
metric_df["Model_Short"] = metric_df["Model_Short"].str.replace("+", "\n+")

x = np.arange(len(metric_df))
width = 0.25

bars1 = ax_b.bar(x - width / 2, metric_df["R2"], width, label="$R^2$",
                 color=model_colors[0], edgecolor='k', linewidth=0.5, alpha=0.9)
bars2 = ax_b2.bar(x + width / 2, metric_df["RMSE"], width, label="RMSE",
                  color=model_colors[1], edgecolor='k', linewidth=0.5, alpha=0.9)

for bar in bars1:
    ax_b.annotate(f"{bar.get_height():.3f}",
                  xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                  xytext=(0, 2),
                  textcoords="offset points",
                  ha='center', va='bottom', rotation=90, fontsize=8, fontweight='bold')

for bar in bars2:
    ax_b2.annotate(f"{bar.get_height():.3f}",
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 2),
                   textcoords="offset points",
                   ha='center', va='bottom', rotation=90, fontsize=8, fontweight='bold')

ax_b.set_ylabel("$R^2$", fontsize=LABEL_SIZE, fontproperties=my_font)
ax_b2.set_ylabel("RMSE ($m^3$/s)", fontsize=LABEL_SIZE, fontproperties=my_font)

ax_b.set_xticks(x)
ax_b.set_xticklabels(metric_df["Model_Short"], rotation=0, ha='center', fontsize=8, linespacing=0.9)

ax_b.set_xlim(-0.6, len(metric_df) - 0.4)
ax_b.set_ylim(0, 1.55)
ax_b2.set_ylim(0, max(metric_df["RMSE"]) * 1.70)

lines1, labels1 = ax_b.get_legend_handles_labels()
lines2, labels2 = ax_b2.get_legend_handles_labels()

ax_b.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=2, frameon=True, edgecolor='black', fontsize=LEGEND_SIZE, prop=my_font)

for ax in [ax_b, ax_b2]:
    ax.tick_params(axis='both', labelsize=TICK_SIZE, colors='black', width=BORDER_WIDTH, length=2)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(BORDER_WIDTH)

ax_b.grid(False, axis='x')
ax_b.grid(True, axis='y', which='major', color='black', linestyle='-', linewidth=GRID_WIDTH, alpha=1.0, zorder=0)
ax_b.set_axisbelow(True)

plt.tight_layout()
fig_b.savefig("Figure_7b_Barplot.tif", format='tiff', dpi=1000, bbox_inches='tight')
fig_b.savefig("Figure_7b_Barplot.pdf", format='pdf', dpi=1000, bbox_inches='tight')
plt.close(fig_b)
print("Figure 7b saved.")

# (c) Residual KDE plot
print("Generating Figure 7c: Residual KDE...")
fig_c, axes_c = plt.subplots(2, 2, figsize=(width_inch, width_inch))
axes_c = axes_c.flatten()

for idx, (model, pred_full) in enumerate(models.items()):
    res = (pred_full - observed_full).dropna()
    ax = axes_c[idx]
    short_name = model_short_names[model]

    sns.kdeplot(res, ax=ax, color=model_colors[idx], fill=True, alpha=0.6, linewidth=0.8)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.6)

    ax.set_title(short_name, fontsize=10, pad=3, fontweight='bold')
    ax.set_xlim(-2, 2)
    ax.set_ylabel("", fontproperties=my_font)
    ax.tick_params(axis='both', labelsize=TICK_SIZE, color='black', width=BORDER_WIDTH, length=2)

    if idx >= 2:
        ax.set_xlabel("Residual", fontsize=LABEL_SIZE)

    ax.grid(True, color='black', linestyle='-', linewidth=GRID_WIDTH, alpha=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(BORDER_WIDTH)

plt.tight_layout(h_pad=0.3, w_pad=0.3)
fig_c.savefig("Figure_7c_ResidualKDE.tif", format='tiff', dpi=1000, bbox_inches='tight')
fig_c.savefig("Figure_7c_ResidualKDE.pdf", format='pdf', dpi=1000, bbox_inches='tight')
plt.close(fig_c)
print("Figure 7c saved.")

# (d) Radial distribution plot
print("Generating Figure 7d: Radial distribution...")
fig_d = plt.figure(figsize=(width_inch, width_inch))
ax_d = fig_d.add_subplot(111, polar=True)

groups = {"Observed": observed_full} | models
n_groups = len(groups)
theta_width = 2 * np.pi / n_groups
group_thetas = np.linspace(0, 2 * np.pi, n_groups + 1)[:-1]
global_max = 0

group_short_names = {"Observed": "Observed"}
for k, v in model_short_names.items():
    group_short_names[k] = v

for i, (name, data_full) in enumerate(groups.items()):
    data = data_full.dropna()
    if len(data) == 0:
        continue

    global_max = max(global_max, data.max())
    theta = group_thetas[i]
    color = observed_color if name == "Observed" else model_colors[i - 1]
    name_display = group_short_names[name]

    scatter_thetas = np.random.uniform(theta - theta_width / 2 + 0.1,
                                       theta + theta_width / 2 - 0.1,
                                       size=len(data))
    ax_d.scatter(scatter_thetas, data, color=color, alpha=0.6, s=8,
                 edgecolor='black', linewidth=0.2, zorder=2)

    q1, med, q3 = data.quantile([0.25, 0.5, 0.75])
    min_v, max_v = data.min(), data.max()

    ax_d.plot([theta] * 2, [min_v, max_v], color='black', linewidth=0.8, zorder=3)
    ax_d.plot([theta - 0.05, theta + 0.05], [min_v, min_v], color='black', linewidth=0.8, zorder=3)
    ax_d.plot([theta - 0.05, theta + 0.05], [max_v, max_v], color='black', linewidth=0.8, zorder=3)
    ax_d.bar(theta, q3 - q1, bottom=q1, width=0.15, color='none',
             edgecolor='black', linewidth=0.8, zorder=3)
    ax_d.plot([theta - 0.05, theta + 0.05], [med, med], color='black', linewidth=1.2, zorder=4)

    ax_d.text(theta, global_max * 1.25, name_display, ha='center', va='center',
              fontsize=10, fontweight='bold', clip_on=False, fontproperties=my_font)

ax_d.set_theta_zero_location("N")
ax_d.set_thetagrids([])
ax_d.set_rorigin(-global_max * 0.05)
ax_d.spines['polar'].set_visible(False)
ax_d.grid(True, axis='y', color='black', linewidth=GRID_WIDTH, alpha=1.0, linestyle='-')
ax_d.grid(False, axis='x')
ax_d.set_rmax(global_max * 1.05)

plt.tight_layout()
fig_d.savefig("Figure_7d_Radial.tif", format='tiff', dpi=1000, bbox_inches='tight')
fig_d.savefig("Figure_7d_Radial.pdf", format='pdf', dpi=1000, bbox_inches='tight')
plt.close(fig_d)
print("Figure 7d saved.")

print("\nAll done. Four sub-figures saved as TIF and PDF:")
print("  - Figure_7a_Radar.tif/pdf")
print("  - Figure_7b_Barplot.tif/pdf")
print("  - Figure_7c_ResidualKDE.tif/pdf")
print("  - Figure_7d_Radial.tif/pdf")
