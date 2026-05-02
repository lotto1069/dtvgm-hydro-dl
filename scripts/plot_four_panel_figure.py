import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.gridspec import GridSpec

FIG_WIDTH = 16
FIG_HEIGHT = 16

plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']

LABEL_SIZE = 16
TITLE_SIZE = 20
TICK_SIZE = 14
TAG_SIZE = 26
LEGEND_SIZE = 14
TEXT_IN_PLOT = 12

BORDER_WIDTH = 2.5
GRID_WIDTH = 1.0

model_colors = sns.color_palette("Set1", 4)
observed_color = "darkblue"

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

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

gs = GridSpec(2, 2, figure=fig,
              left=0.08, right=0.93, top=0.93, bottom=0.1,
              wspace=0.25, hspace=0.45,
              height_ratios=[1.0, 1.3])

ax_a = fig.add_subplot(gs[0, 0], polar=True)

metrics_norm = metrics.copy()
scaler = MinMaxScaler()
for col in ["NSE_Test", "R2_Test"]:
    metrics_norm[col] = scaler.fit_transform(metrics[[col]])
metrics_norm["RMSE_Test"] = 1 - scaler.fit_transform(metrics[["RMSE_Test"]])

categories = ["$R^2$", "NSE", "RMSE (Inv.)"]
N = len(categories)
angles = [n / N * 2 * np.pi for n in range(N)] + [0]

line_styles = {
    "DTVGM": {'color': model_colors[0], 'linestyle': '--', 'linewidth': 3},
    "DTVGM+LSTM": {'color': model_colors[1], 'linestyle': '-', 'linewidth': 4},
    "DTVGM+TCN": {'color': model_colors[2], 'linestyle': '-', 'linewidth': 3},
    "DTVGM+Transformer": {'color': model_colors[3], 'linestyle': '-', 'linewidth': 3}
}

for i, row in metrics_norm.iterrows():
    values = row[["R2_Test", "NSE_Test", "RMSE_Test"]].tolist()
    values_closed = values + [values[0]]
    ax_a.plot(angles, values_closed, label=row["Model"], marker='o', markersize=9, **line_styles[row["Model"]])

ax_a.set_xticks(angles[:-1])
ax_a.set_xticklabels(categories, fontsize=TITLE_SIZE, fontweight='bold')
ax_a.tick_params(axis='x', pad=20)

ax_a.set_yticks([0.5, 1.0])
ax_a.set_yticklabels(["0.5", "1.0"], color="black", fontsize=TICK_SIZE, fontweight='bold')
ax_a.set_ylim(0, 1.1)
ax_a.set_rlabel_position(180)

legend_a = ax_a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2,
                       fontsize=LEGEND_SIZE, frameon=True, edgecolor='black')
legend_a.get_frame().set_linewidth(BORDER_WIDTH)

ax_a.grid(True, color='black', linewidth=GRID_WIDTH, linestyle='-', alpha=1.0)
ax_a.spines['polar'].set_visible(True)
ax_a.spines['polar'].set_color('black')
ax_a.spines['polar'].set_linewidth(BORDER_WIDTH)

ax_b = fig.add_subplot(gs[0, 1])
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
metric_df["Model_Short"] = ["DTVGM", "DT+LSTM", "DT+TCN", "DT+Trans"]

x = np.arange(len(metric_df))
width = 0.35

bars1 = ax_b.bar(x - width / 2, metric_df["R2"], width, label="$R^2$", color=model_colors[0], edgecolor='k',
                 linewidth=1, alpha=0.9)
bars2 = ax_b2.bar(x + width / 2, metric_df["RMSE"], width, label="RMSE", color=model_colors[1], edgecolor='k',
                  linewidth=1, alpha=0.9)

for bar in bars1:
    ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.3f}",
              ha='center', fontsize=TEXT_IN_PLOT, fontweight='bold')
for bar in bars2:
    ax_b2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{bar.get_height():.3f}",
               ha='center', fontsize=TEXT_IN_PLOT, fontweight='bold')

ax_b.set_ylabel("$R^2$", fontsize=TITLE_SIZE, fontweight='bold', labelpad=10)
ax_b2.set_ylabel("RMSE ($m^3$/s)", fontsize=TITLE_SIZE, fontweight='bold', labelpad=10)

ax_b.set_xticks(x)
ax_b.set_xticklabels(metric_df["Model_Short"], rotation=0, ha='center', fontsize=LABEL_SIZE, fontweight='bold')
ax_b.tick_params(axis='y', labelsize=TICK_SIZE, color='black', width=BORDER_WIDTH)
ax_b2.tick_params(axis='y', labelsize=TICK_SIZE, color='black', width=BORDER_WIDTH)
ax_b.tick_params(axis='x', color='black', width=BORDER_WIDTH)

ax_b.set_ylim(0, 1.15)
ax_b2.set_ylim(0, 0.35)

lines1, labels1 = ax_b.get_legend_handles_labels()
lines2, labels2 = ax_b2.get_legend_handles_labels()
ax_b.legend(lines1 + lines2, labels1 + labels2,
            loc='upper center', bbox_to_anchor=(0.5, -0.14),
            ncol=2, frameon=False, fontsize=LEGEND_SIZE)

for ax in [ax_b, ax_b2]:
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(BORDER_WIDTH)

gs_c = gs[1, 0].subgridspec(2, 2, hspace=0.35, wspace=0.25)
ax_c_bg = fig.add_subplot(gs[1, 0], frameon=False)
ax_c_bg.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

inner_axes = [fig.add_subplot(gs_c[i, j]) for i in range(2) for j in range(2)]

for idx, (model, pred_full) in enumerate(models.items()):
    res = (pred_full - observed_full).dropna()
    ax = inner_axes[idx]
    title_str = model.replace("DTVGM+", "DT+") if model != "DTVGM" else "DTVGM"

    sns.kdeplot(res, ax=ax, color=model_colors[idx], fill=True, alpha=0.6, linewidth=2)
    ax.axvline(0, color='crimson', linestyle='--', linewidth=1.5)

    ax.set_title(title_str, fontsize=LABEL_SIZE, pad=5, fontweight='bold')
    ax.set_xlim(-2, 2)
    ax.set_ylabel("")
    ax.tick_params(axis='both', labelsize=TICK_SIZE, color='black', width=BORDER_WIDTH)

    if idx >= 2:

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(BORDER_WIDTH)

ax_d = fig.add_subplot(gs[1, 1], polar=True)

n_groups = len(groups)
theta_width = 2 * np.pi / n_groups
group_thetas = np.linspace(0, 2 * np.pi, n_groups + 1)[:-1]
global_max = 0

for i, (name, data_full) in enumerate(groups.items()):
    data = data_full.dropna()
    if len(data) == 0: continue

    global_max = max(global_max, data.max())
    theta = group_thetas[i]

    scatter_thetas = np.random.uniform(theta - theta_width / 2 + 0.1, theta + theta_width / 2 - 0.1, size=len(data))
    ax_d.scatter(scatter_thetas, data, color=color, alpha=0.6, s=60,
                 edgecolor='white', linewidth=0.8, zorder=2)

    q1, med, q3 = data.quantile([0.25, 0.5, 0.75])
    min_v, max_v = data.min(), data.max()

    ax_d.plot([theta] * 2, [min_v, max_v], color='black', linewidth=1.5, alpha=0.8, zorder=3)
    ax_d.plot([theta - 0.08, theta + 0.08], [min_v, min_v], color='black', linewidth=1.5, zorder=3)
    ax_d.plot([theta - 0.08, theta + 0.08], [max_v, max_v], color='black', linewidth=1.5, zorder=3)
    ax_d.bar(theta, q3 - q1, bottom=q1, width=0.15, color='none', edgecolor='black', linewidth=1.8, zorder=3)
    ax_d.plot([theta - 0.08, theta + 0.08], [med, med], color='red', linewidth=2.5, zorder=4)

    ax_d.text(theta, global_max * 1.02, name_display,
              ha='center', va='center', fontsize=15, fontweight='bold',
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=3), clip_on=False)

ax_d.set_theta_zero_location("N")
ax_d.set_thetagrids([])
ax_d.set_rorigin(-global_max * 0.05)
ax_d.spines['polar'].set_visible(False)
ax_d.grid(True, axis='y', color='black', linewidth=GRID_WIDTH, alpha=1.0, linestyle='-')
ax_d.grid(False, axis='x')
ax_d.set_rmax(global_max * 1.02)
ax_a.text(-0.15, 1.05, '(a)', transform=ax_a.transAxes, fontsize=TAG_SIZE, fontweight='bold', va='bottom', ha='right')
ax_b.text(-0.10, 1.05, '(b)', transform=ax_b.transAxes, fontsize=TAG_SIZE, fontweight='bold', va='bottom', ha='right')
ax_c_bg.text(-0.10, 1.02, '(c)', transform=ax_c_bg.transAxes, fontsize=TAG_SIZE, fontweight='bold', va='bottom', ha='right')
ax_d.text(-0.10, 1.05, '(d)', transform=ax_d.transAxes, fontsize=TAG_SIZE, fontweight='bold', va='bottom', ha='right')

plt.savefig("Final_Perfectly_Aligned.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()