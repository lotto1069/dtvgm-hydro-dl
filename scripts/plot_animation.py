import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

plt.style.use('seaborn-v0_8-whitegrid')

try:
    font_prop = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=10)
except:
    font_prop = None

files = {
}

dfs = {k: pd.read_csv(v) for k, v in files.items()}
for df in dfs.values():
    df['Date'] = pd.to_datetime(df['Date'])

width_cm = 6.6

width_inch = width_cm / 2.54

fig, axes = plt.subplots(3, 1, figsize=(width_inch, height_inch), dpi=300)

colors = {'obs': '#1A365D', 'pred': '#FF4500'}

lines_obs = []
lines_pred = []

for idx, (ax, (title, df)) in enumerate(zip(axes, dfs.items())):
    lines_obs.append(lo)
    lines_pred.append(lp)

    title_font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12, weight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=2, fontproperties=title_font)

    max_y = max(df['Observed'].max(), df['Predicted'].max())
    
    ax.set_xlim(df['Date'].min(), df['Date'].max())

    ax.tick_params(axis='both', labelsize=10)
    
        ax.tick_params(axis='x', labelbottom=False)
    
    ax.tick_params(axis='x', pad=2)

plt.tight_layout(h_pad=0.3, pad=0.5)

fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.05, hspace=0.2)

def update(frame):
    current_idx = int(frame * n_points / total_frames)
    if current_idx >= n_points:
        current_idx = n_points - 1

    for title, lo, lp in zip(dfs.keys(), lines_obs, lines_pred):
        df = dfs[title]
        lo.set_data(df['Date'][:current_idx + 1], df['Observed'][:current_idx + 1])
        lp.set_data(df['Date'][:current_idx + 1], df['Predicted'][:current_idx + 1])

    return lines_obs + lines_pred
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=60, blit=True)

ani.save(output_path, writer='pillow', fps=25)

plt.close()
