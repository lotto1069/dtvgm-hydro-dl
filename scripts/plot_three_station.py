import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

flow_data_path = r"D:\Desktop\flow_observation_prediction_data.xlsx"
summary_data_path = r"D:\Desktop\model_comparison_summary.xlsx"
output_image_path = r"D:\Desktop\Runoff_Flow_Comparison_Model_Focus.png"

try:
    if not os.path.exists(flow_data_path):
    if not os.path.exists(summary_data_path):

    flow_df = pd.read_excel(flow_data_path, parse_dates=['Date'], index_col='Date')
    summary_df = pd.read_excel(summary_data_path, index_col=0)

except FileNotFoundError as e:
    print(e)
    exit()
except Exception as e:
    exit()

print(summary_df.round(4))
print("-" * 30)
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Garamond', 'SimHei'],
    'font.size': 13,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'axes.linewidth': 1.1,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(
    flow_df.index,
    flow_df['Observed_Flow'],
    label='Observed',
    linestyle='-',
)
ax.plot(
    flow_df.index,
    flow_df['LSTM_Predicted'],
    label='LSTM',
    color=color_lstm,
    linestyle='-',
    alpha=0.9,
)

ax.plot(
    flow_df.index,
    flow_df['Transformer_Predicted'],
    label='Transformer',
    color=color_transformer,
    linewidth=2.0,
    linestyle='-.',
    alpha=0.9,
    zorder=7
)

ax.plot(
    flow_df.index,
    flow_df['TCN_Predicted'],
    label='TCN',
    color=color_tcn,
    linewidth=2.0,
    linestyle='--',
    alpha=0.9,
    zorder=9
)

ax.set_title(
    'Model-Predicted vs Observed Runoff at Langrengu Station',
    fontsize=18,
    fontweight='bold',
    pad=25,
    loc='center'
)

ax.set_xlabel('Date', fontsize=15, labelpad=12)
ax.set_ylabel(r'Flow Rate ($m^3/s$)', fontsize=15, labelpad=12)

ax.set_ylim(bottom=0, top=flow_df[['Observed_Flow', 'LSTM_Predicted', 'Transformer_Predicted', 'TCN_Predicted']].max().max() * 1.1)
ax.set_xlim(flow_df.index.min(), flow_df.index.max())

handles, labels = ax.get_legend_handles_labels()
sorted_handles = [handles[1], handles[2], handles[3], handles[0]]
sorted_labels = [labels[1], labels[2], labels[3], labels[0]]
ax.legend(
    sorted_handles, sorted_labels,
    loc='upper right',
    fontsize=12,
    title='Model',
    title_fontsize=13,
    frameon=True,
    framealpha=0.85,
    edgecolor='#CCCCCC',
    facecolor='#FFFFFF',
    labelspacing=0.9,
    handlelength=2.0,
    borderpad=0.8,
    fancybox=True
)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
fig.autofmt_xdate(rotation=30, ha='right')

ax.grid(
    True,
    which='major',
    linestyle='--',
    linewidth=0.7,
    color='#E0E0E0',
    alpha=0.7
)
ax.grid(
    True,
    which='minor',
    linestyle=':',
    linewidth=0.4,
    color='#F0F0F0',
    alpha=0.5
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#666666')
ax.spines['bottom'].set_color('#666666')

ax.tick_params(
    axis='both',
    which='major',
    length=6,
    pad=8
)
ax.tick_params(
    axis='both',
    which='minor',
    length=3
)

plt.tight_layout()
plt.savefig(
    output_image_path,
    dpi=350,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

plt.show()