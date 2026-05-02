import rasterio
import glob
import os

def extract_mean_from_tifs(tif_folder):
    tif_files = sorted(glob.glob(os.path.join(tif_folder, "*.tif")))
    mean_values = []
    for tif in tif_files:
        with rasterio.open(tif) as src:
            data = src.read(1)
            valid_data = data[data > -9999]
            mean_values.append(valid_data.mean())
    return np.array(mean_values)

prec_train_path = r"G:\train\meteorology_1984_2013\xiinghe_001deg_prec_flattened_tif"
prec_test_path = r"G:\test\xiinghe_001deg_prec_flattened_tif"

prec_train = extract_mean_from_tifs(prec_train_path)
prec_test = extract_mean_from_tifs(prec_test_path)

u_stat, p_value_mw = stats.mannwhitneyu(prec_train, prec_test)
ks_stat, p_value_ks = stats.ks_2samp(prec_train, prec_test)
plt.figure(figsize=(10, 5), dpi=300)

plt.subplot(1, 2, 1)
sns.ecdfplot(prec_train, label='Training (1984-2013)', color='#1f77b4')
sns.ecdfplot(prec_test, label='Testing (2014-2022)', color='#d62728', linestyle='--')
plt.title('CDF of Basin-Averaged Precipitation')
plt.xlabel('Precipitation (mm/day)')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# runoff CDF
plt.subplot(1, 2, 2)
sns.ecdfplot(df_train['Flow_Obs'], label='Training (1984-2013)', color='#1f77b4')
sns.ecdfplot(df_test['Flow_Obs'], label='Testing (2014-2022)', color='#d62728', linestyle='--')
plt.title('CDF of Observed Runoff')
plt.xlabel('Runoff ($m^3/s$)')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('Data_Consistency_Analysis.png')
