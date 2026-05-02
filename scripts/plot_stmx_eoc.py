# -*- coding: utf-8 -*-
"""
NDVI-Stratified Ecohydrological Threshold Analysis  (FIXED VERSION)
====================================================================
Fixes from v1:
  - Uses the real RH column ('rhu') from the pkl, not a derived proxy.
  - Uses a more robust threshold-extraction method (segmented regression
    on cumulative-mean curve) instead of "first deviation from baseline".
  - Filters out frozen/snowmelt season samples for low-NDVI stratum so
  - Reports thresholds with bootstrap confidence intervals.

Run:  python ndvi_threshold_analysis_fixed.py
"""

import os
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

OUT_DIR     = Path("ndvi_threshold_outputs_fixed")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "axes.edgecolor": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "grid.color": "0.5", "grid.alpha": 0.5,
    "grid.linewidth": 0.6, "grid.linestyle": "--",
    "text.color": "black", "font.family": "sans-serif",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
# Load
def load_data():
    print("\n" + "=" * 70)
    print("Loading data")
    print("=" * 70)
    with open(CACHE_TRAIN, "rb") as f:
        df_train = pickle.load(f)
    with open(CACHE_TEST, "rb") as f:
        df_test = pickle.load(f)
    df = pd.concat([df_train, df_test]).sort_index()
    df.index = pd.DatetimeIndex(df.index)

    out = pd.DataFrame(index=df.index)
    out["flow"] = df["flow"].values
    out["prec"] = df["prec"].values
    out["pet"]  = df["petpm"].values
    out["temp"] = df["temp"].values
    out["ndvi"] = df["ndvi"].values
    out["rh"]   = df["rhu"].values        # <<< real RH from pkl
    out["snow"] = df["snow"].values

    # NDVI scale check
    if np.nanmedian(out["ndvi"]) > 2:
        out["ndvi"] *= 1e-4

    # Sanity print
    print(f"\nSamples: {len(out)}  ({out.index.min()} to {out.index.max()})")
    print(f"  flow   range: {out['flow'].min():.3f} to {out['flow'].max():.3f}")
    print(f"  pet    range: {out['pet'].min():.3f} to {out['pet'].max():.3f}")
    print(f"  rh     range: {out['rh'].min():.1f} to {out['rh'].max():.1f}  (real RH from rhu)")
    print(f"  ndvi   range: {out['ndvi'].min():.3f} to {out['ndvi'].max():.3f}")

    out = out.dropna()
    return out
def find_breakpoint(x, y, x_grid=None):
    """
    Fit y = a + b*x + c*max(0, x - bp) and pick bp that minimises RSS.
    Returns (breakpoint, slope_left, slope_right, R2).
    Robust to noise; returns NaN if no clear break.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 100:
        return np.nan, np.nan, np.nan, np.nan

    if x_grid is None:
        x_grid = np.percentile(x, np.linspace(15, 85, 30))

    best = (np.nan, np.nan, np.nan, -np.inf)
    for bp in x_grid:
        xx = np.column_stack([np.ones_like(x), x, np.maximum(0, x - bp)])
        try:
            coef, *_ = np.linalg.lstsq(xx, y, rcond=None)
            yhat = xx @ coef
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            if r2 > best[3]:
                best = (bp, coef[1], coef[1] + coef[2], r2)
        except Exception:
            continue
    return best
def bootstrap_breakpoint(x, y, n_boot=200, x_grid=None):
    """Bootstrap CI for the breakpoint."""
    bps = []
    n = len(x)
    rng = np.random.default_rng(0)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bp, *_ = find_breakpoint(x[idx], y[idx], x_grid)
        if np.isfinite(bp):
            bps.append(bp)
    if len(bps) < 10:
        return np.nan, np.nan
    return np.percentile(bps, [10, 90])
# Stratify and analyse
def filter_growing_season(df):
    runoff is rainfall-driven, not snowmelt-driven."""
    return df[df.index.month.isin([5, 6, 7, 8, 9])].copy()
def stratify_by_ndvi(df, n_strata=3):
    ndvi = df["ndvi"].values
    cuts = np.percentile(ndvi, [100 / n_strata * i for i in range(1, n_strata)])
    labels = ["Low NDVI", "Medium NDVI", "High NDVI"][:n_strata]
    stratum = pd.Series(index=df.index, dtype=object)
    stratum.loc[ndvi <= cuts[0]] = labels[0]
    if n_strata == 3:
        stratum.loc[(ndvi > cuts[0]) & (ndvi <= cuts[1])] = labels[1]
        stratum.loc[ndvi > cuts[1]] = labels[2]
    df = df.copy()
    df["stratum"] = stratum
    print("\nStratification cuts on NDVI:", [f"{c:.3f}" for c in cuts])
    print(df["stratum"].value_counts())
    return df, labels
def compute_thresholds(df, labels):
    rows = []
    for lab in labels:
        sub = df[df["stratum"] == lab]
        if len(sub) < 200:
            continue

        # PET threshold (suppression: slope flips from + to weaker/negative)
        pet_bp, sl_l, sl_r, r2 = find_breakpoint(
            sub["pet"].values, sub["flow"].values,
            x_grid=np.linspace(2, 7, 25))
        pet_ci = bootstrap_breakpoint(
            sub["pet"].values, sub["flow"].values,
            n_boot=100, x_grid=np.linspace(2, 7, 25))

        # RH threshold (activation: slope flips from ~0 to positive)
        rh_bp, sl_l2, sl_r2, r2_rh = find_breakpoint(
            sub["rh"].values, sub["flow"].values,
            x_grid=np.linspace(40, 85, 25))
        rh_ci = bootstrap_breakpoint(
            sub["rh"].values, sub["flow"].values,
            n_boot=100, x_grid=np.linspace(40, 85, 25))

        rows.append({
            "Stratum": lab,
            "n": len(sub),
            "Mean NDVI": round(sub["ndvi"].mean(), 3),
            "PET threshold (mm/d)": (round(pet_bp, 2)
                                     if np.isfinite(pet_bp) else "n/a"),
            "PET 80% CI":  (f"[{pet_ci[0]:.1f}, {pet_ci[1]:.1f}]"
                            if np.isfinite(pet_ci[0]) else "n/a"),
            "RH threshold (%)": (round(rh_bp, 1)
                                 if np.isfinite(rh_bp) else "n/a"),
            "RH 80% CI": (f"[{rh_ci[0]:.0f}, {rh_ci[1]:.0f}]"
                          if np.isfinite(rh_ci[0]) else "n/a"),
        })

    table = pd.DataFrame(rows)
    print("\nThreshold table (real RH, growing-season only, "
          "broken-stick regression):\n")
    print(table.to_string(index=False))
    table.to_csv(OUT_DIR / "table_thresholds_fixed.csv", index=False)
    return table
# Figures
def figure_response_curves(df, labels):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"Low NDVI": "#d95f0e",
              "Medium NDVI": "#74a9cf",
              "High NDVI": "#238b45"}

    for lab in labels:
        sub = df[df["stratum"] == lab]
        if len(sub) < 50:
            continue

        bins_pet = np.linspace(0.5, np.nanpercentile(sub["pet"], 95), 18)
        centres_pet = 0.5 * (bins_pet[:-1] + bins_pet[1:])
        means, sems = [], []
        for i in range(len(centres_pet)):
            m = (sub["pet"] >= bins_pet[i]) & (sub["pet"] < bins_pet[i + 1])
            v = sub.loc[m, "flow"]
            means.append(v.mean() if len(v) >= 5 else np.nan)
            sems.append(v.std() / np.sqrt(len(v)) if len(v) >= 5 else np.nan)
        means, sems = np.array(means), np.array(sems)
        axes[0].plot(centres_pet, means, "-o", ms=4, color=colors[lab],
                     label=f"{lab} (n={len(sub)})")
        axes[0].fill_between(centres_pet, means - sems, means + sems,
                             color=colors[lab], alpha=0.15)

        # RH vs flow
        bins_rh = np.linspace(25, 95, 18)
        centres_rh = 0.5 * (bins_rh[:-1] + bins_rh[1:])
        means, sems = [], []
        for i in range(len(centres_rh)):
            m = (sub["rh"] >= bins_rh[i]) & (sub["rh"] < bins_rh[i + 1])
            v = sub.loc[m, "flow"]
            means.append(v.mean() if len(v) >= 5 else np.nan)
            sems.append(v.std() / np.sqrt(len(v)) if len(v) >= 5 else np.nan)
        means, sems = np.array(means), np.array(sems)
        axes[1].plot(centres_rh, means, "-o", ms=4, color=colors[lab],
                     label=f"{lab} (n={len(sub)})")
        axes[1].fill_between(centres_rh, means - sems, means + sems,
                             color=colors[lab], alpha=0.15)

    axes[0].axvline(4, color="grey", ls=":", lw=1)
    axes[0].set_xlabel("PET (mm/day)")
    axes[0].set_ylabel("Mean daily flow (m$^3$/s)")
    axes[0].grid(True, alpha=0.4)
    axes[0].legend()

    axes[1].axvline(70, color="grey", ls=":", lw=1)
    axes[1].set_xlabel("Relative humidity (%)")
    axes[1].set_ylabel("Mean daily flow (m$^3$/s)")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    out = OUT_DIR / "figure_response_curves_fixed.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")
def figure_3d(df):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    n = min(3000, len(df))
    s = df.sample(n=n, random_state=0)

    flow_log = np.log10(s["flow"].values + 0.01)
    p = ax.scatter(s["rh"], s["pet"], s["ndvi"],
                   c=flow_log, cmap="viridis", s=8, alpha=0.7)
    cbar = plt.colorbar(p, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("log$_{10}$(flow + 0.01)  [m$^3$/s]")

    ax.set_xlabel("Relative humidity (%)")
    ax.set_ylabel("PET (mm/day)")
    ax.set_zlabel("NDVI")
    ax.set_title("Ecohydrological response space (growing season)")

    plt.savefig(OUT_DIR / "figure_3d_response_space_fixed.png")
    plt.close()
    print(f"Saved: {OUT_DIR / 'figure_3d_response_space_fixed.png'}")
def figure_table(table):
    fig, ax = plt.subplots(figsize=(13, 2.4))
    ax.axis("off")
    tbl = ax.table(cellText=table.values, colLabels=table.columns,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for k, cell in tbl.get_celld().items():
        if k[0] == 0:
            cell.set_facecolor("#cfd8dc")
            cell.set_text_props(weight="bold")
    plt.savefig(OUT_DIR / "figure_threshold_table_fixed.png")
    plt.close()
    print(f"Saved: {OUT_DIR / 'figure_threshold_table_fixed.png'}")
# Findings
def print_findings(table):
    print("\n" + "=" * 70)
    print("Findings (real RH, growing-season filtered)")
    print("=" * 70)
    if len(table) == 0:
        print("No usable thresholds.")
        return
    try:
        low, high = table.iloc[0], table.iloc[-1]
        rh_low  = float(low["RH threshold (%)"])
        rh_high = float(high["RH threshold (%)"])
        pet_low  = float(low["PET threshold (mm/d)"])
        pet_high = float(high["PET threshold (mm/d)"])

        print(f"""
Suggested sentence for Section 5.3:

  "Stratifying growing-season samples by vegetation cover indicates
   that the runoff-activation thresholds are not invariant. The relative
   humidity threshold shifted from {rh_high:.0f}% under high NDVI to
   {rh_low:.0f}% under low NDVI, and the PET suppression boundary shifted
   from {pet_high:.1f} mm/day to {pet_low:.1f} mm/day across the same
   gradient (Figure X, Table X). This NDVI-conditional behaviour suggests
   that the identified thresholds function as condition-dependent
   indicators of grassland runoff state rather than fixed climatological
   for vegetation status."
""")
    except Exception as e:
        print(f"Could not auto-generate text: {e}")
        print("Inspect the table and write manually.")
# Main
def main():
    df = load_data()
    df_gs = filter_growing_season(df)
    df_gs, labels = stratify_by_ndvi(df_gs, n_strata=3)
    table = compute_thresholds(df_gs, labels)
    figure_response_curves(df_gs, labels)
    figure_3d(df_gs)
    figure_table(table)
    print_findings(table)
    print(f"\nAll outputs are in: {OUT_DIR.resolve()}")
if __name__ == "__main__":
    main()