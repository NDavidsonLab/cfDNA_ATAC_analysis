#!/usr/bin/env python3
"""
qc_plots_cfdna_libnorm85.py

QC plots for cfDNA per-sample 85% |SE| library normalization.

Four PNGs:
  qc1_boxplot_pre_post.png       per-sample SE distribution boxplot, pre vs post
  qc2_q85_scaling_factors.png    histogram of q85|SE|, colored by cancer type
  qc3_nonzero_peak_count.png     non-zero |SE| peak counts per sample, by cancer type
  qc4_per_peak_mean_scatter.png  per-peak mean SE pre vs post (Spearman ρ)

Reads:
  --raw-matrix              pre-normalization (cfdna_se_per_peak)
  --norm-matrix             post-normalization (cfdna_se_per_peak_libnorm85)
  --scaling-factors-tsv     output from libnorm85_cfdna.py
  --metadata-tsv            sample_id + cancer_type
  --output-dir
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def load_matrix_subset(path: Path, n_samples: int = 25,
                        n_peaks: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Load a subsample (peaks x samples) for fast plotting."""
    rng = np.random.default_rng(seed)
    header = pd.read_csv(path, sep="\t", nrows=0)
    all_samples = [c for c in header.columns if c != header.columns[0]]
    step = max(1, len(all_samples) // n_samples)
    pick_idx = list(range(0, len(all_samples), step))[:n_samples]
    pick_samples = [all_samples[i] for i in pick_idx]
    cols_to_read = [header.columns[0]] + pick_samples
    df = pd.read_csv(path, sep="\t", usecols=cols_to_read,
                     index_col=0, dtype={c: np.float32 for c in pick_samples})
    if n_peaks < len(df):
        peak_idx = rng.choice(len(df), size=n_peaks, replace=False)
        df = df.iloc[peak_idx]
    return df


def plot_boxplot_prevpost(raw_df, norm_df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    samples = list(raw_df.columns)
    raw_data = [raw_df[s].values for s in samples]
    norm_data = [norm_df[s].values for s in samples]
    fig, axes = plt.subplots(2, 1, figsize=(max(11, len(samples) * 0.5), 11))
    bp1 = axes[0].boxplot(raw_data, labels=samples, showfliers=True,
                           patch_artist=True,
                           flierprops={"markersize": 1, "alpha": 0.3})
    for patch in bp1["boxes"]:
        patch.set_facecolor("#a8c8e0")
    axes[0].set_title("Pre-normalization: per-sample SE distribution",
                       fontsize=11, fontweight="bold")
    axes[0].set_ylabel("SE")
    axes[0].axhline(0, color="grey", linestyle=":", alpha=0.5)
    for label in axes[0].get_xticklabels():
        label.set_rotation(45); label.set_ha("right"); label.set_fontsize(7)

    bp2 = axes[1].boxplot(norm_data, labels=samples, showfliers=True,
                           patch_artist=True,
                           flierprops={"markersize": 1, "alpha": 0.3})
    for patch in bp2["boxes"]:
        patch.set_facecolor("#a0d8a0")
    axes[1].set_title("Post-normalization (libnorm85): per-sample SE distribution",
                       fontsize=11, fontweight="bold")
    axes[1].set_ylabel("SE (units of q85|SE|)")
    axes[1].axhline(1, color="red", linestyle=":", alpha=0.5,
                     label="q85|SE| target = 1.0")
    axes[1].axhline(-1, color="red", linestyle=":", alpha=0.5)
    axes[1].axhline(0, color="grey", linestyle=":", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=8)
    for label in axes[1].get_xticklabels():
        label.set_rotation(45); label.set_ha("right"); label.set_fontsize(7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_q85_histogram(sf_df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    unique_cts = sorted(sf_df["cancer_type"].unique())
    cmap = plt.cm.get_cmap("tab20")
    ct_colors = {ct: cmap(i % 20) for i, ct in enumerate(unique_cts)}
    for ct in unique_cts:
        vals = sf_df[sf_df["cancer_type"] == ct]["q_value"]
        ax.hist(vals, bins=30, alpha=0.6, color=ct_colors[ct],
                label=f"{ct} (n={len(vals)})",
                edgecolor="white", linewidth=0.5)
    ax.set_xlabel("q85 |SE| scaling factor")
    ax.set_ylabel("Number of samples")
    ax.set_title("Distribution of per-sample q85 |SE| scaling factors",
                  fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    stats = (f"min={sf_df['q_value'].min():.3f}, "
             f"median={sf_df['q_value'].median():.3f}, "
             f"mean={sf_df['q_value'].mean():.3f}, "
             f"max={sf_df['q_value'].max():.3f}")
    ax.text(0.98, 0.02, stats, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_nonzero_count(sf_df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    unique_cts = sorted(sf_df["cancer_type"].unique())
    cmap = plt.cm.get_cmap("tab20")
    ct_colors = {ct: cmap(i % 20) for i, ct in enumerate(unique_cts)}
    for ct in unique_cts:
        vals = sf_df[sf_df["cancer_type"] == ct]["nonzero_peak_count"]
        ax.hist(vals, bins=30, alpha=0.6, color=ct_colors[ct],
                label=f"{ct} (n={len(vals)})",
                edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Non-zero |SE| peak count per sample")
    ax.set_ylabel("Number of samples")
    ax.set_title("Distribution of non-zero |SE| peak counts (cfDNA coverage)",
                  fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_mean_per_peak(raw_df, norm_df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    raw_mean = raw_df.mean(axis=1).values
    norm_mean = norm_df.mean(axis=1).values
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(raw_mean, norm_mean, s=1, alpha=0.2, c="#4c72b0")
    lim_min = min(raw_mean.min(), norm_mean.min()) * 1.05
    lim_max = max(raw_mean.max(), norm_mean.max()) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
             color="red", linestyle=":", alpha=0.6, label="y = x")
    ax.set_xlabel("Mean SE pre-normalization")
    ax.set_ylabel("Mean SE post-normalization (units of q85|SE|)")
    ax.set_title(f"Per-peak mean SE: pre vs post normalization "
                 f"({len(raw_mean)} peaks)",
                 fontsize=11, fontweight="bold")
    ax.axhline(0, color="grey", linestyle=":", alpha=0.4)
    ax.axvline(0, color="grey", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(raw_mean, norm_mean)
    ax.text(0.98, 0.02,
             f"Spearman ρ = {rho:.4f}\n(should be ≈1 — peak order preserved)",
             transform=ax.transAxes, ha="right", va="bottom",
             fontsize=10, family="monospace",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-matrix",            required=True, type=Path)
    p.add_argument("--norm-matrix",           required=True, type=Path)
    p.add_argument("--scaling-factors-tsv",   required=True, type=Path)
    p.add_argument("--metadata-tsv",          required=True, type=Path)
    p.add_argument("--output-dir",            required=True, type=Path)
    p.add_argument("--n-samples-boxplot",     type=int, default=25)
    p.add_argument("--n-peaks-subsample",     type=int, default=50000)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "qc_cfdna.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info("Loading scaling factors + metadata")
    sf = pd.read_csv(args.scaling_factors_tsv, sep="\t")
    meta = pd.read_csv(args.metadata_tsv, sep="\t")
    sf = sf.merge(meta, on="sample_id", how="left")
    sf["cancer_type"] = sf["cancer_type"].fillna("unknown")
    logging.info(f"  {len(sf)} samples with metadata merged")

    logging.info("Plot 2: q85 |SE| histogram")
    plot_q85_histogram(sf, args.output_dir / "qc2_q85_scaling_factors.png")
    logging.info("Plot 3: non-zero peak count histogram")
    plot_nonzero_count(sf, args.output_dir / "qc3_nonzero_peak_count.png")

    logging.info(f"Loading raw matrix subset")
    raw_df = load_matrix_subset(args.raw_matrix,
                                  n_samples=args.n_samples_boxplot,
                                  n_peaks=args.n_peaks_subsample)
    logging.info(f"  raw: {raw_df.shape}")
    logging.info(f"Loading normalized matrix subset")
    norm_df = load_matrix_subset(args.norm_matrix,
                                   n_samples=args.n_samples_boxplot,
                                   n_peaks=args.n_peaks_subsample)
    logging.info(f"  norm: {norm_df.shape}")

    common_samples = [c for c in raw_df.columns if c in norm_df.columns]
    raw_df = raw_df[common_samples]
    norm_df = norm_df[common_samples]
    common_peaks = raw_df.index.intersection(norm_df.index)
    raw_df = raw_df.loc[common_peaks]
    norm_df = norm_df.loc[common_peaks]

    logging.info("Plot 1: pre/post boxplot")
    plot_boxplot_prevpost(raw_df, norm_df,
                            args.output_dir / "qc1_boxplot_pre_post.png")
    logging.info("Plot 4: per-peak mean scatter")
    plot_mean_per_peak(raw_df, norm_df,
                        args.output_dir / "qc4_per_peak_mean_scatter.png")

    logging.info("Done.")


if __name__ == "__main__":
    main()
