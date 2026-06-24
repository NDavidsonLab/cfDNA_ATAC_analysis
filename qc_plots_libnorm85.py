#!/usr/bin/env python3
"""
qc_plots_libnorm85.py

Quality-control plots for the 85% quantile per-sample library normalization.

Four panels:
  1. Per-sample value distribution boxplot (pre vs post normalization)
     - 20 evenly-sampled samples shown for legibility
     - Top panel: pre-norm. Bottom panel: post-norm.
  2. q85 scaling factor histogram, colored by source (TCGA / Blood / AML)
  3. Non-zero peak count distribution per sample, colored by source
  4. Mean expression per peak: pre vs post normalization (10k random peaks)

Reads:
  --raw-matrix              the pre-normalization matrix (super_matrix_persample)
  --norm-matrix             the post-normalization matrix (super_matrix_libnorm85)
  --scaling-factors-tsv     output from libnorm85_persample.py
  --output-dir              where to write the figures

Smart subsampling for speed:
  - Loads only ~20 sample columns and ~50k peaks at random for boxplots
  - Full file used only for the scaling-factor histogram
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def classify_source(sample_id: str) -> str:
    """TCGA samples end with .insertions; Blood with _short_peak; AML with _pk."""
    if sample_id.endswith(".insertions"):
        return "TCGA"
    if sample_id.endswith("_short_peak"):
        return "Blood"
    if sample_id.endswith("_pk"):
        return "AML"
    return "other"


def load_matrix_subset(path: Path, n_samples: int = 20,
                        n_peaks: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Load a fraction of a (peaks x samples) matrix."""
    rng = np.random.default_rng(seed)
    header = pd.read_csv(path, sep="\t", nrows=0)
    all_samples = [c for c in header.columns if c != header.columns[0]]
    # Pick 20 samples spread evenly across the column range to capture
    # cross-cohort variation (TCGA at the end, AML at the start, Blood in middle)
    step = max(1, len(all_samples) // n_samples)
    pick_idx = list(range(0, len(all_samples), step))[:n_samples]
    pick_samples = [all_samples[i] for i in pick_idx]
    cols_to_read = [header.columns[0]] + pick_samples
    logging.info(f"  reading {len(pick_samples)} sample columns: {pick_samples[:3]}...")
    # Read all peaks first (because skiprows is awkward with random sample)
    df = pd.read_csv(path, sep="\t", usecols=cols_to_read,
                     index_col=0, dtype={c: np.float32 for c in pick_samples})
    # Subsample peaks
    if n_peaks < len(df):
        peak_idx = rng.choice(len(df), size=n_peaks, replace=False)
        df = df.iloc[peak_idx]
    return df


def plot_boxplot_prevpost(raw_df, norm_df, out_path):
    """Per-sample boxplot, top=raw, bottom=normalized."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    samples = list(raw_df.columns)
    raw_data = [raw_df[s].values for s in samples]
    norm_data = [norm_df[s].values for s in samples]
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(samples) * 0.5), 10))
    bp1 = axes[0].boxplot(raw_data, labels=samples, showfliers=True,
                           patch_artist=True, flierprops={"markersize": 1, "alpha": 0.3})
    for patch in bp1["boxes"]:
        patch.set_facecolor("#a8c8e0")
    axes[0].set_title("Pre-normalization: per-sample value distribution",
                       fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Value")
    axes[0].axhline(0, color="grey", linestyle=":", alpha=0.5)
    for label in axes[0].get_xticklabels():
        label.set_rotation(45); label.set_ha("right"); label.set_fontsize(7)

    bp2 = axes[1].boxplot(norm_data, labels=samples, showfliers=True,
                           patch_artist=True, flierprops={"markersize": 1, "alpha": 0.3})
    for patch in bp2["boxes"]:
        patch.set_facecolor("#a0d8a0")
    axes[1].set_title("Post-normalization (libnorm85): per-sample value distribution",
                       fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Value (units of q85)")
    axes[1].axhline(1, color="red", linestyle=":", alpha=0.5,
                     label="q85 target = 1.0")
    axes[1].legend(loc="upper right", fontsize=8)
    for label in axes[1].get_xticklabels():
        label.set_rotation(45); label.set_ha("right"); label.set_fontsize(7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info(f"  wrote {out_path.name}")


def plot_q85_histogram(sf_df, out_path):
    """Histogram of q85 scaling factors, colored by source."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sf_df = sf_df.copy()
    sf_df["source"] = sf_df["sample_id"].apply(classify_source)
    colors = {"TCGA": "#4c72b0", "Blood": "#dd8452", "AML": "#55a868", "other": "#888"}
    fig, ax = plt.subplots(figsize=(10, 6))
    sources = sorted(sf_df["source"].unique())
    for src in sources:
        vals = sf_df[sf_df["source"] == src]["q_value"]
        ax.hist(vals, bins=50, alpha=0.55, color=colors.get(src, "#888"),
                label=f"{src} (n={len(vals)})", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("q85 scaling factor (used to divide each sample column)")
    ax.set_ylabel("Number of samples")
    ax.set_title("Distribution of per-sample q85 scaling factors",
                  fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    stats = (f"min={sf_df['q_value'].min():.2f}, "
             f"median={sf_df['q_value'].median():.2f}, "
             f"mean={sf_df['q_value'].mean():.2f}, "
             f"max={sf_df['q_value'].max():.2f}")
    ax.text(0.98, 0.02, stats, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info(f"  wrote {out_path.name}")


def plot_nonzero_count(sf_df, out_path):
    """Non-zero peak count per sample, colored by source."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sf_df = sf_df.copy()
    sf_df["source"] = sf_df["sample_id"].apply(classify_source)
    colors = {"TCGA": "#4c72b0", "Blood": "#dd8452", "AML": "#55a868", "other": "#888"}
    fig, ax = plt.subplots(figsize=(10, 6))
    sources = sorted(sf_df["source"].unique())
    for src in sources:
        vals = sf_df[sf_df["source"] == src]["nonzero_peak_count"]
        ax.hist(vals, bins=50, alpha=0.55, color=colors.get(src, "#888"),
                label=f"{src} (n={len(vals)})", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of non-zero peaks per sample")
    ax.set_ylabel("Number of samples")
    ax.set_title("Distribution of non-zero peak counts (sparsity per sample)",
                  fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info(f"  wrote {out_path.name}")


def plot_mean_per_peak(raw_df, norm_df, out_path):
    """Per-peak mean: pre vs post normalization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    raw_mean = raw_df.mean(axis=1).values
    norm_mean = norm_df.mean(axis=1).values
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(raw_mean, norm_mean, s=1, alpha=0.2, c="#4c72b0")
    # Identity line (would be y=x if normalization didn't change peak structure)
    lim_max = max(raw_mean.max(), norm_mean.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color="red", linestyle=":",
             alpha=0.6, label="y = x")
    ax.set_xlabel("Mean value pre-normalization (across 20 samples)")
    ax.set_ylabel("Mean value post-normalization (units of q85)")
    ax.set_title(f"Per-peak mean: pre vs post normalization "
                 f"({len(raw_mean)} peaks)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    # Spearman correlation: peak structure should be highly preserved
    from scipy.stats import spearmanr
    rho, _ = spearmanr(raw_mean, norm_mean)
    ax.text(0.98, 0.02, f"Spearman ρ = {rho:.4f}\n(should be ≈1 — peak order preserved)",
             transform=ax.transAxes, ha="right", va="bottom",
             fontsize=10, family="monospace",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info(f"  wrote {out_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-matrix",            required=True, type=Path)
    p.add_argument("--norm-matrix",           required=True, type=Path)
    p.add_argument("--scaling-factors-tsv",   required=True, type=Path)
    p.add_argument("--output-dir",            required=True, type=Path)
    p.add_argument("--n-samples-boxplot",     type=int, default=20)
    p.add_argument("--n-peaks-subsample",     type=int, default=50000)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "qc.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info("Loading scaling factors")
    sf = pd.read_csv(args.scaling_factors_tsv, sep="\t")
    logging.info(f"  {len(sf)} samples")

    # Plot 2: q85 histogram (fast, no big matrix needed)
    logging.info("Plot 2: q85 histogram")
    plot_q85_histogram(sf, args.output_dir / "qc2_q85_scaling_factors.png")

    # Plot 3: non-zero count histogram
    logging.info("Plot 3: non-zero peak count histogram")
    plot_nonzero_count(sf, args.output_dir / "qc3_nonzero_peak_count.png")

    # Load subsetted matrices for Plots 1 and 4
    logging.info(f"Loading raw matrix subset (~{args.n_peaks_subsample} peaks, "
                 f"{args.n_samples_boxplot} samples)")
    raw_df = load_matrix_subset(args.raw_matrix,
                                  n_samples=args.n_samples_boxplot,
                                  n_peaks=args.n_peaks_subsample)
    logging.info(f"  loaded raw: {raw_df.shape}")

    logging.info(f"Loading normalized matrix subset")
    norm_df = load_matrix_subset(args.norm_matrix,
                                   n_samples=args.n_samples_boxplot,
                                   n_peaks=args.n_peaks_subsample)
    logging.info(f"  loaded norm: {norm_df.shape}")

    # Ensure same samples + peaks
    common_samples = [c for c in raw_df.columns if c in norm_df.columns]
    raw_df = raw_df[common_samples]
    norm_df = norm_df[common_samples]
    common_peaks = raw_df.index.intersection(norm_df.index)
    raw_df = raw_df.loc[common_peaks]
    norm_df = norm_df.loc[common_peaks]

    # Plot 1
    logging.info("Plot 1: pre/post boxplot")
    plot_boxplot_prevpost(raw_df, norm_df,
                            args.output_dir / "qc1_boxplot_pre_post.png")

    # Plot 4
    logging.info("Plot 4: per-peak mean scatter")
    plot_mean_per_peak(raw_df, norm_df,
                        args.output_dir / "qc4_per_peak_mean_scatter.png")

    logging.info("Done. 4 QC plots written.")


if __name__ == "__main__":
    main()
