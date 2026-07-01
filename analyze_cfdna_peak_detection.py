#!/usr/bin/env python3
"""
analyze_cfdna_peak_detection.py

Per-peak detection / coverage analysis on the cfDNA libnorm-total Subnuc
matrix (1.07M peaks x 43 samples).

Six analyses:
  1. Per-peak detection rate distribution
     "How many of the 43 samples have a non-zero signal at each peak?"
  2. Per-sample peak counts
     "How many peaks are detected per sample? Any outlier samples?"
  3. Per-peak mean signal vs detection rate scatter
     "Are universal peaks high-signal, or are low-detection peaks also weak?"
  4. Pairwise sample intersection heatmap
     "For each pair of samples, what fraction of peaks do both detect?"
  5. Cumulative coverage curve
     "If we require detection in >=N samples to keep a peak, how many survive?"
  6. Within-vs-across cancer-type sharing
     "Do BRCA samples share more peaks with each other than with LUAD?"

Inputs:
  --cfdna-matrix     cfdna_subnuc_libnorm_total.tsv.gz (1.07M peaks x 43)
  --cfdna-metadata   cfdna_sample_metadata.tsv (sample_id, cancer_type)
  --output-dir
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfdna-matrix",   required=True, type=Path)
    p.add_argument("--cfdna-metadata", required=True, type=Path)
    p.add_argument("--output-dir",     required=True, type=Path)
    p.add_argument("--threshold",      type=float, default=0.0,
                   help="Detection threshold; default 0 (any non-zero value)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "analysis.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.info(f"Loading cfDNA matrix: {args.cfdna_matrix}")
    header = pd.read_csv(args.cfdna_matrix, sep="\t", nrows=0)
    sample_cols = [c for c in header.columns if c != header.columns[0]]
    dtype_dict = {c: np.float32 for c in sample_cols}
    M = pd.read_csv(args.cfdna_matrix, sep="\t", index_col=0,
                    dtype=dtype_dict)
    n_peaks, n_samples = M.shape
    logging.info(f"  shape: {n_peaks} peaks x {n_samples} samples")
    samples = list(M.columns)

    # Load metadata
    meta = pd.read_csv(args.cfdna_metadata, sep="\t")
    sample_to_cancer = dict(zip(meta["sample_id"], meta["cancer_type"]))
    cancer_of_sample = np.array([sample_to_cancer.get(s, "unknown")
                                   for s in samples])
    unique_cancers = sorted(set(cancer_of_sample))
    n_per_cancer = {ct: (cancer_of_sample == ct).sum() for ct in unique_cancers}
    logging.info(f"  cancer types: {len(unique_cancers)}")

    # Detection mask (boolean)
    logging.info(f"Building detection mask (threshold > {args.threshold})")
    M_arr = M.values
    detected = M_arr > args.threshold  # shape (n_peaks, n_samples)
    detection_rate = detected.sum(axis=1)  # per peak: how many samples
    sample_peak_count = detected.sum(axis=0)  # per sample: how many peaks
    logging.info(f"  total non-zero entries: {detected.sum():,}")
    logging.info(f"  detection rate range: {detection_rate.min()}-{detection_rate.max()}")
    logging.info(f"  sample peak count range: {sample_peak_count.min():,}-{sample_peak_count.max():,}")

    # ============================================================
    # Analysis 1: Per-peak detection rate histogram
    # ============================================================
    logging.info("Analysis 1: per-peak detection rate histogram")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bins = np.arange(0, n_samples + 2) - 0.5
    counts, _ = np.histogram(detection_rate, bins=bins)
    axes[0].bar(np.arange(0, n_samples + 1), counts, width=0.9,
                color="#4c72b0", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel(f"Number of samples with signal at peak (out of {n_samples})")
    axes[0].set_ylabel("Number of peaks")
    axes[0].set_title(f"Per-peak detection rate\n"
                       f"{n_peaks:,} unified peaks",
                       fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Cumulative version (right plot)
    axes[1].bar(np.arange(0, n_samples + 1), counts, width=0.9,
                color="#dd8452", edgecolor="white", linewidth=0.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(f"Number of samples with signal at peak (out of {n_samples})")
    axes[1].set_ylabel("Number of peaks (log scale)")
    axes[1].set_title(f"Same data, log-Y axis",
                       fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(args.output_dir / "01_detection_rate_histogram.png",
                dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Summary stats for detection rate
    pct_zero = 100.0 * (detection_rate == 0).sum() / n_peaks
    pct_universal = 100.0 * (detection_rate == n_samples).sum() / n_peaks
    pct_majority = 100.0 * (detection_rate >= n_samples / 2).sum() / n_peaks
    pct_one_only = 100.0 * (detection_rate == 1).sum() / n_peaks
    logging.info(f"  peaks with 0 detection (universally absent):  "
                 f"{(detection_rate==0).sum():,} ({pct_zero:.1f}%)")
    logging.info(f"  peaks detected in exactly 1 sample:           "
                 f"{(detection_rate==1).sum():,} ({pct_one_only:.1f}%)")
    logging.info(f"  peaks detected in >=50% of samples:           "
                 f"{(detection_rate >= n_samples/2).sum():,} ({pct_majority:.1f}%)")
    logging.info(f"  peaks detected in all {n_samples} samples (universal): "
                 f"{(detection_rate==n_samples).sum():,} ({pct_universal:.1f}%)")

    # ============================================================
    # Analysis 2: Per-sample peak counts (bar chart, colored by cancer)
    # ============================================================
    logging.info("Analysis 2: per-sample peak counts")
    cmap = plt.cm.get_cmap("tab20")
    ct_colors = {ct: cmap(i % 20) for i, ct in enumerate(unique_cancers)}
    bar_colors = [ct_colors[c] for c in cancer_of_sample]

    fig, ax = plt.subplots(figsize=(max(12, n_samples * 0.3), 6))
    ax.bar(range(n_samples), sample_peak_count, color=bar_colors,
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(n_samples))
    ax.set_xticklabels(samples, rotation=90, fontsize=8)
    ax.set_ylabel("Number of detected peaks per sample")
    ax.set_title(f"Per-sample peak detection count (colored by cancer type)",
                  fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Cancer-type legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=ct_colors[ct])
                for ct in unique_cancers]
    labels = [f"{ct} (n={n_per_cancer[ct]})" for ct in unique_cancers]
    ax.legend(handles, labels, loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(args.output_dir / "02_per_sample_peak_counts.png",
                dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logging.info(f"  per-sample peak count summary:")
    logging.info(f"    min:    {sample_peak_count.min():,} "
                 f"({samples[sample_peak_count.argmin()]})")
    logging.info(f"    median: {int(np.median(sample_peak_count)):,}")
    logging.info(f"    mean:   {int(sample_peak_count.mean()):,}")
    logging.info(f"    max:    {sample_peak_count.max():,} "
                 f"({samples[sample_peak_count.argmax()]})")

    # ============================================================
    # Analysis 3: Signal vs detection rate scatter
    # ============================================================
    logging.info("Analysis 3: per-peak mean signal vs detection rate")
    # Mean signal across samples where peak was detected
    with np.errstate(invalid="ignore", divide="ignore"):
        peak_sum = M_arr.sum(axis=1)
        peak_mean_when_detected = np.where(detection_rate > 0,
                                              peak_sum / detection_rate, 0)

    fig, ax = plt.subplots(figsize=(9, 7))
    # Random subsample for plotting (1.07M points overplot too heavily)
    rng = np.random.default_rng(42)
    n_plot = min(200000, n_peaks)
    idx = rng.choice(n_peaks, size=n_plot, replace=False)
    ax.scatter(detection_rate[idx], peak_mean_when_detected[idx],
                s=2, alpha=0.05, c="#4c72b0")
    ax.set_xlabel(f"Detection rate (number of samples with signal, out of {n_samples})")
    ax.set_ylabel("Mean signal at peak (across detected samples)")
    ax.set_title(f"Per-peak: mean signal vs detection rate\n"
                 f"(showing {n_plot:,} random peaks)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    # Overlay quantile lines per detection rate
    summary_lines = []
    for det_rate in range(1, n_samples + 1):
        mask = detection_rate == det_rate
        if mask.sum() == 0:
            continue
        vals = peak_mean_when_detected[mask]
        summary_lines.append({"detection_rate": det_rate,
                                "n_peaks": int(mask.sum()),
                                "mean_signal_median":
                                  float(np.median(vals)),
                                "mean_signal_q25":
                                  float(np.percentile(vals, 25)),
                                "mean_signal_q75":
                                  float(np.percentile(vals, 75))})
    sl_df = pd.DataFrame(summary_lines)
    ax.plot(sl_df["detection_rate"], sl_df["mean_signal_median"],
            color="red", linewidth=2, label="Median per detection rate")
    ax.fill_between(sl_df["detection_rate"], sl_df["mean_signal_q25"],
                      sl_df["mean_signal_q75"], color="red", alpha=0.2,
                      label="Q25-Q75 band")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output_dir / "03_signal_vs_detection_scatter.png",
                dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    sl_df.to_csv(args.output_dir / "03_signal_vs_detection_summary.tsv",
                  sep="\t", index=False)

    # ============================================================
    # Analysis 4: Pairwise sample intersection heatmap
    # ============================================================
    logging.info("Analysis 4: pairwise sample intersection heatmap")
    # For each pair (i, j): jaccard index of detected peak sets
    # Pairwise via boolean matrix multiplication
    intersect_matrix = (detected.T.astype(np.int32) @ detected.astype(np.int32))
    sample_totals = detected.sum(axis=0)  # peaks per sample
    union_matrix = (sample_totals[:, None] + sample_totals[None, :]
                    - intersect_matrix)
    with np.errstate(invalid="ignore", divide="ignore"):
        jaccard = np.where(union_matrix > 0,
                            intersect_matrix / union_matrix, 0)

    # Reorder samples by cancer type for visual grouping
    order = sorted(range(n_samples),
                    key=lambda i: (cancer_of_sample[i], samples[i]))
    j_ord = jaccard[np.ix_(order, order)]
    samples_ord = [samples[i] for i in order]
    cancers_ord = [cancer_of_sample[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 13))
    im = ax.imshow(j_ord, aspect="auto", cmap="Reds", vmin=0,
                     vmax=float(np.percentile(j_ord, 99)))
    ax.set_xticks(range(n_samples))
    ax.set_xticklabels(samples_ord, rotation=90, fontsize=7)
    ax.set_yticks(range(n_samples))
    ax.set_yticklabels(samples_ord, fontsize=7)
    ax.set_title(f"Pairwise sample Jaccard similarity\n"
                 f"(samples grouped by cancer type)",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Jaccard (intersection / union of detected peaks)",
                   pad=0.02, shrink=0.7)
    # Draw cancer-type boundary lines
    prev_c = cancers_ord[0]
    for i in range(1, n_samples):
        if cancers_ord[i] != prev_c:
            ax.axvline(i - 0.5, color="black", linewidth=0.6, alpha=0.6)
            ax.axhline(i - 0.5, color="black", linewidth=0.6, alpha=0.6)
            prev_c = cancers_ord[i]
    plt.tight_layout()
    plt.savefig(args.output_dir / "04_sample_intersection_heatmap.png",
                dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ============================================================
    # Analysis 5: Cumulative coverage curve
    # ============================================================
    logging.info("Analysis 5: cumulative coverage curve")
    fig, ax = plt.subplots(figsize=(10, 6))
    n_required = np.arange(0, n_samples + 1)
    n_surviving = np.array([(detection_rate >= n).sum()
                              for n in n_required])
    ax.plot(n_required, n_surviving, color="#4c72b0", linewidth=2,
            marker="o", markersize=4)
    ax.set_xlabel(f"Required minimum detection (in N of {n_samples} samples)")
    ax.set_ylabel("Number of peaks surviving filter")
    ax.set_title(f"Cumulative coverage curve\n"
                 f"How many peaks survive 'detected in >= N samples' filter",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    # Annotation at key thresholds
    for n in [1, 2, 5, 10, 20, 30, n_samples]:
        if n > n_samples:
            continue
        val = (detection_rate >= n).sum()
        ax.annotate(f"N≥{n}: {val:,}",
                     xy=(n, val), xytext=(n + 2, val + 5000),
                     fontsize=8, color="darkred",
                     arrowprops=dict(arrowstyle="->", color="darkred",
                                       lw=0.6, alpha=0.6))
    plt.tight_layout()
    plt.savefig(args.output_dir / "05_cumulative_coverage.png",
                dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    cov_df = pd.DataFrame({"min_detection": n_required,
                            "n_surviving_peaks": n_surviving,
                            "pct_surviving":
                              100.0 * n_surviving / n_peaks})
    cov_df.to_csv(args.output_dir / "05_cumulative_coverage.tsv",
                   sep="\t", index=False)

    # ============================================================
    # Analysis 6: Within-vs-across cancer-type Jaccard
    # ============================================================
    logging.info("Analysis 6: within-vs-across cancer-type sharing")
    within_per_cancer = {}
    across_per_cancer = {}
    for ct in unique_cancers:
        idxs = [i for i in range(n_samples)
                if cancer_of_sample[i] == ct]
        if len(idxs) < 2:
            within_per_cancer[ct] = []
        else:
            within_vals = []
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    within_vals.append(jaccard[idxs[a], idxs[b]])
            within_per_cancer[ct] = within_vals
        across_vals = []
        for i in idxs:
            for j in range(n_samples):
                if cancer_of_sample[j] != ct:
                    across_vals.append(jaccard[i, j])
        across_per_cancer[ct] = across_vals

    # Plot for cancer types with at least 2 samples
    fig, ax = plt.subplots(figsize=(max(12, len(unique_cancers) * 0.6), 6))
    pos = 0
    xticks = []
    xtick_labels = []
    box_data = []
    box_colors_list = []
    legend_handles = []
    for ct in unique_cancers:
        within = within_per_cancer[ct]
        across = across_per_cancer[ct]
        if len(within) > 0:
            box_data.append(within)
            box_colors_list.append("#dd8452")  # orange
            xticks.append(pos); xtick_labels.append(f"{ct}\nwithin\n(n={len(within)})")
            pos += 1
        if len(across) > 0:
            box_data.append(across)
            box_colors_list.append("#4c72b0")  # blue
            xticks.append(pos); xtick_labels.append(f"{ct}\nacross\n(n={len(across)})")
            pos += 1
        pos += 0.5  # gap between cancer-type groups
    bp = ax.boxplot(box_data, positions=xticks, widths=0.7,
                     patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors_list):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=0, fontsize=7)
    ax.set_ylabel("Pairwise Jaccard similarity")
    ax.set_title(f"Within-cancer-type vs across-cancer-type sample Jaccard\n"
                 f"Orange = within (same cancer), Blue = across (different cancer).\n"
                 f"If signal exists: orange should be higher than blue.",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    # Legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color="#dd8452", alpha=0.7,
                                       label="Within cancer type"),
                       plt.Rectangle((0, 0), 1, 1, color="#4c72b0", alpha=0.7,
                                       label="Across cancer types")]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.output_dir / "06_within_vs_across_cancer_sharing.png",
                dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Summary TSV for within/across
    rows = []
    for ct in unique_cancers:
        within = within_per_cancer[ct]
        across = across_per_cancer[ct]
        rows.append({"cancer_type":       ct,
                      "n_samples":         n_per_cancer[ct],
                      "n_within_pairs":    len(within),
                      "n_across_pairs":    len(across),
                      "within_jaccard_mean":   float(np.mean(within))
                                                  if len(within) else float("nan"),
                      "within_jaccard_median": float(np.median(within))
                                                  if len(within) else float("nan"),
                      "across_jaccard_mean":   float(np.mean(across))
                                                  if len(across) else float("nan"),
                      "across_jaccard_median": float(np.median(across))
                                                  if len(across) else float("nan")})
    within_df = pd.DataFrame(rows)
    within_df.to_csv(args.output_dir / "06_within_vs_across_summary.tsv",
                      sep="\t", index=False)

    # ============================================================
    # Final summary TSV
    # ============================================================
    summary = pd.DataFrame([{
        "n_peaks_total":               n_peaks,
        "n_samples":                   n_samples,
        "detection_threshold":         args.threshold,
        "peaks_universally_absent":    int((detection_rate == 0).sum()),
        "peaks_one_sample_only":       int((detection_rate == 1).sum()),
        "peaks_geq_half_samples":      int((detection_rate >= n_samples/2).sum()),
        "peaks_all_samples":           int((detection_rate == n_samples).sum()),
        "sample_min_peaks":            int(sample_peak_count.min()),
        "sample_min_peaks_sample":     samples[sample_peak_count.argmin()],
        "sample_max_peaks":            int(sample_peak_count.max()),
        "sample_max_peaks_sample":     samples[sample_peak_count.argmax()],
        "sample_median_peaks":         int(np.median(sample_peak_count)),
        "jaccard_overall_median":      float(np.median(jaccard[np.triu_indices(n_samples, k=1)])),
    }])
    summary.to_csv(args.output_dir / "summary_stats.tsv",
                    sep="\t", index=False)
    logging.info("")
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    for col in summary.columns:
        logging.info(f"  {col}: {summary[col].iloc[0]}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
