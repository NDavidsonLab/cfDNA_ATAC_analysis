#!/usr/bin/env python3
"""
strip_hematopoietic_atac_visuals.py

Produce before/after analysis of stripping hematopoietic-dominant clusters
from the ATAC k=1000 cluster space.

Output: 7 separate PNGs + summary TSVs + JSON

  01_heatmap_before_cluster_enrichment.png    cluster enrichment, all 1000
  02_heatmap_after_cluster_enrichment.png     cluster enrichment, kept only
  03_umap_before_by_source.png                UMAP all clusters, by source
  04_umap_before_by_celltype.png              UMAP all clusters, by cell type
  05_umap_after_by_source.png                 UMAP kept clusters, by source
  06_umap_after_by_celltype.png               UMAP kept clusters, by cell type
  07_drop_counts_barchart.png                 cluster counts dropped/kept

  hematopoietic_clusters.tsv                  the dropped cluster IDs
  cancer_dominant_clusters.tsv                the kept cluster IDs
  umap_coords_before_after.tsv                UMAP coords + sample metadata
  strip_summary.json                          all numeric stats
  strip.log                                   run log

Inputs:
  --atac-matrix       cluster x sample matrix (k=1000 libnorm85)
  --atac-metadata     sample_id -> cell_type
  --output-dir
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


HEMATOPOIETIC_TYPES = {
    "HSC", "MPP", "CMP", "LMPP", "MEP", "GMP",
    "Ery", "Mono", "Bcell", "CD4Tcell", "CD8Tcell",
    "NKcell", "CLP", "HEM", "AML",
}


def parse_sample(sample_id: str) -> tuple[str, str]:
    if sample_id.endswith(".insertions"):
        m = re.match(r"^([A-Z]+)__", sample_id)
        ct = m.group(1) if m else "TCGA_unknown"
        return "TCGA", ct
    if sample_id.endswith("_short_peak"):
        return "Blood", sample_id.replace("_short_peak", "")
    if sample_id.endswith("_pk"):
        return "AML", "AML"
    return "other", "other"


def zscore_rows(M):
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (M - mu) / sd


def compute_dominant(atac, atac_meta):
    sample_to_ct = dict(zip(atac_meta["sample_id"], atac_meta["cell_type"]))
    all_cts = sorted(set(sample_to_ct.values()))
    centroids = {}
    for ct in all_cts:
        sids = [s for s in atac.columns if sample_to_ct.get(s) == ct]
        if len(sids) == 0:
            continue
        centroids[ct] = atac[sids].median(axis=1)
    centroid_df = pd.DataFrame(centroids, index=atac.index)
    z = zscore_rows(centroid_df.values)
    z_df = pd.DataFrame(z, index=centroid_df.index, columns=centroid_df.columns)
    dominant = z_df.idxmax(axis=1)
    return dominant, centroid_df, z_df, list(centroid_df.columns)


def run_umap(M, n_neighbors=15, min_dist=0.1, seed=42):
    try:
        import umap
        UMAPClass = umap.UMAP
    except ImportError:
        from umap_learn import UMAP as UMAPClass
    z = zscore_rows(M.values)
    X = z.T
    reducer = UMAPClass(n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=seed, n_components=2)
    return reducer.fit_transform(X)


def plot_heatmap(z_df, cluster_set, dominant, all_cts, out_path,
                 title, hematopoietic_set):
    """Cluster-enrichment heatmap: rows=dominant types, cols=all cell types."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(cluster_set) == 0:
        # Blank
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "(no clusters)", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.axis("off")
        plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return
    dom_subset = dominant.loc[list(cluster_set)]
    dom_counts = dom_subset.value_counts()
    agg_rows = []
    row_labels = []
    for ct in dom_counts.index:
        cids = dom_subset[dom_subset == ct].index
        agg_rows.append(z_df.loc[cids].mean(axis=0).values)
        flag = " [hema]" if ct in hematopoietic_set else " [CANCER]"
        row_labels.append(f"{ct} ({len(cids)} clusters){flag}")
    agg = np.array(agg_rows)
    n_rows, n_cols = agg.shape
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.32),
                                    max(6, n_rows * 0.35)))
    vmin = max(0.0, float(np.nanmin(agg)))
    vmax = float(np.nanmax(agg))
    im = ax.imshow(agg, aspect="auto", cmap="Reds", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(all_cts, rotation=90, fontsize=6)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Dominant cell type")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="mean z-score", pad=0.02, shrink=0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_umap_by_source(sample_df, x_col, y_col, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    source_colors = {"TCGA": "#4c72b0", "Blood": "#dd8452",
                     "AML": "#55a868", "other": "#888"}
    fig, ax = plt.subplots(figsize=(10, 9))
    for src in sorted(set(sample_df["source"])):
        mask = sample_df["source"] == src
        ax.scatter(sample_df.loc[mask, x_col], sample_df.loc[mask, y_col],
                   s=35, alpha=0.7, c=source_colors.get(src, "#888"),
                   label=f"{src} (n={mask.sum()})",
                   edgecolors="white", linewidths=0.4)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_umap_by_celltype(sample_df, x_col, y_col, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    unique_cts = sorted(set(sample_df["cell_type"]))
    cmap1 = plt.cm.get_cmap("tab20")
    cmap2 = plt.cm.get_cmap("tab20b")
    cmap3 = plt.cm.get_cmap("tab20c")
    ct_colors = {}
    for i, ct in enumerate(unique_cts):
        if i < 20:
            ct_colors[ct] = cmap1(i)
        elif i < 40:
            ct_colors[ct] = cmap2(i - 20)
        else:
            ct_colors[ct] = cmap3((i - 40) % 20)
    fig, ax = plt.subplots(figsize=(13, 9))
    for ct in unique_cts:
        mask = sample_df["cell_type"] == ct
        ax.scatter(sample_df.loc[mask, x_col], sample_df.loc[mask, y_col],
                   s=35, alpha=0.8, c=[ct_colors[ct]],
                   label=f"{ct} (n={mask.sum()})" if mask.sum() >= 2 else None,
                   edgecolors="white", linewidths=0.4)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_drop_counts(hema_counts, cancer_counts, out_path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    drop_items = sorted(hema_counts.items(), key=lambda x: -x[1])
    keep_items = sorted(cancer_counts.items(), key=lambda x: -x[1])
    drop_labels = [k for k, _ in drop_items]
    drop_values = [v for _, v in drop_items]
    keep_labels = [k for k, _ in keep_items]
    keep_values = [v for _, v in keep_items]
    y_drop = np.arange(len(drop_labels))
    y_keep = np.arange(len(keep_labels)) + len(drop_labels) + 1
    fig, ax = plt.subplots(figsize=(11, max(7, (len(drop_labels) + len(keep_labels)) * 0.32)))
    ax.barh(y_drop, drop_values, color="#dd8452",
             label=f"hematopoietic DROPPED (n={sum(drop_values)})")
    ax.barh(y_keep, keep_values, color="#4c72b0",
             label=f"cancer KEPT (n={sum(keep_values)})")
    ax.set_yticks(list(y_drop) + list(y_keep))
    ax.set_yticklabels(drop_labels + keep_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of clusters", fontsize=10)
    ax.set_title(f"Cluster counts by dominant type — {label}",
                  fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    for i, v in enumerate(drop_values):
        ax.text(v + max(drop_values + keep_values) * 0.01, y_drop[i],
                str(v), va="center", fontsize=9)
    for i, v in enumerate(keep_values):
        ax.text(v + max(drop_values + keep_values) * 0.01, y_keep[i],
                str(v), va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atac-matrix",    required=True, type=Path)
    p.add_argument("--atac-metadata",  required=True, type=Path)
    p.add_argument("--output-dir",     required=True, type=Path)
    p.add_argument("--label",          default="libnorm85_k1000")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "strip.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Loading: {args.atac_matrix}")
    atac = pd.read_csv(args.atac_matrix, sep="\t", index_col=0)
    logging.info(f"  shape: {atac.shape} (clusters x samples)")
    meta = pd.read_csv(args.atac_metadata, sep="\t")

    dominant, centroid_df, z_df, all_cts = compute_dominant(atac, meta)
    n_total = len(dominant)
    logging.info(f"Total clusters: {n_total}")

    hema_mask = dominant.isin(HEMATOPOIETIC_TYPES)
    cancer_mask = ~hema_mask
    hema_clusters = dominant.index[hema_mask].tolist()
    cancer_clusters = dominant.index[cancer_mask].tolist()
    logging.info(f"  Hematopoietic-dominant (DROP): {len(hema_clusters)}")
    logging.info(f"  Cancer-dominant (KEEP):       {len(cancer_clusters)}")

    pd.DataFrame({"cluster_id": hema_clusters,
                   "dominant_cell_type": [dominant[c] for c in hema_clusters]}
                 ).to_csv(args.output_dir / "hematopoietic_clusters.tsv",
                          sep="\t", index=False)
    pd.DataFrame({"cluster_id": cancer_clusters,
                   "dominant_cell_type": [dominant[c] for c in cancer_clusters]}
                 ).to_csv(args.output_dir / "cancer_dominant_clusters.tsv",
                          sep="\t", index=False)

    hema_drop_counts = Counter(dominant[c] for c in hema_clusters)
    cancer_keep_counts = Counter(dominant[c] for c in cancer_clusters)
    logging.info(f"Hematopoietic types dropped:")
    for ct, n in sorted(hema_drop_counts.items(), key=lambda x: -x[1]):
        logging.info(f"  {ct}: {n}")
    logging.info(f"Cancer types kept:")
    for ct, n in sorted(cancer_keep_counts.items(), key=lambda x: -x[1]):
        logging.info(f"  {ct}: {n}")

    # ============== Heatmaps (independent of UMAP) ==============
    logging.info("Plot 01: heatmap BEFORE")
    plot_heatmap(z_df, dominant.index, dominant, all_cts,
                 args.output_dir / "01_heatmap_before_cluster_enrichment.png",
                 f"BEFORE: cluster enrichment, all {n_total} clusters\n"
                 f"({args.label})",
                 HEMATOPOIETIC_TYPES)
    logging.info("Plot 02: heatmap AFTER")
    plot_heatmap(z_df, cancer_clusters, dominant, all_cts,
                 args.output_dir / "02_heatmap_after_cluster_enrichment.png",
                 f"AFTER: cluster enrichment, {len(cancer_clusters)} cancer-dominant clusters\n"
                 f"({args.label})",
                 HEMATOPOIETIC_TYPES)

    # ============== Drop counts (independent of UMAP) ==============
    logging.info("Plot 07: drop counts bar chart")
    plot_drop_counts(hema_drop_counts, cancer_keep_counts,
                      args.output_dir / "07_drop_counts_barchart.png",
                      args.label)

    # ============== UMAPs ==============
    parsed = [parse_sample(s) for s in atac.columns]
    sample_df = pd.DataFrame({
        "sample_id": atac.columns,
        "source":    [p[0] for p in parsed],
        "cell_type": [p[1] for p in parsed],
    })

    logging.info("UMAP BEFORE (all clusters)")
    coords_before = run_umap(atac)
    sample_df["umap1_before"] = coords_before[:, 0]
    sample_df["umap2_before"] = coords_before[:, 1]

    if len(cancer_clusters) >= 5:
        logging.info(f"UMAP AFTER ({len(cancer_clusters)} cancer-dominant clusters)")
        coords_after = run_umap(atac.loc[cancer_clusters])
        sample_df["umap1_after"] = coords_after[:, 0]
        sample_df["umap2_after"] = coords_after[:, 1]
    else:
        logging.warning(f"Only {len(cancer_clusters)} cancer-dominant clusters "
                        f"— UMAP AFTER skipped")
        sample_df["umap1_after"] = np.nan
        sample_df["umap2_after"] = np.nan

    sample_df.to_csv(args.output_dir / "umap_coords_before_after.tsv",
                     sep="\t", index=False)

    logging.info("Plot 03: UMAP BEFORE by source")
    plot_umap_by_source(sample_df, "umap1_before", "umap2_before",
                         args.output_dir / "03_umap_before_by_source.png",
                         f"UMAP BEFORE strip — by SOURCE\n"
                         f"(all {n_total} clusters, {args.label})")
    logging.info("Plot 04: UMAP BEFORE by cell type")
    plot_umap_by_celltype(sample_df, "umap1_before", "umap2_before",
                           args.output_dir / "04_umap_before_by_celltype.png",
                           f"UMAP BEFORE strip — by CELL TYPE\n"
                           f"(all {n_total} clusters, {args.label})")
    if not sample_df["umap1_after"].isna().all():
        logging.info("Plot 05: UMAP AFTER by source")
        plot_umap_by_source(sample_df, "umap1_after", "umap2_after",
                             args.output_dir / "05_umap_after_by_source.png",
                             f"UMAP AFTER strip — by SOURCE\n"
                             f"({len(cancer_clusters)} cancer-dominant clusters, "
                             f"{args.label})")
        logging.info("Plot 06: UMAP AFTER by cell type")
        plot_umap_by_celltype(sample_df, "umap1_after", "umap2_after",
                               args.output_dir / "06_umap_after_by_celltype.png",
                               f"UMAP AFTER strip — by CELL TYPE\n"
                               f"({len(cancer_clusters)} cancer-dominant clusters, "
                               f"{args.label})")

    # Summary JSON
    pct_dropped = 100.0 * len(hema_clusters) / max(n_total, 1)
    summary_dict = {
        "label": args.label,
        "n_total_clusters":    n_total,
        "n_dropped_hema":      len(hema_clusters),
        "n_kept_cancer":       len(cancer_clusters),
        "pct_dropped":         float(pct_dropped),
        "pct_kept":            float(100.0 - pct_dropped),
        "hema_drop_counts":    dict(hema_drop_counts),
        "cancer_keep_counts":  dict(cancer_keep_counts),
        "n_tcga_samples":      int((sample_df["source"] == "TCGA").sum()),
        "n_blood_samples":     int((sample_df["source"] == "Blood").sum()),
        "n_aml_samples":       int((sample_df["source"] == "AML").sum()),
    }
    with open(args.output_dir / "strip_summary.json", "w") as fh:
        json.dump(summary_dict, fh, indent=2)
    logging.info(f"Wrote strip_summary.json")
    logging.info(f"")
    logging.info(f"== FINAL ==")
    logging.info(f"  Total clusters:    {n_total}")
    logging.info(f"  Dropped (hema):    {len(hema_clusters)} ({pct_dropped:.1f}%)")
    logging.info(f"  Kept (cancer):     {len(cancer_clusters)} ({100-pct_dropped:.1f}%)")
    logging.info("Done.")


if __name__ == "__main__":
    main()
