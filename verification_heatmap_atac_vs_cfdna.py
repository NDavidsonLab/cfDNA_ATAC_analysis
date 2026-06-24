#!/usr/bin/env python3
"""
verification_heatmap_atac_vs_cfdna.py

Verification heatmap testing whether ATAC k=1000 libnorm85 clusters can
predict cfDNA cancer type identity.

Layout (mirrors existing verification_heatmap_libnorm85_k40000.png):
  - Rows: ATAC dominant cell type per cluster (with cluster count, [CANCER]
    flag for non-hematopoietic).
  - Cols: cfDNA cancer type (from cfdna_sample_metadata, with sample count
    annotated).
  - Color: mean z-score per cluster group in each cfDNA cancer-type centroid.

If the diagonal is sharp (BRCA-dominant clusters light up in BRCA cfDNA,
LUAD in LUAD, etc.) — ATAC clusters predict cfDNA identity well. Off-diagonal
heat = misclassification or shared signal between cancer types.

Inputs:
  --cfdna-cluster-matrix   cfdna_cluster_matrix_median.tsv.gz (1000 x 43)
  --clusters-tsv           clusters_persample_k1000_libnorm85/clusters.tsv
  --cfdna-metadata         cfdna_sample_metadata.tsv (sample_id, cancer_type)
  --output-dir
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


HEMATOPOIETIC_TYPES = {
    "HSC", "MPP", "CMP", "LMPP", "MEP", "GMP",
    "Ery", "Mono", "Bcell", "CD4Tcell", "CD8Tcell",
    "NKcell", "CLP", "HEM", "AML",
}


def zscore_rows(M):
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (M - mu) / sd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfdna-cluster-matrix", required=True, type=Path)
    p.add_argument("--clusters-tsv",         required=True, type=Path)
    p.add_argument("--cfdna-metadata",       required=True, type=Path)
    p.add_argument("--output-dir",           required=True, type=Path)
    p.add_argument("--label",                default="libnorm85_k1000")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "verification.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Load cfDNA cluster matrix
    logging.info(f"Loading cfDNA cluster matrix: {args.cfdna_cluster_matrix}")
    cfdna = pd.read_csv(args.cfdna_cluster_matrix, sep="\t", index_col=0)
    cfdna.index = cfdna.index.astype(str)
    logging.info(f"  shape: {cfdna.shape} (clusters x cfDNA samples)")

    # Load ATAC cluster -> dominant cell type from clusters.tsv
    logging.info(f"Loading ATAC dominant cell type per cluster")
    clusters = pd.read_csv(args.clusters_tsv, sep="\t",
                            usecols=["peak_id", "cluster", "dominant_cell_type"])
    # One row per peak — collapse to one row per cluster
    cluster_dom = (clusters[["cluster", "dominant_cell_type"]]
                    .drop_duplicates(subset="cluster")
                    .set_index("cluster")["dominant_cell_type"])
    cluster_dom.index = cluster_dom.index.astype(str)
    logging.info(f"  {len(cluster_dom)} clusters with dominant type")
    dom_counts = cluster_dom.value_counts()
    logging.info(f"  unique dominant types: {len(dom_counts)}")

    # Load cfDNA sample metadata
    meta = pd.read_csv(args.cfdna_metadata, sep="\t")
    logging.info(f"  {len(meta)} cfDNA samples in metadata")
    sample_to_cancer = dict(zip(meta["sample_id"], meta["cancer_type"]))

    # Restrict to samples present in both matrix and metadata
    common = [s for s in cfdna.columns if s in sample_to_cancer]
    cfdna = cfdna[common]
    n_per_cancer = pd.Series([sample_to_cancer[s] for s in common]
                              ).value_counts()
    logging.info(f"cfDNA cancer-type sample counts:")
    for ct, n in n_per_cancer.items():
        logging.info(f"  {ct}: {n}")

    # Build per-cancer-type centroid (mean across samples)
    cancer_types = sorted(n_per_cancer.index)
    centroids = {}
    for ct in cancer_types:
        sids = [s for s in cfdna.columns if sample_to_cancer.get(s) == ct]
        centroids[ct] = cfdna[sids].mean(axis=1)
    centroid_df = pd.DataFrame(centroids, index=cfdna.index)
    logging.info(f"Centroid matrix: {centroid_df.shape} (clusters x cancer types)")

    # Z-score per cluster across cancer-type centroids
    z = zscore_rows(centroid_df.values)
    z_df = pd.DataFrame(z, index=centroid_df.index,
                         columns=centroid_df.columns)

    # Align cluster_dom and z_df
    common_clusters = z_df.index.intersection(cluster_dom.index)
    z_df = z_df.loc[common_clusters]
    cluster_dom = cluster_dom.loc[common_clusters]
    logging.info(f"After alignment: {len(common_clusters)} clusters with both "
                 f"cfDNA values and ATAC dominant type")

    # For each ATAC dominant type (rows), aggregate mean z-score across its
    # clusters, in each cfDNA cancer type column.
    row_order = list(dom_counts.index)  # ordered by ATAC cluster count desc
    n_total_clusters = len(cluster_dom)
    n_cancer_dom = sum(1 for ct in row_order if ct not in HEMATOPOIETIC_TYPES)
    n_kept = sum(int(dom_counts[ct]) for ct in row_order
                  if ct not in HEMATOPOIETIC_TYPES)

    agg_rows = []
    row_labels = []
    for dom_ct in row_order:
        # All clusters where ATAC dominant = dom_ct
        cluster_ids = cluster_dom[cluster_dom == dom_ct].index
        cluster_ids = cluster_ids.intersection(z_df.index)
        if len(cluster_ids) == 0:
            continue
        agg_rows.append(z_df.loc[cluster_ids].mean(axis=0).values)
        flag = " [CANCER]" if dom_ct not in HEMATOPOIETIC_TYPES else ""
        row_labels.append(f"{dom_ct} ({len(cluster_ids)} clusters){flag}")
    agg = np.array(agg_rows)
    logging.info(f"Aggregated matrix: {agg.shape} (ATAC dom types x cfDNA cancer types)")

    # Column labels with sample counts
    col_labels = [f"{ct} (n={n_per_cancer[ct]})" for ct in cancer_types]

    # Save the underlying matrix
    out_df = pd.DataFrame(agg, index=row_labels, columns=col_labels)
    out_df.to_csv(args.output_dir / "atac_dom_x_cfdna_cancer_matrix.tsv",
                  sep="\t")
    logging.info(f"Wrote underlying matrix")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = agg.shape
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.45),
                                    max(8, n_rows * 0.32)))
    vmin = max(0.0, float(np.nanmin(agg)))
    vmax = float(np.nanmax(agg))
    im = ax.imshow(agg, aspect="auto", cmap="Reds", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("cfDNA cancer type (n samples)")
    ax.set_ylabel("ATAC dominant cell type (with cluster count); [CANCER] = kept")
    title = (f"cfDNA prediction verification\n"
             f"ATAC clusters (rows) vs cfDNA cancer types (cols) — "
             f"{args.label}\n"
             f"{n_cancer_dom} cancer-dominant cluster types out of "
             f"{len(row_labels)} total")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="mean z-score", pad=0.02, shrink=0.85)
    plt.tight_layout()
    out_png = (args.output_dir /
                f"verification_atac_vs_cfdna_{args.label}.png")
    plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info(f"Wrote {out_png}")

    # Sanity diagonal score: for each cfDNA cancer type, find the row whose
    # dominant type matches and report its z-score. Higher = better prediction.
    logging.info("Per-cfDNA-cancer-type diagonal score:")
    diag_scores = {}
    for j, ct in enumerate(cancer_types):
        # Find matching row index
        match_row = None
        for i, label in enumerate(row_labels):
            row_ct = label.split(" (")[0]
            if row_ct == ct:
                match_row = i
                break
        if match_row is not None:
            score = agg[match_row, j]
            diag_scores[ct] = float(score)
            logging.info(f"  {ct} (n={n_per_cancer[ct]}): z={score:.3f}")
        else:
            logging.info(f"  {ct}: no matching ATAC dominant row")
    diag_df = pd.DataFrame({"cancer_type": list(diag_scores.keys()),
                             "diagonal_z_score": list(diag_scores.values()),
                             "n_samples": [n_per_cancer[c]
                                            for c in diag_scores.keys()]})
    diag_df.to_csv(args.output_dir / "diagonal_scores.tsv",
                    sep="\t", index=False)

    logging.info("Done.")


if __name__ == "__main__":
    main()
