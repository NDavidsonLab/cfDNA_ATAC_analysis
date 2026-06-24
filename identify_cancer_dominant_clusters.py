#!/usr/bin/env python3
"""
identify_cancer_dominant_clusters.py

Given a new ATAC cluster matrix (clusters x samples) and metadata, compute
each cluster's dominant cell type across ALL cell types (15 hematopoietic +
TCGA cancer types). Filter clusters where dominant is a cancer type (i.e.,
NOT hematopoietic), and produce a verification heatmap.

Inputs:
  --atac-matrix      atac_cluster_matrix_median.tsv.gz (clusters x samples)
  --atac-metadata    atac_cluster_metadata.tsv with sample_id, cell_type
  --output-dir       where to write
  --kept-clusters-tsv  cluster IDs whose dominant cell type is cancer
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
    p.add_argument("--atac-matrix",   required=True, type=Path)
    p.add_argument("--atac-metadata", required=True, type=Path)
    p.add_argument("--output-dir",    required=True, type=Path)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "identify_cancer.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Loading ATAC matrix: {args.atac_matrix}")
    atac = pd.read_csv(args.atac_matrix, sep="\t", index_col=0)
    logging.info(f"  shape: {atac.shape} (clusters x samples)")

    meta = pd.read_csv(args.atac_metadata, sep="\t")
    sample_to_ct = dict(zip(meta["sample_id"], meta["cell_type"]))

    # All cell types present in metadata
    all_cts = sorted(set(sample_to_ct.values()))
    logging.info(f"All cell types in ATAC metadata: {len(all_cts)}")
    hema_cts = [ct for ct in all_cts if ct in HEMATOPOIETIC_TYPES]
    cancer_cts = [ct for ct in all_cts if ct not in HEMATOPOIETIC_TYPES]
    logging.info(f"  Hematopoietic ({len(hema_cts)}): {hema_cts}")
    logging.info(f"  Cancer ({len(cancer_cts)}): {cancer_cts}")

    # Build per-cell-type centroids: median across samples of that cell type
    centroids = {}
    for ct in all_cts:
        sids = [s for s in atac.columns if sample_to_ct.get(s) == ct]
        if len(sids) == 0:
            continue
        centroids[ct] = atac[sids].median(axis=1)
    centroid_df = pd.DataFrame(centroids, index=atac.index)
    logging.info(f"Centroid matrix: {centroid_df.shape}")

    # Z-score per cluster across all cell types, then dominant = argmax
    z = zscore_rows(centroid_df.values)
    z_df = pd.DataFrame(z, index=centroid_df.index, columns=centroid_df.columns)
    dominant = z_df.idxmax(axis=1)

    # Split clusters by dominant type category
    is_cancer_dominant = dominant.isin(cancer_cts)
    is_hema_dominant   = dominant.isin(hema_cts)
    kept = dominant.index[is_cancer_dominant].tolist()
    dropped = dominant.index[is_hema_dominant].tolist()
    logging.info(f"Cluster categorization:")
    logging.info(f"  total clusters:               {len(dominant)}")
    logging.info(f"  kept (cancer-dominant):       {len(kept)}")
    logging.info(f"  dropped (hematopoietic-dom.): {len(dropped)}")

    # Save the kept-cluster list AND the full assignment table
    kept_df = pd.DataFrame({
        "cluster_id":         kept,
        "dominant_cell_type": [dominant[c] for c in kept],
    })
    kept_path = args.output_dir / "kept_clusters_cancer_dominant.tsv"
    kept_df.to_csv(kept_path, sep="\t", index=False)
    logging.info(f"Wrote {kept_path}")

    all_assign = pd.DataFrame({
        "cluster_id":         dominant.index,
        "dominant_cell_type": dominant.values,
        "is_cancer_dominant": is_cancer_dominant.values,
    })
    all_assign.to_csv(args.output_dir / "all_clusters_dominant_assignment.tsv",
                      sep="\t", index=False)

    # Cluster counts per dominant type, sorted descending
    dom_counts = dominant.value_counts()
    logging.info(f"Cluster counts per dominant cell type:")
    for ct, n in dom_counts.items():
        flag = " [CANCER]" if ct in cancer_cts else " [hema]"
        logging.info(f"  {ct}: {n} clusters{flag}")

    # ---------- Verification heatmap ----------
    # Rows: dominant cell type (with cluster count), sorted by count desc
    # Cols: same cell types (the centroid columns)
    # Color: mean z-score
    # Visual: should show diagonal block, with kept-cancer rows in their own
    # part of the figure visually distinct from the hematopoietic rows.
    logging.info("Building verification heatmap")
    row_order = list(dom_counts.index)
    agg_rows = []
    row_labels = []
    for dom_ct in row_order:
        cluster_ids = dominant.index[dominant == dom_ct]
        agg_rows.append(z_df.loc[cluster_ids].mean(axis=0).values)
        flag = " [CANCER]" if dom_ct in cancer_cts else ""
        row_labels.append(f"{dom_ct} ({len(cluster_ids)} clusters){flag}")
    agg = np.array(agg_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_rows, n_cols = agg.shape
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.32),
                                    max(8, n_rows * 0.32)))
    vmin = max(0.0, float(np.nanmin(agg)))
    vmax = float(np.nanmax(agg))
    im = ax.imshow(agg, aspect="auto", cmap="Reds", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(z_df.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_ylabel("Dominant cell type (with cluster count); [CANCER] = kept")
    ax.set_xlabel("Cell type")
    title = (f"Cluster enrichment by dominant cell type (verification)\n"
             f"libnorm85 k=40000 — {len(kept)} cancer-dominant kept "
             f"of {len(dominant)} total")
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="mean z-score", pad=0.02, shrink=0.85)
    plt.tight_layout()
    out_png = args.output_dir / "verification_heatmap_libnorm85_k40000.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info(f"Wrote {out_png}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
