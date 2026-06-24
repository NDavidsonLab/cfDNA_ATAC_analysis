#!/usr/bin/env python3
"""
umap_atac_samples.py

UMAP of ATAC samples in k=1000 (or any k) cluster space.

Input: clusters x samples matrix (from build_atac_cluster_matrix.py)
Output: 2-panel figure
  - Left:  UMAP colored by source (TCGA / Blood / AML)
  - Right: UMAP colored by cell type (cancer types or hematopoietic types)

Sample type parsing (from sample_id suffixes):
  *.insertions    -> TCGA, cell_type = cancer-type prefix before "__" (e.g. ACC, BLCA, BRCA)
  *_short_peak    -> Blood, cell_type = prefix before _short_peak (e.g. Bcell, HSC, CMP)
  *_pk            -> AML, cell_type = "AML"
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_sample(sample_id: str) -> tuple[str, str]:
    """Return (source, cell_type)."""
    if sample_id.endswith(".insertions"):
        # TCGA prefix is everything before the first "__"
        m = re.match(r"^([A-Z]+)__", sample_id)
        ct = m.group(1) if m else "TCGA_unknown"
        return "TCGA", ct
    if sample_id.endswith("_short_peak"):
        ct = sample_id.replace("_short_peak", "")
        return "Blood", ct
    if sample_id.endswith("_pk"):
        return "AML", "AML"
    return "other", "other"


def zscore_rows(M):
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (M - mu) / sd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atac-matrix",   required=True, type=Path,
                   help="cluster x sample matrix")
    p.add_argument("--output-dir",    required=True, type=Path)
    p.add_argument("--label",         default="libnorm85_k1000")
    p.add_argument("--n-neighbors",   type=int, default=15)
    p.add_argument("--min-dist",      type=float, default=0.1)
    p.add_argument("--seed",          type=int, default=42)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "umap.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Loading: {args.atac_matrix}")
    M = pd.read_csv(args.atac_matrix, sep="\t", index_col=0)
    logging.info(f"  shape: {M.shape} (clusters x samples)")

    # Z-score per cluster across samples
    z = zscore_rows(M.values)
    # UMAP wants observations x features. Samples are observations, clusters
    # are features. So transpose: rows = samples, cols = clusters.
    X = z.T  # (n_samples, n_clusters)
    logging.info(f"  z-scored, transposed: {X.shape} (samples x clusters)")

    sample_ids = list(M.columns)
    parsed = [parse_sample(s) for s in sample_ids]
    sources = [p[0] for p in parsed]
    cell_types = [p[1] for p in parsed]
    logging.info(f"  source counts: {pd.Series(sources).value_counts().to_dict()}")
    logging.info(f"  cell-type unique count: {len(set(cell_types))}")

    # UMAP fit
    logging.info("Running UMAP")
    try:
        import umap
        UMAPClass = umap.UMAP
    except ImportError:
        try:
            from umap_learn import UMAP as UMAPClass
        except ImportError:
            raise ImportError(
                "Neither 'umap' nor 'umap_learn' is installed. "
                "Install with: conda install -c conda-forge umap-learn "
                "or pip install umap-learn"
            )
    reducer = UMAPClass(n_neighbors=args.n_neighbors,
                        min_dist=args.min_dist,
                        random_state=args.seed,
                        n_components=2)
    coords = reducer.fit_transform(X)
    logging.info(f"  UMAP shape: {coords.shape}")

    df = pd.DataFrame({
        "sample_id":  sample_ids,
        "source":     sources,
        "cell_type":  cell_types,
        "umap1":      coords[:, 0],
        "umap2":      coords[:, 1],
    })
    df.to_csv(args.output_dir / f"umap_coords_{args.label}.tsv",
              sep="\t", index=False)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: color by source
    source_colors = {
        "TCGA": "#4c72b0", "Blood": "#dd8452", "AML": "#55a868", "other": "#888"
    }
    for src in sorted(set(sources)):
        mask = df["source"] == src
        axes[0].scatter(df.loc[mask, "umap1"], df.loc[mask, "umap2"],
                        s=25, alpha=0.7, c=source_colors.get(src, "#888"),
                        label=f"{src} (n={mask.sum()})",
                        edgecolors="white", linewidths=0.3)
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].set_title(f"ATAC samples in k=1000 cluster space — by source\n"
                       f"({args.label})", fontsize=11, fontweight="bold")
    axes[0].legend(loc="best", fontsize=9, framealpha=0.9)
    axes[0].grid(alpha=0.3)

    # Right: color by cell type. Many categories — use tab20 + extend.
    unique_cts = sorted(set(cell_types))
    n_ct = len(unique_cts)
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
    # Group small categories visually as outline-only to reduce legend clutter
    # but plot every point
    for ct in unique_cts:
        mask = df["cell_type"] == ct
        axes[1].scatter(df.loc[mask, "umap1"], df.loc[mask, "umap2"],
                        s=25, alpha=0.8, c=[ct_colors[ct]],
                        label=f"{ct} (n={mask.sum()})",
                        edgecolors="white", linewidths=0.3)
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].set_title(f"ATAC samples in k=1000 cluster space — by cell type\n"
                       f"({args.label})", fontsize=11, fontweight="bold")
    # Legend: many entries, put outside plot
    leg = axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                         fontsize=6, ncol=2, framealpha=0.9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = args.output_dir / f"umap_atac_{args.label}.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info(f"Wrote {out_path}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
