#!/usr/bin/env python3
"""
replot_aml_atac_clean.py

Re-render the AML ATAC patient heatmap (matrix already computed) with:
  - AML row dropped (it dominates the colormap and the comparison is
    semi-circular — AML samples enriching in AML-dominant clusters)
  - Tighter colorbar so the remaining rows show structure
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-tsv", required=True, type=Path,
                   help="aml_atac_patients_matrix.tsv from the earlier run")
    p.add_argument("--output-png", required=True, type=Path)
    p.add_argument("--drop-aml-row", action="store_true", default=True)
    p.add_argument("--keep-aml-row", action="store_true",
                   help="Override --drop-aml-row")
    args = p.parse_args()

    df = pd.read_csv(args.matrix_tsv, sep="\t", index_col=0)
    print(f"Loaded matrix: {df.shape}")
    print(f"Original rows: {list(df.index)}")

    if args.drop_aml_row and not args.keep_aml_row:
        aml_rows = [i for i in df.index if i.startswith("AML")]
        df = df.drop(index=aml_rows)
        print(f"Dropped AML row(s): {aml_rows}")
        print(f"Remaining rows: {len(df)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = df.values
    n_rows, n_cols = arr.shape
    vmax = float(np.nanmax(arr))
    vmin = max(0.0, float(np.nanmin(arr)))
    print(f"Color range: [{vmin:.2f}, {vmax:.2f}]")

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.55),
                                    max(6, n_rows * 0.45)))
    im = ax.imshow(arr, aspect="auto", cmap="Reds", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(df.columns, rotation=90, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(df.index, fontsize=9)
    ax.set_ylabel("Dominant ATAC cell type (with cluster count)")
    ax.set_xlabel("AML ATAC sample")
    ax.set_title("Cluster enrichment, aggregated by dominant ATAC cell type\n"
                 "(AML ATAC patients on x-axis)",
                 fontsize=11, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("mean z-score", fontsize=9)
    plt.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_png, dpi=150, bbox_inches="tight",
                  facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
