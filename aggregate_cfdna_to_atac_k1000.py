#!/usr/bin/env python3
"""
aggregate_cfdna_to_atac_k1000.py

Aggregate the libnorm85-normalized cfDNA per-peak SE matrix (1.07M peaks
× 43 samples) into the ATAC k=1000 libnorm85 cluster space, using median
aggregation per cluster.

The mapping is taken from clusters_persample_k1000_libnorm85/clusters.tsv
(peak_id -> cluster column). This produces a 1000-cluster × 43-sample
cfDNA matrix at the same cluster space as the ATAC cluster matrix, enabling
direct sample-by-sample comparison between cfDNA and ATAC.

Inputs:
  --cfdna-matrix      cfdna_se_per_peak_libnorm85.tsv.gz (1.07M peaks x 43)
  --clusters-tsv      clusters_persample_k1000_libnorm85/clusters.tsv
  --output-dir        where to write
  --aggregation       median (default) or mean

Outputs:
  cfdna_cluster_matrix_<agg>.tsv.gz    1000 clusters x 43 samples
  cfdna_cluster_sizes.tsv              n_peaks_in_cluster, n_peaks_with_signal
  aggregate.log
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
    p.add_argument("--cfdna-matrix",  required=True, type=Path)
    p.add_argument("--clusters-tsv",  required=True, type=Path)
    p.add_argument("--output-dir",    required=True, type=Path)
    p.add_argument("--aggregation",   choices=["median", "mean"],
                   default="median")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "aggregate.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Loading cluster assignments: {args.clusters_tsv}")
    clusters = pd.read_csv(args.clusters_tsv, sep="\t")
    logging.info(f"  shape: {clusters.shape}, columns: {list(clusters.columns)}")
    # Explicit column names (avoid the rank_in_cluster bug from earlier)
    if "peak_id" not in clusters.columns or "cluster" not in clusters.columns:
        raise ValueError(f"Expected 'peak_id' and 'cluster' columns; got "
                         f"{list(clusters.columns)}")
    peak_to_cluster = dict(zip(clusters["peak_id"].astype(str),
                                clusters["cluster"].astype(str)))
    unique_clusters = sorted(set(peak_to_cluster.values()),
                              key=lambda x: int(x) if x.isdigit() else x)
    logging.info(f"  {len(unique_clusters)} unique cluster IDs")
    logging.info(f"  {len(peak_to_cluster)} peak->cluster assignments")

    logging.info(f"Loading cfDNA matrix: {args.cfdna_matrix}")
    # Read header to get sample columns
    header = pd.read_csv(args.cfdna_matrix, sep="\t", nrows=0)
    sample_cols = [c for c in header.columns if c != header.columns[0]]
    dtype_dict = {c: np.float32 for c in sample_cols}
    cfdna = pd.read_csv(args.cfdna_matrix, sep="\t", index_col=0,
                        dtype=dtype_dict)
    logging.info(f"  shape: {cfdna.shape} (peaks x samples)")

    # Add cluster column
    cfdna.index = cfdna.index.astype(str)
    cfdna["_cluster"] = [peak_to_cluster.get(p) for p in cfdna.index]
    n_unmapped = cfdna["_cluster"].isna().sum()
    logging.info(f"  {n_unmapped} peaks not in clusters.tsv "
                 f"(should be 0 if cfDNA was computed on same peak set as ATAC)")

    n_with_signal_before_drop = len(cfdna)
    cfdna = cfdna.dropna(subset=["_cluster"])
    logging.info(f"  {len(cfdna)} peaks retained after cluster filter")

    # Aggregate
    logging.info(f"Aggregating by cluster ({args.aggregation})")
    if args.aggregation == "median":
        agg = cfdna.groupby("_cluster").median()
    else:
        agg = cfdna.groupby("_cluster").mean()
    logging.info(f"  cluster matrix shape: {agg.shape}")

    # Reorder columns to match the original sample order
    agg = agg[sample_cols]

    # Save cluster sizes (n_peaks per cluster)
    sizes = cfdna.groupby("_cluster").size().rename("n_peaks").to_frame()
    sizes["n_peaks_with_any_signal"] = (
        cfdna.groupby("_cluster")[sample_cols]
        .apply(lambda x: (x != 0).any(axis=1).sum())
    )
    sizes.index.name = "cluster"
    sizes.to_csv(args.output_dir / "cfdna_cluster_sizes.tsv", sep="\t")
    logging.info(f"Wrote cluster sizes")

    # Save cluster matrix
    agg.index.name = "cluster"
    out_matrix = args.output_dir / f"cfdna_cluster_matrix_{args.aggregation}.tsv.gz"
    agg.to_csv(out_matrix, sep="\t", compression="gzip", float_format="%.6g")
    logging.info(f"Wrote {out_matrix}")

    # Summary stats
    logging.info(f"Summary:")
    logging.info(f"  cfDNA peaks aggregated:  {len(cfdna)}")
    logging.info(f"  clusters in output:      {len(agg)}")
    logging.info(f"  samples in output:       {len(sample_cols)}")
    logging.info(f"  median peaks per cluster: {sizes['n_peaks'].median():.0f}")
    logging.info(f"  cluster size range:       {sizes['n_peaks'].min()} - {sizes['n_peaks'].max()}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
