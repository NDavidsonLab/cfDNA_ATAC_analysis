#!/usr/bin/env python3
"""
libnorm85_persample.py

Per-sample 85% quantile library-size normalization of the ATAC supermatrix.

For each sample (column), compute the 85th percentile of its NON-ZERO values,
then divide every value in that column by that scalar. This is the
upper-quartile-style normalization used by edgeR but with 85 instead of 75.

After normalization, each sample's 85th percentile of non-zero values lands
at 1.0. This corrects for differences in library size between samples without
being dominated by the very highest peaks the way total-sum scaling is.

The matrix is processed in chunks to keep memory usage modest (a 1.07M x 784
matrix as float32 is ~3.4 GB; we read/write in chunks to avoid duplicate
copies and to stream gzip output).

Input:  super_matrix_persample.tsv.gz (peak_id x 784 samples)
Output: super_matrix_libnorm85.tsv.gz  (same shape, values scaled per column)
        libnorm85_scaling_factors.tsv  (per-sample q85 used for division)
"""
from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-matrix",  required=True, type=Path)
    p.add_argument("--output-matrix", required=True, type=Path)
    p.add_argument("--scaling-factors-tsv", required=True, type=Path,
                   help="Write per-sample q85 scalar to this TSV")
    p.add_argument("--quantile", type=float, default=85.0,
                   help="Quantile percentage (default 85)")
    args = p.parse_args()

    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output_matrix.parent / "libnorm85.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Loading: {args.input_matrix}")
    # Load all at once — matrix is ~3.4 GB as float32, fits in 54 GB free
    # Read header to determine sample columns
    header = pd.read_csv(args.input_matrix, sep="\t", nrows=0)
    sample_cols = [c for c in header.columns if c != header.columns[0]]
    dtype_dict = {c: np.float32 for c in sample_cols}
    M = pd.read_csv(args.input_matrix, sep="\t", index_col=0, dtype=dtype_dict)
    logging.info(f"  shape: {M.shape} (peaks x samples)")
    n_samples = M.shape[1]

    # Step 1: compute per-sample 85th percentile of non-zero values
    logging.info(f"Computing per-sample {args.quantile}th percentile of "
                 f"non-zero values")
    q_values = np.zeros(n_samples, dtype=np.float64)
    nonzero_counts = np.zeros(n_samples, dtype=np.int64)
    for j, col_name in enumerate(M.columns):
        col = M.iloc[:, j].values
        nonzero = col[col > 0]
        nonzero_counts[j] = len(nonzero)
        if len(nonzero) == 0:
            logging.warning(f"  {col_name}: NO non-zero values "
                            f"— scaling factor will be 1.0 (column unchanged)")
            q_values[j] = 1.0
        else:
            q_values[j] = np.percentile(nonzero, args.quantile)
        if (j + 1) % 100 == 0 or j == n_samples - 1:
            logging.info(f"  processed {j+1}/{n_samples} samples")

    # Sanity check on q values
    logging.info(f"q{args.quantile:.0f} distribution across samples:")
    logging.info(f"  min:    {q_values.min():.4f}")
    logging.info(f"  median: {np.median(q_values):.4f}")
    logging.info(f"  mean:   {q_values.mean():.4f}")
    logging.info(f"  max:    {q_values.max():.4f}")

    # Step 2: divide each column by its q85
    logging.info("Applying scaling")
    M = M.divide(q_values, axis=1)
    M = M.astype(np.float32)

    # Step 3: write scaling factors TSV
    sf = pd.DataFrame({
        "sample_id":           M.columns,
        "q_value":             q_values,
        "nonzero_peak_count":  nonzero_counts,
    })
    sf.to_csv(args.scaling_factors_tsv, sep="\t", index=False)
    logging.info(f"Wrote scaling factors: {args.scaling_factors_tsv}")

    # Step 4: write normalized matrix
    logging.info(f"Writing normalized matrix: {args.output_matrix}")
    M.to_csv(args.output_matrix, sep="\t",
             compression="gzip" if str(args.output_matrix).endswith(".gz") else None,
             float_format="%.6g")
    logging.info(f"Done.")


if __name__ == "__main__":
    main()
