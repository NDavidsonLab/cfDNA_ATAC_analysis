#!/usr/bin/env python3
"""
libnorm85_cfdna.py

Per-sample 85% library normalization of the cfDNA SE matrix.

For each sample, divide column by 85th percentile of ABSOLUTE values
(SE is bidirectional, near-zero is the meaningful midpoint).
After normalization, each sample's q85 of |SE| lands at 1.0.

Input:  cfdna_se_per_peak.tsv.gz (1.07M peaks x 43 samples)
Output: cfdna_se_per_peak_libnorm85.tsv.gz (same shape, normalized)
        libnorm85_cfdna_scaling_factors.tsv
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
    p.add_argument("--input-matrix",  required=True, type=Path)
    p.add_argument("--output-matrix", required=True, type=Path)
    p.add_argument("--scaling-factors-tsv", required=True, type=Path)
    p.add_argument("--quantile", type=float, default=85.0)
    args = p.parse_args()

    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_matrix.parent / "libnorm85_cfdna.log",
                                  mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Loading: {args.input_matrix}")
    # Read header to set per-column dtype
    header = pd.read_csv(args.input_matrix, sep="\t", nrows=0)
    sample_cols = [c for c in header.columns if c != header.columns[0]]
    dtype_dict = {c: np.float32 for c in sample_cols}
    M = pd.read_csv(args.input_matrix, sep="\t", index_col=0, dtype=dtype_dict)
    logging.info(f"  shape: {M.shape} (peaks x samples)")
    n_samples = M.shape[1]

    logging.info(f"Computing per-sample {args.quantile}th percentile of |SE| "
                 f"(non-zero only)")
    q_values = np.zeros(n_samples, dtype=np.float64)
    nonzero_counts = np.zeros(n_samples, dtype=np.int64)
    for j, col_name in enumerate(M.columns):
        col = M.iloc[:, j].values
        abs_col = np.abs(col)
        nonzero = abs_col[abs_col > 0]
        nonzero_counts[j] = len(nonzero)
        if len(nonzero) == 0:
            logging.warning(f"  {col_name}: NO non-zero |SE| values "
                            f"— scaling factor = 1.0")
            q_values[j] = 1.0
        else:
            q_values[j] = np.percentile(nonzero, args.quantile)
        if (j + 1) % 10 == 0 or j == n_samples - 1:
            logging.info(f"  processed {j+1}/{n_samples} samples")

    logging.info(f"q{args.quantile:.0f} |SE| distribution across samples:")
    logging.info(f"  min:    {q_values.min():.4f}")
    logging.info(f"  median: {np.median(q_values):.4f}")
    logging.info(f"  mean:   {q_values.mean():.4f}")
    logging.info(f"  max:    {q_values.max():.4f}")

    logging.info("Applying scaling")
    M = M.divide(q_values, axis=1)
    M = M.astype(np.float32)

    sf = pd.DataFrame({
        "sample_id":              M.columns,
        "q_value":                q_values,
        "nonzero_peak_count":     nonzero_counts,
    })
    sf.to_csv(args.scaling_factors_tsv, sep="\t", index=False)
    logging.info(f"Wrote scaling factors: {args.scaling_factors_tsv}")

    logging.info(f"Writing normalized matrix: {args.output_matrix}")
    M.to_csv(args.output_matrix, sep="\t",
             compression="gzip" if str(args.output_matrix).endswith(".gz") else None,
             float_format="%.6g")
    logging.info(f"Done.")


if __name__ == "__main__":
    main()
