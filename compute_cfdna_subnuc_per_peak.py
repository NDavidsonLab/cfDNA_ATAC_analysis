#!/usr/bin/env python3
"""
compute_cfdna_subnuc_per_peak.py

Build cfDNA Subnuc-count matrix at the UNIFIED peak level (1.07M peaks)
for the 43-sample relabeled cohort.

For each sample:
  1. Concatenate all <sample>_bin.*.bed.gz files (37 reference peak sets)
  2. Dedupe fragments by (chrom, start, end, fragment_lengths)
  3. bedtools intersect against super_peaks_filtered.bed
  4. Per unified peak, count fragments with length <= 130 (Subnuc bin only)

This is DIFFERENT from compute_cfdna_se_per_peak.py:
  - No SE formula applied
  - Only Subnuc bin counted (fragments with length 1..130)
  - Values are non-negative integer counts, not log2 ratios

Outputs:
  - cfdna_subnuc_count_per_peak.tsv.gz   (1.07M peaks x 43 samples, float32)
  - qc_intersection_retention.tsv         (per-sample stats)
  - compute.log
"""
from __future__ import annotations

import argparse
import gzip
import logging
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SUBNUC_MAX = 130


def concat_dedupe_sample_beds(data_dir: Path, sample_id: str,
                                out_bed_path: Path) -> tuple[int, int]:
    pattern = f"{sample_id}_bin.*.bed.gz"
    files = sorted(data_dir.glob(pattern))
    seen = set()
    total_count_before = 0
    n_unique = 0
    with open(out_bed_path, "w") as out:
        for f in files:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 6:
                        continue
                    chrom, start, end = parts[0], parts[1], parts[2]
                    lengths_str = parts[4]
                    try:
                        count = int(parts[5])
                    except ValueError:
                        continue
                    total_count_before += count
                    key = (chrom, start, end, lengths_str)
                    if key in seen:
                        continue
                    seen.add(key)
                    n_unique += 1
                    out.write(f"{chrom}\t{start}\t{end}\t{lengths_str}\n")
    return n_unique, total_count_before


def intersect_and_count_subnuc(sample_bed: Path, unified_peak_bed: Path,
                                  out_dir: Path, sample_id: str) -> dict[str, int]:
    """Run bedtools intersect, count ONLY Subnuc (<=130 bp) fragments per peak."""
    out_isect = out_dir / f"{sample_id}.intersect.bed"
    cmd = [
        "bedtools", "intersect",
        "-a", str(sample_bed),
        "-b", str(unified_peak_bed),
        "-wa", "-wb",
    ]
    with open(out_isect, "w") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE,
                                  check=False)
    if result.returncode != 0:
        raise RuntimeError(f"bedtools failed: {result.stderr.decode()[:500]}")

    peak_counts = defaultdict(int)
    total_short = 0
    total_long = 0
    with open(out_isect) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            lengths_str = parts[3]
            unified_peak = parts[7]
            try:
                lengths = [int(x) for x in lengths_str.split("-") if x]
            except ValueError:
                continue
            for L in lengths:
                if 1 <= L <= SUBNUC_MAX:
                    peak_counts[unified_peak] += 1
                    total_short += 1
                else:
                    total_long += 1
    out_isect.unlink()
    return dict(peak_counts), total_short, total_long


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",         required=True, type=Path)
    p.add_argument("--metadata-tsv",     required=True, type=Path)
    p.add_argument("--unified-peak-bed", required=True, type=Path)
    p.add_argument("--output-dir",       required=True, type=Path)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "compute.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    meta = pd.read_csv(args.metadata_tsv, sep="\t")
    samples = list(meta["sample_id"])
    logging.info(f"Processing {len(samples)} samples")

    logging.info(f"Loading unified peak list")
    peaks_df = pd.read_csv(args.unified_peak_bed, sep="\t", header=None,
                            names=["chrom", "start", "end", "peak_id"])
    all_peak_ids = list(peaks_df["peak_id"])
    n_peaks = len(all_peak_ids)
    peak_id_to_idx = {pid: idx for idx, pid in enumerate(all_peak_ids)}
    logging.info(f"  {n_peaks} unified peaks")

    tempdir = args.output_dir / "_tmp"
    tempdir.mkdir(exist_ok=True)

    # Pre-allocate result matrix
    count_per_sample = {}
    qc_rows = []

    for i, sample_id in enumerate(samples):
        logging.info(f"=== [{i+1}/{len(samples)}] {sample_id} ===")
        concat_bed = tempdir / f"{sample_id}.concat.bed"
        n_unique, n_total_before = concat_dedupe_sample_beds(
            args.data_dir, sample_id, concat_bed)
        logging.info(f"  fragments: {n_total_before} total -> "
                     f"{n_unique} unique after dedupe")

        peak_counts, n_subnuc, n_long = intersect_and_count_subnuc(
            concat_bed, args.unified_peak_bed, tempdir, sample_id)
        logging.info(f"  {len(peak_counts)} peaks have Subnuc fragments")
        logging.info(f"  Subnuc fragments retained: {n_subnuc} "
                     f"(other lengths: {n_long}, "
                     f"% Subnuc: {100*n_subnuc/max(n_subnuc+n_long,1):.1f}%)")
        concat_bed.unlink()

        # Build per-peak count array
        cnt_arr = np.zeros(n_peaks, dtype=np.float32)
        for pk, c in peak_counts.items():
            idx = peak_id_to_idx.get(pk)
            if idx is not None:
                cnt_arr[idx] = c
        count_per_sample[sample_id] = cnt_arr

        qc_rows.append({
            "sample_id":                  sample_id,
            "total_fragments_before_dedupe": n_total_before,
            "unique_fragments":            n_unique,
            "peaks_with_subnuc":           len(peak_counts),
            "subnuc_fragments_retained":   n_subnuc,
            "long_fragments_retained":     n_long,
            "pct_subnuc_of_retained":      100.0 * n_subnuc / max(n_subnuc + n_long, 1),
        })

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(args.output_dir / "qc_intersection_retention.tsv",
                  sep="\t", index=False)
    logging.info(f"QC summary saved")

    logging.info(f"Building final matrix: {n_peaks} peaks x {len(samples)} samples")
    cnt_df = pd.DataFrame(count_per_sample, index=all_peak_ids)
    cnt_df.index.name = "peak_id"
    out_path = args.output_dir / "cfdna_subnuc_count_per_peak.tsv.gz"
    cnt_df.to_csv(out_path, sep="\t", compression="gzip", float_format="%.6g")
    logging.info(f"Wrote {out_path}")

    try:
        tempdir.rmdir()
    except OSError:
        pass
    logging.info("Done.")


if __name__ == "__main__":
    main()
