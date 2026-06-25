#!/usr/bin/env python3
"""
compute_cfdna_all_bins_per_peak.py

For the 43-sample relabeled cohort, compute:
  1. Four bin-count matrices at the UNIFIED peak level (1.07M peaks x 43):
     - Subnuc (length 1..130)
     - NUC    (length 131..200)
     - OLDN   (length 201..330)
     - DINUC  (length 331..400)
  2. Per-sample fragment BED in unified peak space (one BED per sample),
     with each (fragment, peak) overlap as one line.

Same intersect + dedupe logic as compute_cfdna_subnuc_per_peak.py but emits
all four bins AND writes the intermediate fragments-mapped-to-peaks for
the collaborator data package.

Per-sample fragment BED format:
  chrom  start  end  fragment_length  unified_peak_id  sample_id

Outputs:
  matrices_dir/
    cfdna_subnuc_per_peak.tsv.gz
    cfdna_nuc_per_peak.tsv.gz
    cfdna_oldn_per_peak.tsv.gz
    cfdna_dinuc_per_peak.tsv.gz
  fragments_dir/
    <sample>_unified_peak_fragments.bed.gz  (one per sample)
  qc_intersection_retention.tsv
  compute.log
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
NUC_MIN, NUC_MAX = 131, 200
OLDN_MIN, OLDN_MAX = 201, 330
DINUC_MIN, DINUC_MAX = 331, 400
MAX_FRAG = 400


def classify(length: int) -> str | None:
    if length > MAX_FRAG or length < 1:
        return None
    if length <= SUBNUC_MAX:
        return "Subnuc"
    if length <= NUC_MAX:
        return "NUC"
    if length <= OLDN_MAX:
        return "OLDN"
    return "DINUC"


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


def intersect_count_and_write_bed(sample_bed: Path, unified_peak_bed: Path,
                                     tmp_dir: Path, sample_id: str,
                                     out_fragment_bed: Path) -> dict:
    """
    Run bedtools intersect, count Subnuc/NUC/OLDN/DINUC fragments per peak,
    AND write a per-sample fragment BED with (chrom, start, end, frag_len,
    unified_peak_id, sample_id) one line per (fragment-length, peak-overlap).

    Returns:
      {
        "peak_counts": {peak_id: {Subnuc: N, NUC: N, OLDN: N, DINUC: N}},
        "bin_totals":  {Subnuc: int, NUC: int, OLDN: int, DINUC: int},
        "n_excluded":  int (fragments out of bin range, e.g. >400 bp)
      }
    """
    out_isect = tmp_dir / f"{sample_id}.intersect.bed"
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

    peak_counts = defaultdict(lambda: {"Subnuc": 0, "NUC": 0,
                                         "OLDN": 0, "DINUC": 0})
    bin_totals = {"Subnuc": 0, "NUC": 0, "OLDN": 0, "DINUC": 0}
    n_excluded = 0

    with gzip.open(out_fragment_bed, "wt") as fh_out, \
         open(out_isect) as fh_in:
        # No header in BED format
        for line in fh_in:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom = parts[0]; start = parts[1]; end = parts[2]
            lengths_str = parts[3]
            unified_peak = parts[7]
            try:
                lengths = [int(x) for x in lengths_str.split("-") if x]
            except ValueError:
                continue
            for L in lengths:
                b = classify(L)
                if b is None:
                    n_excluded += 1
                    continue
                peak_counts[unified_peak][b] += 1
                bin_totals[b] += 1
                fh_out.write(f"{chrom}\t{start}\t{end}\t{L}\t"
                              f"{unified_peak}\t{sample_id}\n")

    out_isect.unlink()
    return {
        "peak_counts": dict(peak_counts),
        "bin_totals":  bin_totals,
        "n_excluded":  n_excluded,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",         required=True, type=Path)
    p.add_argument("--metadata-tsv",     required=True, type=Path)
    p.add_argument("--unified-peak-bed", required=True, type=Path)
    p.add_argument("--matrices-dir",     required=True, type=Path)
    p.add_argument("--fragments-dir",    required=True, type=Path)
    p.add_argument("--output-dir",       required=True, type=Path,
                   help="For QC + log")
    args = p.parse_args()

    args.matrices_dir.mkdir(parents=True, exist_ok=True)
    args.fragments_dir.mkdir(parents=True, exist_ok=True)
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

    # Pre-allocate matrices (peaks x samples) for each bin
    subnuc_mat = np.zeros((n_peaks, len(samples)), dtype=np.float32)
    nuc_mat    = np.zeros((n_peaks, len(samples)), dtype=np.float32)
    oldn_mat   = np.zeros((n_peaks, len(samples)), dtype=np.float32)
    dinuc_mat  = np.zeros((n_peaks, len(samples)), dtype=np.float32)

    qc_rows = []
    for j, sample_id in enumerate(samples):
        logging.info(f"=== [{j+1}/{len(samples)}] {sample_id} ===")
        concat_bed = tempdir / f"{sample_id}.concat.bed"
        n_unique, n_total_before = concat_dedupe_sample_beds(
            args.data_dir, sample_id, concat_bed)
        logging.info(f"  fragments: {n_total_before} total -> {n_unique} unique")

        out_fragment_bed = args.fragments_dir / f"{sample_id}_unified_peak_fragments.bed.gz"
        result = intersect_count_and_write_bed(
            concat_bed, args.unified_peak_bed, tempdir,
            sample_id, out_fragment_bed)
        concat_bed.unlink()

        peak_counts = result["peak_counts"]
        bin_totals  = result["bin_totals"]
        n_excluded  = result["n_excluded"]
        n_peaks_with_signal = len(peak_counts)
        total_retained = sum(bin_totals.values())
        logging.info(f"  Subnuc:{bin_totals['Subnuc']}  "
                     f"NUC:{bin_totals['NUC']}  "
                     f"OLDN:{bin_totals['OLDN']}  "
                     f"DINUC:{bin_totals['DINUC']}  "
                     f"excluded:{n_excluded}")
        logging.info(f"  {n_peaks_with_signal} peaks have fragments")
        logging.info(f"  wrote {out_fragment_bed.name}")

        # Fill matrices
        for pk, bins in peak_counts.items():
            idx = peak_id_to_idx.get(pk)
            if idx is None:
                continue
            subnuc_mat[idx, j] = bins["Subnuc"]
            nuc_mat[idx, j]    = bins["NUC"]
            oldn_mat[idx, j]   = bins["OLDN"]
            dinuc_mat[idx, j]  = bins["DINUC"]

        qc_rows.append({
            "sample_id":             sample_id,
            "total_fragments_in":    n_total_before,
            "unique_fragments":      n_unique,
            "peaks_with_signal":     n_peaks_with_signal,
            "subnuc_total":          bin_totals["Subnuc"],
            "nuc_total":             bin_totals["NUC"],
            "oldn_total":            bin_totals["OLDN"],
            "dinuc_total":           bin_totals["DINUC"],
            "excluded_too_long":     n_excluded,
            "total_retained":        total_retained,
        })

    # Save QC
    pd.DataFrame(qc_rows).to_csv(
        args.output_dir / "qc_intersection_retention.tsv",
        sep="\t", index=False)
    logging.info(f"QC summary saved")

    # Save matrices
    logging.info(f"Building and writing 4 bin-count matrices")
    for name, mat in [("subnuc", subnuc_mat), ("nuc", nuc_mat),
                       ("oldn", oldn_mat), ("dinuc", dinuc_mat)]:
        df = pd.DataFrame(mat, index=all_peak_ids, columns=samples)
        df.index.name = "peak_id"
        out_path = args.matrices_dir / f"cfdna_{name}_per_peak.tsv.gz"
        df.to_csv(out_path, sep="\t", compression="gzip", float_format="%.6g")
        logging.info(f"  Wrote {out_path}")

    try:
        tempdir.rmdir()
    except OSError:
        pass
    logging.info("Done.")


if __name__ == "__main__":
    main()
