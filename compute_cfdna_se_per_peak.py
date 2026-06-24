#!/usr/bin/env python3
"""
compute_cfdna_se_per_peak.py

Build cfDNA SE matrix at the UNIFIED peak level (1.07M peaks) for the
43-sample relabeled cohort.

For each sample:
  1. Concatenate all <sample>_bin.*.bed.gz files (37 reference peak sets each)
  2. bedtools intersect against super_peaks_filtered.bed (1.07M unified peaks)
  3. Per unified peak, count Subnuc/NUC/OLDN/DINUC fragments
  4. Compute SE = log2((Subnuc + OLDN + 1) / (NUC + DINUC + 1))

Important: many fragments appear in multiple reference peak sets per sample
(if their peaks overlap multiple references). We deduplicate by
(chrom, start, end, fragment_lengths) before counting to avoid inflating
fragment counts.

Outputs:
  - cfdna_se_per_peak.tsv.gz   (1.07M peaks x 43 samples)
  - qc_intersection_retention.tsv  (per-sample retention stats)
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
NUC_MIN, NUC_MAX = 131, 200
OLDN_MIN, OLDN_MAX = 201, 330
DINUC_MIN, DINUC_MAX = 331, 400
MAX_FRAG = 400


def classify(length: int) -> str | None:
    if length > MAX_FRAG:
        return None
    if length <= SUBNUC_MAX:
        return "Subnuc"
    if length <= NUC_MAX:
        return "NUC"
    if length <= OLDN_MAX:
        return "OLDN"
    if length <= DINUC_MAX:
        return "DINUC"
    return None


def concat_dedupe_sample_beds(data_dir: Path, sample_id: str,
                                out_bed_path: Path) -> tuple[int, int]:
    """Concatenate all BED files for one sample, dedupe by genomic coord +
    fragment_lengths, write 4-column (chrom, start, end, lengths) BED.

    Returns (n_unique_fragments_after_dedupe, total_count_before_dedupe).
    """
    pattern = f"{sample_id}_bin.*.bed.gz"
    files = sorted(data_dir.glob(pattern))
    # Set of (chrom, start, end, lengths_str) to dedupe across reference peak sets
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


def intersect_and_count(sample_bed: Path, unified_peak_bed: Path,
                          out_dir: Path, sample_id: str) -> dict[str, dict[str, int]]:
    """Run bedtools intersect, per unified peak aggregate length bins."""
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

    peak_bins = defaultdict(lambda: defaultdict(int))
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
                b = classify(L)
                if b is not None:
                    peak_bins[unified_peak][b] += 1
    out_isect.unlink()
    return dict(peak_bins)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",         required=True, type=Path,
                   help="Dir containing <sample>_bin.<ref>.bed.gz files")
    p.add_argument("--metadata-tsv",     required=True, type=Path,
                   help="sample_id + cancer_type metadata for 43 samples")
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

    # Load sample list
    meta = pd.read_csv(args.metadata_tsv, sep="\t")
    samples = list(meta["sample_id"])
    logging.info(f"Processing {len(samples)} samples")

    # Load unified peak list
    logging.info(f"Loading unified peak list")
    peaks_df = pd.read_csv(args.unified_peak_bed, sep="\t", header=None,
                            names=["chrom", "start", "end", "peak_id"])
    all_peak_ids = list(peaks_df["peak_id"])
    logging.info(f"  {len(all_peak_ids)} unified peaks")

    tempdir = args.output_dir / "_tmp"
    tempdir.mkdir(exist_ok=True)

    # Per-sample: concat -> dedupe -> intersect -> bin -> SE
    se_per_sample = {}
    qc_rows = []

    for i, sample_id in enumerate(samples):
        logging.info(f"=== [{i+1}/{len(samples)}] {sample_id} ===")
        concat_bed = tempdir / f"{sample_id}.concat.bed"
        n_unique, n_total_before = concat_dedupe_sample_beds(
            args.data_dir, sample_id, concat_bed)
        logging.info(f"  fragments: {n_total_before} total -> "
                     f"{n_unique} unique after dedupe across references")

        peak_bins = intersect_and_count(concat_bed, args.unified_peak_bed,
                                          tempdir, sample_id)
        logging.info(f"  {len(peak_bins)} unified peaks have coverage")
        concat_bed.unlink()

        # Compute SE per peak
        se_arr = np.zeros(len(all_peak_ids), dtype=np.float32)
        # Need fast peak_id -> idx lookup
        # First time: build a Series indexed map
        if i == 0:
            peak_id_to_idx = {pid: idx for idx, pid in enumerate(all_peak_ids)}
        sum_subnuc = sum_nuc = sum_oldn = sum_dinuc = 0
        for pk, bins in peak_bins.items():
            idx = peak_id_to_idx.get(pk)
            if idx is None:
                continue
            subnuc = bins.get("Subnuc", 0)
            nuc    = bins.get("NUC", 0)
            oldn   = bins.get("OLDN", 0)
            dinuc  = bins.get("DINUC", 0)
            sum_subnuc += subnuc; sum_nuc += nuc
            sum_oldn += oldn; sum_dinuc += dinuc
            se_arr[idx] = np.log2((subnuc + oldn + 1) / (nuc + dinuc + 1))
        se_per_sample[sample_id] = se_arr
        retained = sum_subnuc + sum_nuc + sum_oldn + sum_dinuc
        qc_rows.append({
            "sample_id":                  sample_id,
            "total_fragments_before_dedupe": n_total_before,
            "unique_fragments":            n_unique,
            "unified_peaks_with_coverage": len(peak_bins),
            "retained_fragments":          retained,
            "pct_retained_vs_unique":      100.0 * retained / max(n_unique, 1),
        })
        logging.info(f"  retention vs unique: "
                     f"{100*retained/max(n_unique,1):.1f}%")

    # Save QC summary
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(args.output_dir / "qc_intersection_retention.tsv",
                  sep="\t", index=False)
    logging.info(f"QC summary saved")

    # Build final matrix and save
    logging.info(f"Building final cfDNA SE matrix: "
                 f"{len(all_peak_ids)} peaks x {len(samples)} samples")
    se_df = pd.DataFrame(se_per_sample, index=all_peak_ids)
    se_df.index.name = "peak_id"
    out_path = args.output_dir / "cfdna_se_per_peak.tsv.gz"
    se_df.to_csv(out_path, sep="\t", compression="gzip", float_format="%.6g")
    logging.info(f"Wrote {out_path}")

    # Cleanup
    try:
        tempdir.rmdir()
    except OSError:
        pass
    logging.info("Done.")


if __name__ == "__main__":
    main()
