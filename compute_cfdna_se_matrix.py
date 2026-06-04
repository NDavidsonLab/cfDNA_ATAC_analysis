#!/usr/bin/env python3
"""
compute_cfdna_se_matrix.py  (v2: configurable file pattern)

Build cfDNA per-cluster Submucosal Enrichment (SE) matrix from per-sample
cfDNA fragment-length BED files.

This version adds two flags for working with non-IC datasets (like the
healthy GSE171434 cohort):
  --file-glob       Glob pattern for finding sample BED files
                    (default "IC*_super_peaks_filtered_dummy.bed.gz")
  --name-pattern    Regex for extracting sample_id from filename
                    (default r"^(IC\\d+)_")

For the healthy cohort (BC02_1, BC02_2, ..., F02, F05, SporeA2, ...):
  --file-glob       "*_super_peaks_filtered_dummy.bed.gz"
  --name-pattern    "^([A-Za-z0-9]+(?:_[0-9]+)?)_super_peaks"

Fragment-length bins (whiteboard, configurable NUC range):
  Subnuc   0   - 130 bp
  GAP      131 - (nuc_min-1)  (excluded; empty if nuc_min == 131)
  NUC      nuc_min - nuc_max  (default 151-200 LEGACY; pass 131 for corrected)
  OLDN     (nuc_max+1) - 330 bp
  DINUC    331 - 400 bp
  REMOVE   > 400 bp

SE = log2((Subnuc + OLDN + 1) / (NUC + DINUC + 1))

Inputs:
  --cfdna-dir       directory of *.bed.gz files
  --clusters-tsv    peak_id -> cluster mapping
  --sample-map      sample_map.tsv (cancer_type, sample_id, has_peaks)
  --output-dir      where to write outputs
  --nuc-min         lower bound of NUC bin (default 151; pass 131 for corrected)
  --nuc-max         upper bound of NUC bin (default 200)
  --file-glob       glob to find sample BED files (see above)
  --name-pattern    regex to extract sample_id from filename (see above)

Outputs:
  cfdna_se_matrix.tsv.gz       -- clusters x samples, SE values
  cfdna_bincount_*.tsv.gz      -- raw per-bin counts (Subnuc/NUC/OLDN/DINUC)
  cfdna_sample_metadata.tsv    -- sample_id -> cancer_type
  cfdna_se.log
  _run_config.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


MAX_FRAG_LEN = 400
SUBNUC_MAX = 130
OLDN_MAX = 330
DINUC_MAX = 400

DEFAULT_FILE_GLOB = "IC*_super_peaks_filtered_dummy.bed.gz"
DEFAULT_NAME_PATTERN = r"^(IC\d+)_"


def build_bin_defs(nuc_min: int, nuc_max: int) -> list[tuple[str, int, int]]:
    if not (131 <= nuc_min <= 200):
        sys.exit(f"ERROR: --nuc-min must be in [131, 200], got {nuc_min}")
    if not (nuc_min <= nuc_max <= 200):
        sys.exit(f"ERROR: --nuc-max must satisfy {nuc_min} <= nuc-max <= 200, "
                 f"got {nuc_max}")
    oldn_min = nuc_max + 1
    if oldn_min > OLDN_MAX:
        sys.exit(f"ERROR: nuc-max={nuc_max} leaves no room for OLDN bin")
    return [
        ("Subnuc", 0,        SUBNUC_MAX),
        ("NUC",    nuc_min,  nuc_max),
        ("OLDN",   oldn_min, OLDN_MAX),
        ("DINUC",  331,      DINUC_MAX),
    ]


def make_classifier(bin_defs):
    def classify(length):
        if length > MAX_FRAG_LEN:
            return None
        for name, lo, hi in bin_defs:
            if lo <= length <= hi:
                return name
        return None
    return classify


def process_one_cfdna_file(bed_path, peak_to_cluster, bin_defs, classify_fn):
    cluster_bins = {name: defaultdict(int) for name, _, _ in bin_defs}
    n_rows = n_peaks_in_cluster = n_frags_total = n_frags_kept = 0

    with gzip.open(bed_path, "rt") as fh:
        for line in fh:
            n_rows += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            peak_id = parts[3]
            cluster = peak_to_cluster.get(peak_id)
            if cluster is None:
                continue
            n_peaks_in_cluster += 1
            try:
                lengths = [int(x) for x in parts[4].split("-") if x]
            except ValueError:
                continue
            n_frags_total += len(lengths)
            for L in lengths:
                bin_name = classify_fn(L)
                if bin_name is None:
                    continue
                cluster_bins[bin_name][cluster] += 1
                n_frags_kept += 1
    return cluster_bins, {
        "rows": n_rows,
        "peaks_in_cluster": n_peaks_in_cluster,
        "frags_total": n_frags_total,
        "frags_kept": n_frags_kept,
        "frags_dropped": n_frags_total - n_frags_kept,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfdna-dir",    required=True, type=Path)
    p.add_argument("--clusters-tsv", required=True, type=Path)
    p.add_argument("--sample-map",   required=True, type=Path)
    p.add_argument("--output-dir",   required=True, type=Path)
    p.add_argument("--nuc-min", type=int, default=151)
    p.add_argument("--nuc-max", type=int, default=200)
    p.add_argument("--file-glob", type=str, default=DEFAULT_FILE_GLOB,
                   help=f"Glob to find sample BED files (default: {DEFAULT_FILE_GLOB})")
    p.add_argument("--name-pattern", type=str, default=DEFAULT_NAME_PATTERN,
                   help="Regex with one capture group for sample_id "
                        f"(default: {DEFAULT_NAME_PATTERN})")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.output_dir / "cfdna_se.log", mode="w"),
                  logging.StreamHandler()],
    )

    bin_defs = build_bin_defs(args.nuc_min, args.nuc_max)
    classify_fn = make_classifier(bin_defs)
    gap_present = args.nuc_min > 131
    gap_range = f"131-{args.nuc_min - 1}" if gap_present else "none"

    logging.info("=" * 70)
    logging.info(f"NUC range: {args.nuc_min}-{args.nuc_max} bp  (GAP: {gap_range})")
    logging.info(f"  Convention: {'WHITEBOARD-CORRECTED (no gap)' if not gap_present else 'LEGACY (with gap)'}")
    logging.info(f"File glob:    {args.file_glob}")
    logging.info(f"Name pattern: {args.name_pattern}")
    logging.info("=" * 70)
    for name, lo, hi in bin_defs:
        logging.info(f"  {name:<7s} {lo:>4d} - {hi:>4d} bp")

    # Load clusters
    logging.info(f"Loading clusters: {args.clusters_tsv}")
    clusters_df = pd.read_csv(args.clusters_tsv, sep="\t", index_col=0)
    if "cluster" not in clusters_df.columns:
        sys.exit("ERROR: 'cluster' column missing from clusters TSV")
    peak_to_cluster = clusters_df["cluster"].to_dict()
    all_clusters = sorted(set(peak_to_cluster.values()))
    n_clusters = len(all_clusters)
    logging.info(f"  {len(peak_to_cluster):,} peaks across {n_clusters} clusters")

    # Load sample map
    logging.info(f"Loading sample map: {args.sample_map}")
    sample_map = pd.read_csv(args.sample_map, sep="\t")
    sample_map = sample_map[sample_map["has_peaks"] == "yes"].copy()
    logging.info(f"  {len(sample_map)} samples with has_peaks=yes")

    # Find files
    name_re = re.compile(args.name_pattern)
    cfdna_files = sorted(args.cfdna_dir.glob(args.file_glob))
    logging.info(f"  {len(cfdna_files)} files match glob {args.file_glob}")
    if len(cfdna_files) == 0:
        sys.exit(f"ERROR: no files found in {args.cfdna_dir} matching {args.file_glob}")

    file_sample_ids = []
    for f in cfdna_files:
        m = name_re.search(f.name)
        if m:
            file_sample_ids.append(m.group(1))
        else:
            file_sample_ids.append(None)
    valid_sample_ids = [s for s in file_sample_ids if s is not None]
    logging.info(f"  extracted {len(valid_sample_ids)} sample IDs from filenames")

    map_samples = set(sample_map["sample_id"])
    on_disk_samples = set(valid_sample_ids)
    usable_samples = sorted(map_samples & on_disk_samples)
    only_in_map = map_samples - on_disk_samples
    only_on_disk = on_disk_samples - map_samples
    logging.info(f"  Usable (in both): {len(usable_samples)}")
    if only_in_map:
        logging.warning(f"  In map but not on disk: {sorted(only_in_map)}")
    if only_on_disk:
        logging.warning(f"  On disk but not in map: {sorted(only_on_disk)[:10]}{'...' if len(only_on_disk) > 10 else ''}")

    sample_to_file = {}
    for f in cfdna_files:
        m = name_re.search(f.name)
        if m and m.group(1) in usable_samples:
            sample_to_file[m.group(1)] = f
    if not sample_to_file:
        sys.exit("ERROR: no usable cfDNA files. Check --file-glob / --name-pattern.")

    samples_ordered = sorted(sample_to_file.keys())
    bin_matrices = {name: np.zeros((n_clusters, len(samples_ordered)), dtype=np.int64)
                     for name, _, _ in bin_defs}
    cluster_to_idx = {c: i for i, c in enumerate(all_clusters)}

    for j, sample_id in enumerate(samples_ordered):
        bed_path = sample_to_file[sample_id]
        logging.info(f"[{j+1}/{len(samples_ordered)}] {sample_id}: {bed_path.name}")
        cluster_bins, stats = process_one_cfdna_file(
            bed_path, peak_to_cluster, bin_defs, classify_fn
        )
        for bin_name, counts in cluster_bins.items():
            for c, n in counts.items():
                bin_matrices[bin_name][cluster_to_idx[c], j] += n
        logging.info(f"  rows={stats['rows']:,}  "
                     f"peaks_in_cluster={stats['peaks_in_cluster']:,}  "
                     f"frags_total={stats['frags_total']:,}  "
                     f"frags_kept={stats['frags_kept']:,}  "
                     f"dropped={stats['frags_dropped']:,}")

    # Write bin counts
    logging.info("Writing bin-count matrices")
    for bin_name, mat in bin_matrices.items():
        df = pd.DataFrame(mat, index=pd.Index(all_clusters, name="cluster"),
                          columns=samples_ordered)
        out = args.output_dir / f"cfdna_bincount_{bin_name.lower()}.tsv.gz"
        df.to_csv(out, sep="\t", compression="gzip")
        logging.info(f"  wrote {out}")

    # SE matrix
    logging.info("Computing SE matrix: log2((Subnuc + OLDN + 1) / (NUC + DINUC + 1))")
    num = bin_matrices["Subnuc"] + bin_matrices["OLDN"] + 1
    den = bin_matrices["NUC"]    + bin_matrices["DINUC"] + 1
    se = np.log2(num / den)
    se_df = pd.DataFrame(se, index=pd.Index(all_clusters, name="cluster"),
                         columns=samples_ordered)
    se_path = args.output_dir / "cfdna_se_matrix.tsv.gz"
    se_df.to_csv(se_path, sep="\t", compression="gzip")
    logging.info(f"  wrote {se_path} ({se_df.shape[0]} x {se_df.shape[1]})")
    logging.info(f"  value range: {se_df.values.min():.3f} to {se_df.values.max():.3f}")
    logging.info(f"  median: {np.median(se_df.values):.3f}")

    # Metadata
    meta = sample_map[sample_map["sample_id"].isin(samples_ordered)][["sample_id", "cancer_type"]]
    meta = meta.set_index("sample_id").loc[samples_ordered].reset_index()
    meta_path = args.output_dir / "cfdna_sample_metadata.tsv"
    meta.to_csv(meta_path, sep="\t", index=False)
    logging.info(f"  wrote {meta_path}")
    logging.info(f"  cancer-type counts: {meta['cancer_type'].value_counts().to_dict()}")

    # Run config
    run_config = {
        "nuc_range": {
            "min_bp": args.nuc_min, "max_bp": args.nuc_max,
            "convention": "whiteboard_corrected_no_gap" if not gap_present else "legacy_with_gap",
            "gap_range": gap_range,
        },
        "file_glob": args.file_glob,
        "name_pattern": args.name_pattern,
        "bin_defs": [{"name": n, "min_bp": lo, "max_bp": hi} for n, lo, hi in bin_defs],
        "max_frag_len": MAX_FRAG_LEN,
        "inputs": {
            "cfdna_dir": str(args.cfdna_dir),
            "clusters_tsv": str(args.clusters_tsv),
            "sample_map": str(args.sample_map),
        },
        "outputs": {
            "n_clusters": n_clusters,
            "n_samples": len(samples_ordered),
            "se_matrix_shape": list(se_df.shape),
        },
        "samples_processed": samples_ordered,
        "samples_only_in_map": sorted(only_in_map),
        "samples_only_on_disk": sorted(only_on_disk),
    }
    with open(args.output_dir / "_run_config.json", "w") as fh:
        json.dump(run_config, fh, indent=2)

    logging.info("Done.")


if __name__ == "__main__":
    main()
