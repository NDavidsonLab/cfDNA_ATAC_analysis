#!/usr/bin/env python3
"""
generate_collaborator_readme.py

Walk the collaborator data package folder, gather file sizes and basic stats,
and emit:
  - README.md (Markdown, GitHub-friendly)
  - manifest.tsv (file_path, size_bytes, n_rows, description)

Reads sample metadata files to report sample counts accurately.
"""
from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path

import pandas as pd


def human_size(nbytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def count_lines_gz(path: Path) -> int:
    n = 0
    with gzip.open(path, "rt") as fh:
        for _ in fh:
            n += 1
    return n


def count_lines_plain(path: Path) -> int:
    n = 0
    with open(path) as fh:
        for _ in fh:
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--package-dir",     required=True, type=Path)
    p.add_argument("--atac-metadata",   required=True, type=Path)
    p.add_argument("--cfdna-metadata",  required=True, type=Path)
    p.add_argument("--peaks-bed",       required=True, type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                          format="%(asctime)s %(levelname)s %(message)s")

    pkg = args.package_dir
    if not pkg.exists():
        raise FileNotFoundError(f"Package dir does not exist: {pkg}")

    # Gather stats
    atac_meta = pd.read_csv(args.atac_metadata, sep="\t")
    cfdna_meta = pd.read_csv(args.cfdna_metadata, sep="\t")
    n_atac_samples = len(atac_meta)
    n_cfdna_samples = len(cfdna_meta)
    n_peaks = count_lines_plain(args.peaks_bed)
    logging.info(f"ATAC samples:  {n_atac_samples}")
    logging.info(f"cfDNA samples: {n_cfdna_samples}")
    logging.info(f"Unified peaks: {n_peaks}")

    cfdna_cancer_counts = cfdna_meta["cancer_type"].value_counts().to_dict()
    atac_cell_counts = atac_meta["cell_type"].value_counts() if "cell_type" in atac_meta.columns else None

    # Build manifest
    files_to_describe = [
        ("README.md", "This file. Description of package contents."),
        ("manifest.tsv", "Machine-readable file inventory."),
        ("peaks/super_peaks_filtered.bed",
         f"Unified peak set ({n_peaks} peaks). Columns: chrom, start, end, peak_id. "
         "Used as the row index for all matrices below."),
        ("atac/atac_matrix_per_sample.tsv.gz",
         f"ATAC accessibility matrix ({n_peaks} peaks x {n_atac_samples} samples). "
         "Raw values pre-normalization. First column = peak_id; remaining columns "
         "= sample IDs. No log, no z-score, no library normalization applied."),
        ("atac/atac_metadata.tsv",
         f"Per-sample annotations for ATAC ({n_atac_samples} rows). "
         "Columns include sample_id, cell_type, source. Use to subset/group samples."),
        ("cfdna/matrices/cfdna_subnuc_per_peak.tsv.gz",
         f"cfDNA fragment counts per peak per sample, Subnuc bin "
         f"(fragment length 1-130 bp). Shape: {n_peaks} peaks x {n_cfdna_samples} samples. "
         "Raw counts, no normalization."),
        ("cfdna/matrices/cfdna_nuc_per_peak.tsv.gz",
         f"cfDNA NUC bin (131-200 bp). Shape: {n_peaks} x {n_cfdna_samples}."),
        ("cfdna/matrices/cfdna_oldn_per_peak.tsv.gz",
         f"cfDNA OLDN bin (201-330 bp). Shape: {n_peaks} x {n_cfdna_samples}."),
        ("cfdna/matrices/cfdna_dinuc_per_peak.tsv.gz",
         f"cfDNA DINUC bin (331-400 bp). Shape: {n_peaks} x {n_cfdna_samples}."),
        ("cfdna/cfdna_metadata.tsv",
         f"Per-sample annotations for cfDNA ({n_cfdna_samples} rows). "
         "Columns: sample_id, cancer_type."),
    ]
    manifest_rows = []
    for rel, desc in files_to_describe:
        p = pkg / rel
        if p.exists():
            manifest_rows.append({
                "file":         rel,
                "size_bytes":   p.stat().st_size,
                "size_human":   human_size(p.stat().st_size),
                "description":  desc,
            })
        else:
            manifest_rows.append({
                "file":         rel,
                "size_bytes":   0,
                "size_human":   "MISSING",
                "description":  desc,
            })

    fragments_dir = pkg / "cfdna" / "fragments"
    if fragments_dir.exists():
        for f in sorted(fragments_dir.glob("*.bed.gz")):
            sample = f.name.replace("_unified_peak_fragments.bed.gz", "")
            manifest_rows.append({
                "file":         f"cfdna/fragments/{f.name}",
                "size_bytes":   f.stat().st_size,
                "size_human":   human_size(f.stat().st_size),
                "description":  f"Per-fragment BED for cfDNA sample {sample}, "
                                "intersected with unified peaks. Columns: chrom, "
                                "start, end, fragment_length, unified_peak_id, "
                                "sample_id. One line per fragment-peak overlap.",
            })

    pd.DataFrame(manifest_rows).to_csv(pkg / "manifest.tsv",
                                          sep="\t", index=False)
    logging.info(f"Wrote manifest.tsv with {len(manifest_rows)} entries")

    total_size = sum(r["size_bytes"] for r in manifest_rows)
    logging.info(f"Total package size: {human_size(total_size)}")

    # =============================================
    # Build README.md
    # =============================================
    cancer_breakdown_lines = []
    for ct, n in sorted(cfdna_cancer_counts.items(),
                          key=lambda x: -x[1]):
        cancer_breakdown_lines.append(f"  - **{ct}**: {n}")
    cancer_breakdown = "\n".join(cancer_breakdown_lines)

    cell_breakdown = ""
    if atac_cell_counts is not None:
        lines = []
        for ct, n in atac_cell_counts.items():
            lines.append(f"  - **{ct}**: {n}")
        cell_breakdown = "\n".join(lines)
    else:
        cell_breakdown = "  - (cell type column not found in metadata file)"

    n_fragment_beds = len(list(fragments_dir.glob("*.bed.gz"))) if fragments_dir.exists() else 0

    readme_text = f"""# Collaborator Data Package

ATAC and cfDNA data on a unified peak set, with no normalization applied.
Suitable for downstream GTF-based annotation (promoter regions, gene bodies,
distal regulatory elements) and custom analyses.

**Package size**: {human_size(total_size)}
**Generated**: programmatically from `generate_collaborator_readme.py`

---

## What you get

### Unified peak set

`peaks/super_peaks_filtered.bed` — {n_peaks:,} peaks across the genome,
unified across:
- TCGA ATAC reference cohorts (cancer-type peak sets)
- AML ATAC samples (15 AML patients)
- Blood ATAC reference (15 hematopoietic samples)
- cfDNA cohort peak sets

**File format**: 4-column BED (no header)
```
chrom  start  end  peak_id
chr1   10050  10550  peak_00000001
chr1   180000 180500 peak_00000002
...
```

Use this as the input to `bedtools intersect` against a GTF to map peaks
to genes / promoters / distal regions.

### ATAC matrix

`atac/atac_matrix_per_sample.tsv.gz` — accessibility values for
{n_peaks:,} peaks x {n_atac_samples} ATAC samples.

**File format**: tab-separated, gzipped
- First column: `peak_id` (matches the peak_id in `super_peaks_filtered.bed`)
- Remaining {n_atac_samples} columns: one per ATAC sample

**Important**: These are RAW values, with NO downstream processing applied
(no library normalization, no log transform, no z-score). If you want to
compare across samples, you will likely want to library-normalize first.

`atac/atac_metadata.tsv` — per-sample annotations for the {n_atac_samples}
ATAC samples. Use the `sample_id` column to match column headers in the
matrix.

### cfDNA matrices

`cfdna/matrices/` contains four count matrices, one per fragment length bin:

| File | Length range | Biology |
|------|--------------|---------|
| `cfdna_subnuc_per_peak.tsv.gz` | 1-130 bp | Sub-nucleosomal — often enriched in tumor-derived cfDNA |
| `cfdna_nuc_per_peak.tsv.gz`    | 131-200 bp | Mono-nucleosomal — dominant fraction in plasma |
| `cfdna_oldn_per_peak.tsv.gz`   | 201-330 bp | Long-OLD nucleosomal |
| `cfdna_dinuc_per_peak.tsv.gz`  | 331-400 bp | Di-nucleosomal |

**File format**: tab-separated, gzipped
- First column: `peak_id` (matches the peak_id in `super_peaks_filtered.bed`)
- Remaining {n_cfdna_samples} columns: one per cfDNA sample
- Values: integer counts (number of fragments in that length bin overlapping
  that peak in that sample). Stored as float for compatibility but values
  are whole numbers.

**Important**: These are raw counts, no normalization. Counts vary substantially
across samples due to library size differences — normalize before any
cross-sample comparison.

### cfDNA fragment BEDs

`cfdna/fragments/` contains {n_fragment_beds} per-sample BED files of fragments
mapped to unified peaks.

**File format**: 6-column BED (no header), gzipped
```
chrom  start  end  fragment_length  unified_peak_id  sample_id
```

Each line represents one (fragment, peak) overlap. A single fragment that
spans two adjacent unified peaks will appear on two lines. Use these if
you want fragment-level resolution beyond the bin-aggregated matrices.

### Metadata

`cfdna/cfdna_metadata.tsv` — per-sample cancer type labels.

**Columns**: `sample_id`, `cancer_type`

**Cancer type breakdown** (n={n_cfdna_samples} samples):
{cancer_breakdown}

`atac/atac_metadata.tsv` — per-sample cell type labels.

**Cell type breakdown** (n={n_atac_samples} samples):
{cell_breakdown}

---

## Suggested workflow

### Map peaks to gene promoters

```bash
# Annotate peaks with the nearest TSS / overlapping promoter
# (example using bedtools)
bedtools intersect -a peaks/super_peaks_filtered.bed \\
    -b your_gtf_promoters.bed \\
    -wa -wb > peaks_with_promoters.tsv
```

### Load matrices in Python

```python
import pandas as pd
atac = pd.read_csv("atac/atac_matrix_per_sample.tsv.gz",
                    sep="\\t", index_col=0)
# atac.index = peak_id, atac.columns = sample_id
print(atac.shape)  # ({n_peaks:,}, {n_atac_samples})

cfdna_subnuc = pd.read_csv("cfdna/matrices/cfdna_subnuc_per_peak.tsv.gz",
                              sep="\\t", index_col=0)
print(cfdna_subnuc.shape)  # ({n_peaks:,}, {n_cfdna_samples})
```

### Join peak coordinates with matrix values

```python
import pandas as pd
peaks = pd.read_csv("peaks/super_peaks_filtered.bed",
                      sep="\\t", header=None,
                      names=["chrom", "start", "end", "peak_id"])
atac = pd.read_csv("atac/atac_matrix_per_sample.tsv.gz",
                     sep="\\t", index_col=0)
joined = peaks.merge(atac, left_on="peak_id", right_index=True)
```

---

## Processing notes (what's been done)

- **Peak unification**: peaks from different cohorts (TCGA, AML, blood, cfDNA)
  were intersected and merged into a single unified set.
- **cfDNA fragment deduplication**: in the source data, each cfDNA sample was
  scored against multiple reference peak sets (one per cell type / cancer
  type). The same fragment can appear in multiple files. We deduplicated by
  (chrom, start, end, fragment_lengths) before counting.
- **Per-peak fragment binning**: fragments were assigned to a length bin
  (Subnuc/NUC/OLDN/DINUC) and counted per peak.

## Processing notes (what has NOT been done)

- **No normalization**: values are raw counts (cfDNA) or raw signal (ATAC).
  Library size / sequencing depth differences are not corrected.
- **No log/transform/scaling**: values are reported as-is.
- **No SE (Subnuc-Enrichment) transformation**: the cfDNA data is provided as
  separate bin counts, not as a log2 ratio.
- **No clustering / dimensionality reduction**: matrices are full-resolution
  at the peak level.
- **No filtering of low-signal peaks or samples**: all peaks and samples are
  retained as-is.

---

## Questions

Feel free to reach out if anything is unclear.
"""

    (pkg / "README.md").write_text(readme_text)
    logging.info(f"Wrote README.md ({len(readme_text)} chars)")
    logging.info("Done.")


if __name__ == "__main__":
    main()
