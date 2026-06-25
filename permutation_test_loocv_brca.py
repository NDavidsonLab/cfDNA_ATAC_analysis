#!/usr/bin/env python3
"""
permutation_test_loocv_brca.py

Permutation test for the LOOCV L1 LogReg AUC on BRCA vs not-BRCA
prediction from k=1000 cfDNA cluster features.

Procedure:
  1. Compute the observed LR L1 LOOCV AUC on real labels (~30 sec)
  2. Shuffle binary labels (BRCA/not-BRCA), keep features fixed
  3. Re-run LOOCV LR L1 on shuffled labels, record shuffled AUC
  4. Repeat n_permutations times
  5. p-value = fraction of shuffled AUCs >= observed AUC

If p < 0.05, the observed signal is unlikely to be explained by random
labels — there's real (if weak) signal.
If p >= 0.05, the observed AUC is consistent with random chance.

Outputs:
  - permutation_results.tsv     observed AUC, p-value, null distribution stats
  - permutation_distribution.png   histogram of null AUCs with observed marked
  - permutation.log
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd


def run_loocv_lr(X, y, C=1.0, seed=42):
    """Run LOOCV with L1 LogReg + standardization. Return per-sample
    predicted probabilities for class 1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()
    proba = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        if len(set(y_train)) < 2:
            # Degenerate: training set has only one class
            proba[test_idx[0]] = 0.5
            continue
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        lr = LogisticRegression(
            penalty="l1", C=C, solver="liblinear",
            class_weight="balanced",
            random_state=seed, max_iter=2000,
        )
        lr.fit(X_train_s, y_train)
        proba[test_idx[0]] = lr.predict_proba(X_test_s)[0, 1]
    return proba


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfdna-cluster-matrix", required=True, type=Path)
    p.add_argument("--cfdna-metadata",       required=True, type=Path)
    p.add_argument("--positive-class",       default="BRCA")
    p.add_argument("--n-permutations",       type=int, default=100)
    p.add_argument("--lr-C",                 type=float, default=1.0)
    p.add_argument("--seed",                 type=int, default=42)
    p.add_argument("--output-dir",           required=True, type=Path)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "permutation.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Loading cfDNA cluster matrix: {args.cfdna_cluster_matrix}")
    cfdna = pd.read_csv(args.cfdna_cluster_matrix, sep="\t", index_col=0)
    logging.info(f"  shape: {cfdna.shape} (clusters x samples)")

    meta = pd.read_csv(args.cfdna_metadata, sep="\t")
    sample_to_cancer = dict(zip(meta["sample_id"], meta["cancer_type"]))

    X_df = cfdna.T
    X_df = X_df.loc[[s for s in X_df.index if s in sample_to_cancer]]
    y_str = np.array([sample_to_cancer[s] for s in X_df.index])
    y = (y_str == args.positive_class).astype(int)
    X = X_df.values.astype(np.float64)

    logging.info(f"Samples: {len(y)}, features: {X.shape[1]}")
    logging.info(f"Positive class '{args.positive_class}': {y.sum()} samples")
    logging.info(f"Negative class: {(1-y).sum()} samples")

    from sklearn.metrics import roc_auc_score

    # ============== Step 1: Observed AUC ==============
    logging.info("Step 1: computing observed LOOCV AUC on real labels")
    t0 = time.time()
    proba_obs = run_loocv_lr(X, y, C=args.lr_C, seed=args.seed)
    auc_obs = roc_auc_score(y, proba_obs)
    t1 = time.time()
    logging.info(f"  observed AUC: {auc_obs:.4f}  ({t1-t0:.1f} sec)")

    per_fold_estimate = (t1 - t0)
    total_est = per_fold_estimate * args.n_permutations
    logging.info(f"  estimated time for {args.n_permutations} permutations: "
                 f"{total_est:.0f} sec = {total_est/60:.1f} min")

    # ============== Step 2: Permutations ==============
    logging.info(f"Step 2: running {args.n_permutations} permutations")
    rng = np.random.default_rng(args.seed)
    null_aucs = np.zeros(args.n_permutations, dtype=float)
    n_warnings = 0
    for i in range(args.n_permutations):
        y_perm = rng.permutation(y)
        # Edge case: rare, but if permutation puts all positives or all
        # negatives on one side, AUC will be undefined. Re-permute.
        retry = 0
        while (y_perm.sum() == 0 or y_perm.sum() == len(y_perm)) and retry < 10:
            y_perm = rng.permutation(y)
            retry += 1
        try:
            proba_perm = run_loocv_lr(X, y_perm, C=args.lr_C, seed=args.seed)
            null_aucs[i] = roc_auc_score(y_perm, proba_perm)
        except Exception as e:
            n_warnings += 1
            null_aucs[i] = 0.5  # neutral fallback
            logging.warning(f"  permutation {i+1}: {type(e).__name__}: "
                            f"{str(e)[:80]}")
        if (i + 1) % 10 == 0 or i == args.n_permutations - 1:
            elapsed = time.time() - t1
            eta = elapsed / (i+1) * (args.n_permutations - i - 1)
            logging.info(f"  permutation {i+1}/{args.n_permutations}, "
                         f"AUC={null_aucs[i]:.3f}, elapsed={elapsed:.0f}s, "
                         f"ETA={eta:.0f}s")
    if n_warnings:
        logging.warning(f"{n_warnings} permutations had warnings")

    # ============== Step 3: p-value ==============
    # p = fraction of null AUCs >= observed AUC
    # Add 1 to numerator and denominator for biased estimator (Phipson & Smyth)
    n_ge_obs = (null_aucs >= auc_obs).sum()
    p_value = (n_ge_obs + 1) / (args.n_permutations + 1)
    logging.info("")
    logging.info("=" * 60)
    logging.info("PERMUTATION TEST RESULTS")
    logging.info("=" * 60)
    logging.info(f"Observed AUC:              {auc_obs:.4f}")
    logging.info(f"N permutations:            {args.n_permutations}")
    logging.info(f"Null distribution:")
    logging.info(f"  mean:                    {null_aucs.mean():.4f}")
    logging.info(f"  median:                  {np.median(null_aucs):.4f}")
    logging.info(f"  std:                     {null_aucs.std():.4f}")
    logging.info(f"  min:                     {null_aucs.min():.4f}")
    logging.info(f"  max:                     {null_aucs.max():.4f}")
    logging.info(f"  95th percentile:         {np.percentile(null_aucs, 95):.4f}")
    logging.info(f"  99th percentile:         {np.percentile(null_aucs, 99):.4f}")
    logging.info(f"N permutations >= observed: {n_ge_obs}")
    logging.info(f"p-value:                    {p_value:.4f}")
    logging.info("")
    if p_value < 0.01:
        logging.info(f"  *** Strong evidence against null (p < 0.01) ***")
    elif p_value < 0.05:
        logging.info(f"  * Some evidence against null (p < 0.05) *")
    elif p_value < 0.1:
        logging.info(f"  Weak evidence against null (p < 0.1)")
    else:
        logging.info(f"  No evidence against null (p >= 0.1) — AUC consistent "
                     f"with random labels")

    # Save numeric results
    results_df = pd.DataFrame([{
        "observed_auc":           auc_obs,
        "n_permutations":         args.n_permutations,
        "null_mean":              null_aucs.mean(),
        "null_median":            np.median(null_aucs),
        "null_std":               null_aucs.std(),
        "null_p95":               np.percentile(null_aucs, 95),
        "null_p99":               np.percentile(null_aucs, 99),
        "n_perm_gte_observed":    int(n_ge_obs),
        "p_value":                p_value,
    }])
    results_df.to_csv(args.output_dir / "permutation_results.tsv",
                      sep="\t", index=False)

    # Save full null distribution
    pd.DataFrame({"perm_idx": np.arange(args.n_permutations),
                   "null_auc": null_aucs}
                  ).to_csv(args.output_dir / "permutation_null_aucs.tsv",
                           sep="\t", index=False)

    # ============== Step 4: Plot ==============
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(null_aucs, bins=30, color="#888", alpha=0.7,
            edgecolor="white", linewidth=0.6,
            label=f"Null distribution (n={args.n_permutations})")
    ax.axvline(auc_obs, color="#dd8452", linewidth=2.5,
                label=f"Observed AUC = {auc_obs:.3f}")
    ax.axvline(0.5, color="grey", linestyle=":", alpha=0.5,
                label="Chance (AUC=0.5)")
    ax.axvline(np.percentile(null_aucs, 95), color="red", linestyle="--",
                alpha=0.4, label=f"Null 95th %ile = {np.percentile(null_aucs, 95):.3f}")
    ax.set_xlabel("LOOCV AUC")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation test — BRCA vs not-BRCA prediction\n"
                 f"observed AUC = {auc_obs:.3f}, p = {p_value:.4f} "
                 f"({n_ge_obs}/{args.n_permutations} permutations ≥ observed)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "permutation_distribution.png",
                  dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info(f"Wrote {args.output_dir / 'permutation_distribution.png'}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
