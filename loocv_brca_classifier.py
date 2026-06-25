#!/usr/bin/env python3
"""
loocv_brca_classifier.py

Leave-one-out cross-validation binary classifier on the k=1000 cfDNA
cluster matrix: BRCA (n=13) vs not-BRCA (n=30).

Two classifiers compared:
  1. Random Forest (200 trees, balanced class weights)
  2. L1-regularized Logistic Regression (with feature standardization)

Outputs:
  - loocv_predictions.tsv          per-sample predictions + probabilities
  - confusion_matrices.tsv         TN/FP/FN/TP for both classifiers
  - metrics_summary.tsv            accuracy, AUC, precision, recall, F1
  - top_clusters_rf.tsv            top features by RF importance
  - top_clusters_lr.tsv            top features by LR L1 coefficient magnitude
  - prediction_probabilities.png   probability distribution per class, both classifiers
  - roc_curves.png                 ROC curve for both classifiers
  - loocv.log
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
    p.add_argument("--cfdna-cluster-matrix", required=True, type=Path,
                   help="k=1000 cfDNA cluster matrix (clusters x samples)")
    p.add_argument("--cfdna-metadata",       required=True, type=Path,
                   help="cfdna_sample_metadata.tsv (sample_id, cancer_type)")
    p.add_argument("--positive-class",       default="BRCA",
                   help="Positive class label (default: BRCA)")
    p.add_argument("--output-dir",           required=True, type=Path)
    p.add_argument("--rf-trees",             type=int, default=200)
    p.add_argument("--lr-C",                 type=float, default=1.0)
    p.add_argument("--seed",                 type=int, default=42)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "loocv.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Loading cfDNA cluster matrix: {args.cfdna_cluster_matrix}")
    cfdna = pd.read_csv(args.cfdna_cluster_matrix, sep="\t", index_col=0)
    logging.info(f"  shape: {cfdna.shape} (clusters x samples)")

    meta = pd.read_csv(args.cfdna_metadata, sep="\t")
    sample_to_cancer = dict(zip(meta["sample_id"], meta["cancer_type"]))

    # Transpose to samples x clusters
    X_df = cfdna.T  # rows = samples, cols = clusters
    X_df = X_df.loc[[s for s in X_df.index if s in sample_to_cancer]]
    y_str = np.array([sample_to_cancer[s] for s in X_df.index])
    y = (y_str == args.positive_class).astype(int)
    sample_ids = list(X_df.index)
    X = X_df.values.astype(np.float64)

    logging.info(f"Samples: {len(sample_ids)}")
    logging.info(f"Features (clusters): {X.shape[1]}")
    logging.info(f"Positive class '{args.positive_class}': {y.sum()} samples")
    logging.info(f"Negative class (not-{args.positive_class}): {(1-y).sum()} samples")
    logging.info(f"Cancer types in cohort:")
    for ct, n in pd.Series(y_str).value_counts().items():
        logging.info(f"  {ct}: {n}")

    if y.sum() < 2 or (1-y).sum() < 2:
        raise ValueError("Need at least 2 samples per class for LOOCV")

    # Baseline accuracy: always predict majority class
    majority = 0 if (1-y).sum() > y.sum() else 1
    baseline_acc = ((y == majority).sum() / len(y)) * 100
    logging.info(f"Baseline (always predict majority class {majority}): "
                 f"{baseline_acc:.1f}%")

    # ============== LOOCV setup ==============
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )

    loo = LeaveOneOut()

    # Storage for predictions
    rf_pred = np.zeros(len(y), dtype=int)
    rf_proba = np.zeros(len(y), dtype=float)
    lr_pred = np.zeros(len(y), dtype=int)
    lr_proba = np.zeros(len(y), dtype=float)

    # Storage for feature importance (averaged across folds)
    n_features = X.shape[1]
    rf_imp_sum = np.zeros(n_features, dtype=float)
    lr_coef_sum = np.zeros(n_features, dtype=float)

    logging.info(f"Running LOOCV ({len(y)} folds)")
    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # --- Random Forest ---
        rf = RandomForestClassifier(
            n_estimators=args.rf_trees,
            class_weight="balanced",
            max_features="sqrt",
            random_state=args.seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_pred[test_idx[0]] = rf.predict(X_test)[0]
        rf_proba[test_idx[0]] = rf.predict_proba(X_test)[0, 1]
        rf_imp_sum += rf.feature_importances_

        # --- Logistic Regression L1 (standardized) ---
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        lr = LogisticRegression(
            penalty="l1", C=args.lr_C, solver="liblinear",
            class_weight="balanced",
            random_state=args.seed,
            max_iter=2000,
        )
        lr.fit(X_train_s, y_train)
        lr_pred[test_idx[0]] = lr.predict(X_test_s)[0]
        lr_proba[test_idx[0]] = lr.predict_proba(X_test_s)[0, 1]
        lr_coef_sum += lr.coef_[0]

        if (fold_idx + 1) % 10 == 0 or fold_idx == len(y) - 1:
            logging.info(f"  fold {fold_idx+1}/{len(y)}")

    # Average feature importance
    rf_imp_avg = rf_imp_sum / len(y)
    lr_coef_avg = lr_coef_sum / len(y)

    # ============== Metrics ==============
    logging.info("Computing metrics")

    def compute_metrics(name, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred) * 100
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = float("nan")
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        rec  = recall_score(y_true, y_pred, zero_division=0) * 100
        f1   = f1_score(y_true, y_pred, zero_division=0) * 100
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                            labels=[0, 1]).ravel()
        logging.info(f"  {name}:")
        logging.info(f"    accuracy:  {acc:.1f}% (baseline: {baseline_acc:.1f}%)")
        logging.info(f"    AUC:       {auc:.3f}")
        logging.info(f"    precision: {prec:.1f}% (positive class)")
        logging.info(f"    recall:    {rec:.1f}% (positive class)")
        logging.info(f"    F1:        {f1:.1f}%")
        logging.info(f"    confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
        return dict(name=name, accuracy=acc, auc=auc, precision=prec,
                    recall=rec, f1=f1, TN=int(tn), FP=int(fp),
                    FN=int(fn), TP=int(tp))

    m_rf = compute_metrics("Random Forest", y, rf_pred, rf_proba)
    m_lr = compute_metrics("LogReg L1", y, lr_pred, lr_proba)

    pd.DataFrame([m_rf, m_lr]).to_csv(
        args.output_dir / "metrics_summary.tsv", sep="\t", index=False)

    # Predictions table
    pred_df = pd.DataFrame({
        "sample_id":       sample_ids,
        "true_cancer":     y_str,
        "true_label":      y,
        "rf_pred":         rf_pred,
        "rf_proba_pos":    rf_proba,
        "rf_correct":      (rf_pred == y).astype(int),
        "lr_pred":         lr_pred,
        "lr_proba_pos":    lr_proba,
        "lr_correct":      (lr_pred == y).astype(int),
    })
    pred_df.to_csv(args.output_dir / "loocv_predictions.tsv",
                    sep="\t", index=False)

    # Top features
    cluster_ids = list(cfdna.index)
    rf_top = pd.DataFrame({
        "cluster_id":    cluster_ids,
        "rf_importance": rf_imp_avg,
    }).sort_values("rf_importance", ascending=False)
    rf_top.head(50).to_csv(args.output_dir / "top_clusters_rf.tsv",
                            sep="\t", index=False)

    lr_top = pd.DataFrame({
        "cluster_id":      cluster_ids,
        "lr_coef":         lr_coef_avg,
        "lr_abs_coef":     np.abs(lr_coef_avg),
    }).sort_values("lr_abs_coef", ascending=False)
    lr_top.head(50).to_csv(args.output_dir / "top_clusters_lr.tsv",
                            sep="\t", index=False)

    # ============== Plots ==============
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: prediction probability per sample, by true class
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, classifier_name, proba in [
        (axes[0], "Random Forest", rf_proba),
        (axes[1], "LogReg L1",     lr_proba),
    ]:
        pos_proba = proba[y == 1]
        neg_proba = proba[y == 0]
        ax.scatter([0]*len(neg_proba), neg_proba, alpha=0.6, s=60,
                   color="#4c72b0",
                   label=f"not-{args.positive_class} (n={len(neg_proba)})")
        ax.scatter([1]*len(pos_proba), pos_proba, alpha=0.6, s=60,
                   color="#dd8452",
                   label=f"{args.positive_class} (n={len(pos_proba)})")
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.7,
                   label="decision threshold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["true=neg", "true=pos"])
        ax.set_ylabel(f"Predicted P({args.positive_class})")
        ax.set_title(f"{classifier_name}\n"
                     f"acc={accuracy_score(y, (proba>=0.5).astype(int))*100:.1f}%, "
                     f"AUC={roc_auc_score(y, proba):.3f}",
                     fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"LOOCV prediction probabilities — "
                 f"{args.positive_class} vs not-{args.positive_class}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.output_dir / "prediction_probabilities.png",
                dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Plot 2: ROC curves
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(8, 8))
    for name, proba, color in [
        ("Random Forest", rf_proba, "#4c72b0"),
        ("LogReg L1",     lr_proba, "#dd8452"),
    ]:
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":",
            label="chance (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"LOOCV ROC — {args.positive_class} vs not-{args.positive_class}\n"
                 f"({len(sample_ids)} samples, {n_features} cluster features)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "roc_curves.png",
                dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ============== Summary ==============
    logging.info("")
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Baseline (majority-class):    {baseline_acc:.1f}%")
    logging.info(f"Random Forest accuracy:       {m_rf['accuracy']:.1f}%, "
                 f"AUC={m_rf['auc']:.3f}")
    logging.info(f"LogReg L1 accuracy:           {m_lr['accuracy']:.1f}%, "
                 f"AUC={m_lr['auc']:.3f}")
    above_baseline_rf = m_rf['accuracy'] - baseline_acc
    above_baseline_lr = m_lr['accuracy'] - baseline_acc
    logging.info(f"Above baseline (RF):          {above_baseline_rf:+.1f}%")
    logging.info(f"Above baseline (LR):          {above_baseline_lr:+.1f}%")
    if max(m_rf['auc'], m_lr['auc']) > 0.7:
        logging.info("  Strong signal (AUC > 0.7)")
    elif max(m_rf['auc'], m_lr['auc']) > 0.6:
        logging.info("  Modest signal (AUC > 0.6)")
    elif max(m_rf['auc'], m_lr['auc']) > 0.55:
        logging.info("  Weak signal (AUC > 0.55)")
    else:
        logging.info("  No detectable signal (AUC near 0.5)")
    logging.info("Done.")


if __name__ == "__main__":
    main()
