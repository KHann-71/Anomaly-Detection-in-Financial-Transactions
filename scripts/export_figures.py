#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import load_config, get_logger, paths
from scripts.evaluate_thresholds import _probs_from_task, _artifacts_dir

_LOG = get_logger("export_figures")

def _ensure_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d

def _plot_roc(y_true: np.ndarray, y_prob: np.ndarray, out: Path):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out); plt.close()

def _plot_pr(y_true: np.ndarray, y_prob: np.ndarray, out: Path):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve"); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(out); plt.close()

def main():
    ap = argparse.ArgumentParser(description="Export ROC & PR figures")
    ap.add_argument("--task", choices=["tabular","autoencoder","gnn"], help="Pipeline to evaluate")
    ap.add_argument("--model", default="lr", help="tabular: lr|rf ; gnn: gcn|gat|gin; ignored for autoencoder")
    ap.add_argument("--config", default=None)
    ap.add_argument("--from-probs", default=None, help="CSV with y_true,y_prob (skip model loading)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    fig_dir = _ensure_dir(_artifacts_dir(cfg) / "figures")

    if args.from-probs:
        df = pd.read_csv(args.from-probs)
        y_true, y_prob = df["y_true"].to_numpy(), df["y_prob"].to_numpy()
    else:
        if not args.task:
            ap.error("--task is required when --from-probs is not provided")
        y_true, y_prob = _probs_from_task(cfg, args.task, args.model.lower())

    _plot_roc(y_true, y_prob, fig_dir / "roc.png")
    _plot_pr(y_true, y_prob, fig_dir / "pr.png")
    _LOG.info("Saved figures to %s", fig_dir)

if __name__ == "__main__":
    main()
