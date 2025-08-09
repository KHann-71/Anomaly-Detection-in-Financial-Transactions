#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src import load_config
from src.data_loader import make_tabular_dataset, make_autoencoder_splits, build_graph_dataset, load_artifacts
from src.model import build_model
from src.train import _find_best_threshold  # đã có trong train.py
from src import get_logger, paths

_LOG = get_logger("evaluate_thresholds")


def _artifacts_dir(cfg: Dict[str, Any]) -> Path:
    d = Path(cfg.get("artifacts", {}).get("dir", paths()["artifacts_dir"])).resolve()
    d.mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(exist_ok=True)
    return d

def _probs_from_task(cfg: Dict[str, Any], task: str, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if task == "tabular":
        # chỉ dùng test set để đánh giá
        _, _, _, _, Xte, yte, arts = make_tabular_dataset(cfg)
        # nạp model từ artifacts hoặc build + fit nhanh (ưu tiên nạp)
        out = _artifacts_dir(cfg) / "models"
        model_path = out / f"{model_name}.joblib"
        if model_path.exists():
            model = build_model(cfg, model_name).__class__.load(model_path)  # type: ignore
        else:
            # train nhanh nếu chưa có
            model = build_model(cfg, model_name)
            Xtr, ytr, *_ = make_tabular_dataset(cfg)[:2]
            model.fit(Xtr, ytr)
        y_prob = model.predict_proba(Xte)
        return yte, y_prob

    if task == "autoencoder":
        Xtr, Xte, yte, arts = make_autoencoder_splits(cfg)
        out = _artifacts_dir(cfg) / "models"
        model_path = out / "autoencoder.keras"
        if model_path.exists():
            from src.model import KerasAutoencoder
            model = KerasAutoencoder.load(model_path)
        else:
            from src.model import KerasAutoencoder
            input_dim = Xtr.shape[1]
            cfg.setdefault("models", {}).setdefault("autoencoder", {})["input_dim"] = int(input_dim)
            model = build_model(cfg, "autoencoder")
            n = len(Xtr); n_val = max(1, int(0.1 * n))
            model.fit(Xtr[:-n_val], X_val=Xtr[-n_val:])
        y_prob = model.predict_proba(Xte)
        return yte, y_prob

    if task == "gnn":
        ds = build_graph_dataset(cfg)
        g = ds["graph"]
        if g is None:
            raise RuntimeError("Graph not built. Install dgl and ensure edges > 0.")
        out = _artifacts_dir(cfg) / "models"
        model_path = out / f"{model_name}.pth"
        from src.model import GNNWrapper
        model = build_model(cfg, model_name)
        if model_path.exists():
            model = GNNWrapper.load(model_path)  # type: ignore
            model.warm_start(g)
        else:
            model.fit(g)
        y_prob = model.predict_proba(g)
        y_true = g.ndata["label"].cpu().numpy()
        return y_true, y_prob

    raise ValueError("task must be tabular|autoencoder|gnn")


def main():
    ap = argparse.ArgumentParser(description="Scan thresholds & pick best by F1-macro")
    ap.add_argument("--task", choices=["tabular", "autoencoder", "gnn"], help="Which pipeline to evaluate")
    ap.add_argument("--model", default="lr", help="tabular: lr|rf ; gnn: gcn|gat|gin; ignored for autoencoder")
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--from-probs", default=None, help="CSV file with columns y_true,y_prob (skip model loading)")
    ap.add_argument("--out", default=None, help="Output CSV path for threshold table (optional)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    art_dir = _artifacts_dir(cfg)

    if args.from_probs:
        df = pd.read_csv(args.from_probs)
        y_true, y_prob = df["y_true"].to_numpy(), df["y_prob"].to_numpy()
    else:
        if not args.task:
            ap.error("--task is required when --from-probs is not provided")
        y_true, y_prob = _probs_from_task(cfg, args.task, args.model.lower())

    # tạo bảng thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        })
    tbl = pd.DataFrame(rows)
    best_idx = tbl["f1_macro"].idxmax()
    best = tbl.loc[best_idx].to_dict()
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    best["auc"] = auc

    out_csv = Path(args.out) if args.out else art_dir / "thresholds_scan.csv"
    tbl.to_csv(out_csv, index=False)
    (art_dir / "best_threshold.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    _LOG.info("Best threshold = %.3f | P=%.3f R=%.3f F1=%.3f AUC=%.3f",
              best["threshold"], best["precision"], best["recall"], best["f1_macro"], best["auc"])
    _LOG.info("Saved table -> %s", out_csv)


if __name__ == "__main__":
    main()
