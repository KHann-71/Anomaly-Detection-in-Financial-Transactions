from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from . import get_logger, load_config, paths
from .data_loader import (
    make_tabular_dataset,
    make_autoencoder_splits,
    build_graph_dataset,
    save_artifacts,
)
from .model import build_model, compute_metrics, KerasAutoencoder, GNNWrapper

_LOG = get_logger("train")


# ---------------- utils ----------------
def _artifacts_dir(cfg: Dict[str, Any]) -> Path:
    from . import paths as _p
    d = Path(cfg.get("artifacts", {}).get("dir", _p()["artifacts_dir"])).resolve()
    d.mkdir(parents=True, exist_ok=True)
    (d / "models").mkdir(exist_ok=True)
    return d


def _save_metrics(metrics: Dict[str, Any], out_dir: Path, name: str):
    p = out_dir / f"{name}_metrics.json"
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _LOG.info("Saved metrics -> %s", p)


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Scan 200 thresholds; return thresh with best F1_macro + its metrics."""
    ts = np.linspace(0.01, 0.99, 200)
    best_t, best_m, best_f1 = 0.5, {}, -1.0
    for t in ts:
        m = compute_metrics(y_true, (y_prob >= t).astype(int))
        if m["f1_macro"] > best_f1:
            best_t, best_m, best_f1 = t, m, m["f1_macro"]
    # recompute auc with probabilities
    from sklearn.metrics import roc_auc_score
    best_m["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    return float(best_t), best_m


# ---------------- runners ----------------
def run_tabular(cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    Xtr, ytr, Xv, yv, Xte, yte, arts = make_tabular_dataset(cfg)
    # models.<name> hyperparams in config; factory builds right model class
    model = build_model(cfg, model_name)
    model.fit(np.vstack([Xtr, Xv]), np.concatenate([ytr, yv]))

    # eval on test
    prob = model.predict_proba(Xte)
    t_best, m_best = _find_best_threshold(yte, prob)

    # save
    out_dir = _artifacts_dir(cfg)
    model_path = out_dir / "models" / f"{model_name}.joblib"
    model.save(model_path)
    _save_metrics({"best_threshold": t_best, **m_best}, out_dir, model_name)
    save_artifacts(arts, cfg)
    return {"model_path": str(model_path), "metrics": m_best, "threshold": t_best}


def run_autoencoder(cfg: Dict[str, Any]) -> Dict[str, Any]:
    Xtr, Xte, yte, arts = make_autoencoder_splits(cfg)
    # get input_dim from encoded features
    input_dim = Xtr.shape[1]
    cfg.setdefault("models", {}).setdefault("autoencoder", {})["input_dim"] = input_dim
    model = build_model(cfg, "autoencoder")  # KerasAutoencoder

    # split a bit of Xtr as val for early stopping
    n = len(Xtr); n_val = max(1, int(0.1 * n))
    Xtrain, Xval = Xtr[:-n_val], Xtr[-n_val:]
    model: KerasAutoencoder
    model.fit(Xtrain, X_val=Xval)

    # eval
    prob = model.predict_proba(Xte)
    t_best, m_best = _find_best_threshold(yte, prob)

    # save (Keras saves to a dir or .keras)
    out_dir = _artifacts_dir(cfg)
    model_path = out_dir / "models" / "autoencoder.keras"
    model.save(model_path)
    _save_metrics({"best_threshold": t_best, **m_best}, out_dir, "autoencoder")
    save_artifacts(arts, cfg)
    return {"model_path": str(model_path), "metrics": m_best, "threshold": t_best}


def run_gnn(cfg: Dict[str, Any], arch: str) -> Dict[str, Any]:
    ds = build_graph_dataset(cfg)
    g = ds["graph"]
    if g is None:
        raise RuntimeError("No DGL graph built. Please install `dgl` and ensure edges were created.")
    # Build + train
    model: GNNWrapper = build_model(cfg, arch)
    model.fit(g)  # internal split 70/15/15 if masks not provided

    # eval on all nodes (simple report)
    prob = model.predict_proba(g)
    y = g.ndata["label"].cpu().numpy()
    t_best, m_best = _find_best_threshold(y, prob)

    # save
    out_dir = _artifacts_dir(cfg)
    model_path = out_dir / "models" / f"{arch}.pth"
    model.save(model_path)
    _save_metrics({"best_threshold": t_best, **m_best}, out_dir, arch)
    return {"model_path": str(model_path), "metrics": m_best, "threshold": t_best}


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Training pipeline for fraud detection")
    parser.add_argument("--task", choices=["tabular", "autoencoder", "gnn"], required=True,
                        help="What to train: tabular (LR/RF), autoencoder, or gnn (gcn/gat/gin)")
    parser.add_argument("--model", type=str, default="lr",
                        help="For tabular: lr|rf. For gnn: gcn|gat|gin. Ignored for autoencoder.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config)) if args.config else load_config()
    _LOG.info("Loaded config.")

    if args.task == "tabular":
        if args.model.lower() not in {"lr", "logreg", "logistic_regression", "rf", "random_forest"}:
            raise ValueError("For --task tabular, --model must be lr|rf")
        result = run_tabular(cfg, args.model.lower())

    elif args.task == "autoencoder":
        result = run_autoencoder(cfg)

    elif args.task == "gnn":
        if args.model.lower() not in {"gcn", "gat", "gin"}:
            raise ValueError("For --task gnn, --model must be gcn|gat|gin")
        result = run_gnn(cfg, args.model.lower())

    else:
        raise ValueError("Unknown task")

    _LOG.info("Training complete. %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
