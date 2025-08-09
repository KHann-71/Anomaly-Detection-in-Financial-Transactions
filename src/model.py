from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import get_logger

_LOG = get_logger("model")


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int = 42):
    try:
        import torch, random, os
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random as pyrand
        pyrand.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        np.random.seed(seed)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Return precision, recall, f1_macro, auc using sklearn."""
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
    }


# ============================================================
# Base class
# ============================================================
class BaseModel:
    def fit(self, *args, **kwargs): raise NotImplementedError
    def predict_proba(self, X): raise NotImplementedError
    def predict(self, X, threshold: float = 0.5):
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)
    def save(self, path: Path): raise NotImplementedError
    @classmethod
    def load(cls, path: Path): raise NotImplementedError


# ============================================================
# 1) Scikit-learn models
# ============================================================
class SklearnLR(BaseModel):
    def __init__(self, **params):
        from sklearn.linear_model import LogisticRegression
        default = dict(max_iter=1000, n_jobs=None, class_weight="balanced", solver="lbfgs")
        default.update(params or {})
        self.model = LogisticRegression(**default)

    def fit(self, X, y):
        _LOG.info("Training LogisticRegression on %s samples...", len(y))
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path):
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path):
        import joblib
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        return obj


class SklearnRF(BaseModel):
    def __init__(self, **params):
        from sklearn.ensemble import RandomForestClassifier
        default = dict(
            n_estimators=300, max_depth=None, n_jobs=-1,
            class_weight="balanced_subsample", random_state=42
        )
        default.update(params or {})
        self.model = RandomForestClassifier(**default)

    def fit(self, X, y):
        _LOG.info("Training RandomForest on %s samples...", len(y))
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path):
        import joblib
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path):
        import joblib
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        return obj


# ============================================================
# 2) Autoencoder (Keras / TensorFlow)
# ============================================================
class KerasAutoencoder(BaseModel):
    """
    Unsupervised: fit on NORMAL only (X_normal). For scoring:
    - predict_proba returns anomaly probability via normalized reconstruction error.
    - threshold selection có thể đưa vào train.py (percentile).
    """
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden: Tuple[int, ...] = (128, 64),
                 lr: float = 1e-3, epochs: int = 20, batch_size: int = 512, seed: int = 42):
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden = tuple(hidden)
        self.lr = lr
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = seed
        self._build()

    def _build(self):
        try:
            import tensorflow as tf
            from tensorflow import keras
            tf.keras.utils.set_random_seed(self.seed)
            inputs = keras.Input(shape=(self.input_dim,))
            x = inputs
            for h in self.hidden:
                x = keras.layers.Dense(h, activation="relu")(x)
            z = keras.layers.Dense(self.latent_dim, activation="relu", name="z")(x)
            x = z
            for h in reversed(self.hidden):
                x = keras.layers.Dense(h, activation="relu")(x)
            outputs = keras.layers.Dense(self.input_dim, activation="linear")(x)
            self.model = keras.Model(inputs, outputs, name="autoencoder")
            self.model.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mse")
        except Exception as e:
            raise ImportError("TensorFlow/Keras is required for KerasAutoencoder.") from e

    def fit(self, X_normal, y=None, X_val: Optional[np.ndarray] = None):
        import tensorflow as tf
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
        self.model.fit(
            X_normal, X_normal,
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=1, callbacks=callbacks
        )
        return self

    def _recon_error(self, X):
        X_hat = self.model.predict(X, verbose=0)
        # MSE per sample
        return np.mean((X - X_hat) ** 2, axis=1)

    def predict_proba(self, X):
        """Map reconstruction error to [0,1] as anomaly probability using min-max on the batch."""
        err = self._recon_error(X)
        # avoid division by zero
        if err.max() - err.min() < 1e-12:
            return np.zeros_like(err)
        return (err - err.min()) / (err.max() - err.min())

    def save(self, path: Path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        # Save weights and minimal config
        self.model.save(path.as_posix())
        meta = {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden": self.hidden,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }
        (path.parent / (path.stem + ".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        import tensorflow as tf
        from tensorflow import keras
        meta_path = Path(path).parent / (Path(path).stem + ".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        obj = cls(**meta) if meta else cls(input_dim=1)  # dummy init then replace
        obj.model = keras.models.load_model(Path(path).as_posix())
        # fix input_dim in case dummy
        obj.input_dim = obj.model.input_shape[-1]
        return obj


# ============================================================
# 3) GNNs with PyTorch + DGL (GCN, GAT, GIN)
# ============================================================
def _require_torch_dgl():
    try:
        import torch, dgl  # noqa
    except Exception as e:
        raise ImportError("GNN requires PyTorch (`torch`) and DGL (`dgl`) to be installed.") from e


class _GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden=64, out_classes=2, dropout=0.2):
        super().__init__()
        import dgl.nn as dglnn
        self.conv1 = dglnn.GraphConv(in_feats, hidden)
        self.conv2 = dglnn.GraphConv(hidden, out_classes)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, g, x):
        import torch.nn.functional as F
        h = self.conv1(g, x); h = F.relu(h); h = self.drop(h)
        h = self.conv2(g, h)
        return h


class _GAT(torch.nn.Module):
    def __init__(self, in_feats, hidden=32, heads=4, out_classes=2, dropout=0.2):
        super().__init__()
        import dgl.nn as dglnn
        self.gat1 = dglnn.GATConv(in_feats, hidden, num_heads=heads, feat_drop=dropout, attn_drop=dropout)
        self.gat2 = dglnn.GATConv(hidden * heads, out_classes, num_heads=1, feat_drop=dropout, attn_drop=dropout)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, g, x):
        import torch.nn.functional as F
        h = self.gat1(g, x); h = h.flatten(1); h = F.elu(h); h = self.drop(h)
        h = self.gat2(g, h).squeeze(1)
        return h


class _GIN(torch.nn.Module):
    def __init__(self, in_feats, hidden=64, out_classes=2, dropout=0.2):
        super().__init__()
        import dgl.nn as dglnn
        import torch.nn as nn

        def mlp(in_d, out_d):
            return nn.Sequential(nn.Linear(in_d, hidden), nn.ReLU(), nn.Linear(hidden, out_d))

        self.gin1 = dglnn.GINConv(mlp(in_feats, hidden), "sum")
        self.gin2 = dglnn.GINConv(mlp(hidden, hidden), "sum")
        self.lin = nn.Linear(hidden, out_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, g, x):
        import torch.nn.functional as F
        h = self.gin1(g, x); h = F.relu(h); h = self.drop(h)
        h = self.gin2(g, h); h = F.relu(h)
        h = self.lin(h)
        return h


class GNNWrapper(BaseModel):
    """
    Wrapper for node classification:
    - expects a DGLGraph with ndata["feat"] (float) and ndata["label"] (long 0/1).
    - train/val/test masks optional; if absent, trains on all nodes.
    """
    def __init__(self, arch: str = "gin", hidden: int = 64, lr: float = 1e-3,
                 weight_decay: float = 0.0, dropout: float = 0.2, epochs: int = 30,
                 seed: int = 42, class_weight: Optional[Tuple[float, float]] = None):
        _require_torch_dgl()
        import torch
        self.arch = arch.lower()
        self.hidden = hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.epochs = epochs
        self.seed = seed
        self.class_weight = class_weight
        self.net = None  # initialized at fit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _make_net(self, in_feats: int, out_classes: int = 2):
        import torch
        if self.arch == "gcn":
            self.net = _GCN(in_feats, self.hidden, out_classes, self.dropout)
        elif self.arch == "gat":
            self.net = _GAT(in_feats, hidden=max(8, self.hidden // 2), heads=4, out_classes=out_classes, dropout=self.dropout)
        elif self.arch == "gin":
            self.net = _GIN(in_feats, self.hidden, out_classes, self.dropout)
        else:
            raise ValueError("Unknown GNN arch. Use one of: gcn, gat, gin.")
        self.net.to(self.device)

    def fit(self, graph, masks: Optional[Dict[str, np.ndarray]] = None):
        import torch
        set_seed(self.seed)
        g = graph.to(self.device)
        x = g.ndata.get("feat", None)
        if x is None:
            raise ValueError("Graph missing ndata['feat']. Provide node features in loader.")
        y = g.ndata["label"].to(self.device)
        in_feats = x.shape[1]
        self._make_net(in_feats)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Loss
        if self.class_weight is not None:
            w = torch.tensor(self.class_weight, dtype=torch.float32, device=self.device)
        else:
            # weight inversely proportional to class frequency
            counts = torch.bincount(y)
            w = counts.sum() / (2.0 * counts.clamp(min=1).float())
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)

        # Masks
        if masks:
            tr_mask = torch.tensor(masks.get("train"), dtype=torch.bool, device=self.device)
            va_mask = torch.tensor(masks.get("val"), dtype=torch.bool, device=self.device)
        else:
            n = y.shape[0]
            idx = torch.randperm(n)
            n_tr = int(0.7 * n); n_va = int(0.15 * n)
            tr_mask = torch.zeros(n, dtype=torch.bool, device=self.device); tr_mask[idx[:n_tr]] = True
            va_mask = torch.zeros(n, dtype=torch.bool, device=self.device); va_mask[idx[n_tr:n_tr+n_va]] = True

        for ep in range(1, self.epochs + 1):
            self.net.train()
            logits = self.net(g, x)
            loss = loss_fn(logits[tr_mask], y[tr_mask])
            opt.zero_grad(); loss.backward(); opt.step()

            if ep % 5 == 0 or ep == 1 or ep == self.epochs:
                self.net.eval()
                with torch.no_grad():
                    logits_va = self.net(g, x)[va_mask]
                    prob_va = torch.softmax(logits_va, dim=1)[:, 1].detach().cpu().numpy()
                    y_va = y[va_mask].detach().cpu().numpy()
                    metrics = compute_metrics(y_va, prob_va)
                    _LOG.info("GNN[%s] epoch %d | loss=%.4f | val AUC=%.4f F1=%.4f",
                              self.arch.upper(), ep, float(loss.item()), metrics["auc"], metrics["f1_macro"])
        return self

    def predict_proba(self, graph):
        import torch
        g = graph.to(self.device)
        x = g.ndata["feat"]
        self.net.eval()
        with torch.no_grad():
            logits = self.net(g, x)
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        return prob

    def save(self, path: Path):
        import torch
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "arch": self.arch,
            "hidden": self.hidden,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "seed": self.seed,
            "class_weight": self.class_weight,
        }, path)

    @classmethod
    def load(cls, path: Path):
        import torch
        chkpt = torch.load(Path(path), map_location="cpu")
        obj = cls(
            arch=chkpt["arch"], hidden=chkpt["hidden"], lr=chkpt["lr"],
            weight_decay=chkpt["weight_decay"], dropout=chkpt["dropout"],
            epochs=chkpt["epochs"], seed=chkpt["seed"], class_weight=chkpt["class_weight"]
        )
        # net will be built on first call to predict_proba/fit (needs in_feats)
        # To restore immediately, we need in_feats; caller should call `warm_start(graph)`
        obj._loaded_state = chkpt["state_dict"]
        return obj

    def warm_start(self, graph):
        """Call once after load to restore net weights given a graph (to get in_feats)."""
        import torch
        in_feats = graph.ndata["feat"].shape[1]
        self._make_net(in_feats)
        self.net.load_state_dict(self._loaded_state)
        delattr(self, "_loaded_state")


# ============================================================
# Factory
# ============================================================
def build_model(cfg: Dict[str, Any], name: str) -> BaseModel:
    """
    name in {'lr','logreg','rf','random_forest','autoencoder','gcn','gat','gin'}
    Hyperparameters read from cfg['models'][name] if exists.
    """
    models_cfg = (cfg or {}).get("models", {})
    key = name.lower()
    params = models_cfg.get(key, {})

    if key in {"lr", "logreg", "logistic_regression"}:
        return SklearnLR(**params)
    if key in {"rf", "random_forest"}:
        return SklearnRF(**params)
    if key in {"autoencoder", "ae"}:
        input_dim = params.get("input_dim")
        if input_dim is None:
            raise ValueError("models.autoencoder.input_dim must be set (number of features after preprocessing).")
        return KerasAutoencoder(
            input_dim=input_dim,
            latent_dim=params.get("latent_dim", 32),
            hidden=tuple(params.get("hidden", (128, 64))),
            lr=params.get("lr", 1e-3),
            epochs=params.get("epochs", 20),
            batch_size=params.get("batch_size", 512),
            seed=params.get("seed", 42),
        )
    if key in {"gcn", "gat", "gin"}:
        return GNNWrapper(
            arch=key,
            hidden=params.get("hidden", 64),
            lr=params.get("lr", 1e-3),
            weight_decay=params.get("weight_decay", 0.0),
            dropout=params.get("dropout", 0.2),
            epochs=params.get("epochs", 30),
            seed=params.get("seed", 42),
            class_weight=tuple(params.get("class_weight", (1.0, 1.0))),
        )
    raise ValueError(f"Unknown model name: {name}")
