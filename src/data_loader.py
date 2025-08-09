from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from . import get_logger, load_config, paths

# Optional deps
try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False

try:
    import dgl, torch
    _HAS_DGL = True
except Exception:
    _HAS_DGL = False

_LOG = get_logger("data_loader")

# ---------------- dataclasses ----------------
@dataclass
class DataPaths:
    root: Path; transactions: Path; labels: Path; users: Path; cards: Path; merchants: Path

@dataclass
class SplitCfg:
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    stratify: bool = True

# ---------------- tiny helpers ----------------
def _p(x: Any) -> Path: return Path(x).resolve()
def _req(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"{name}: missing cols {miss}")

def _make_onehot():
    # Handle sklearn >=1.2 (sparse_output) vs older (sparse)
    try:
        from sklearn.preprocessing import OneHotEncoder
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        from sklearn.preprocessing import OneHotEncoder
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ---------------- locate & load raw files ----------------
def resolve_data_paths(cfg: Dict[str, Any]) -> DataPaths:
    prj = paths(); root = _p(cfg.get("data", {}).get("dir", prj["data_dir"]))
    f = cfg.get("data", {}).get("files", {})
    tx = root / f.get("transactions", "transactions.csv")
    lb = root / f.get("labels", "fraud_labels.csv")
    us = root / f.get("users", "users.json")
    ca = root / f.get("cards", "cards.json")
    me = root / f.get("merchants", "merchant.json")
    for p in (tx, lb, us, ca, me):
        if not p.exists(): raise FileNotFoundError(p)
    return DataPaths(root, tx, lb, us, ca, me)

def load_raw_tables(d: DataPaths) -> Dict[str, pd.DataFrame]:
    _LOG.info("Loading data from %s", d.root)
    to_df = lambda p: pd.read_json(p, lines=False) if p.suffix==".json" else pd.read_csv(p)
    return {
        "transactions": to_df(d.transactions),
        "labels": pd.read_csv(d.labels),
        "users": to_df(d.users),
        "cards": to_df(d.cards),
        "merchants": to_df(d.merchants),
    }

# ---------------- clean / merge / features ----------------
def clean_transactions(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy(); dt = cfg.get("preprocess", {}).get("datetime_col")
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str)
            # strip $, commas if looks numeric
            if s.str.replace(r"[0-9\.\-\,]", "", regex=True).str.strip().isin({"$", ""}).all():
                df[c] = s.str.replace(r"[\$,]", "", regex=True)
            try: df[c] = pd.to_numeric(df[c])
            except Exception: pass
    if dt and dt in df.columns: df[dt] = pd.to_datetime(df[dt], errors="coerce")
    return df

def merge_all(t: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> pd.DataFrame:
    target = cfg.get("preprocess", {}).get("target", "fraud_label")
    tx, lb, us, ca, me = t["transactions"].copy(), t["labels"].copy(), t["users"].copy(), t["cards"].copy(), t["merchants"].copy()

    tid = next((c for c in ["transaction_id","tx_id","id"] if c in tx.columns), None)
    if not tid: raise KeyError("transaction id missing")

    lb = lb.rename(columns={lb.columns[-1]: target})
    lid = next((c for c in [tid,"transaction_id","tx_id","id"] if c in lb.columns), None)
    if not lid: raise KeyError("labels id missing")

    df = tx.merge(lb[[lid, target]], left_on=tid, right_on=lid, how="left")
    if lid != tid: df = df.drop(columns=[lid])

    # guess foreign keys
    cid = next((c for c in ["card_id","cardId","pan_id"] if c in df.columns), None)
    uid = next((c for c in ["client_id","user_id","customer_id"] if c in df.columns), None)
    mid = next((c for c in ["merchant_id","m_id","merchant"] if c in df.columns), None)

    if uid and uid in us.columns: df = df.merge(us, on=uid, how="left", suffixes=("", "_user"))
    if cid and cid in ca.columns: df = df.merge(ca, on=cid, how="left", suffixes=("", "_card"))
    if mid and mid in me.columns: df = df.merge(me, on=mid, how="left", suffixes=("", "_merchant"))

    if df[target].isna().any():
        _LOG.warning("Dropping %d unlabeled rows", df[target].isna().sum())
        df = df[~df[target].isna()].copy()

    df[target] = df[target].astype(int)

    drops = [c for c in cfg.get("preprocess", {}).get("drop_cols", []) if c in df.columns]
    if drops: df = df.drop(columns=drops)
    return df

def feature_engineering(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy(); dt = cfg.get("preprocess", {}).get("datetime_col")
    if "age" in df.columns:
        if "age_group" not in df.columns:
            df["age_group"]=pd.cut(df["age"],[-1,25,35,50,65,math.inf],labels=["<=25","26-35","36-50","51-65","65+"])
        if "retirement_status" not in df.columns:
            df["retirement_status"]=(df["age"]>=60).astype(int)
    if dt and dt in df.columns and pd.api.types.is_datetime64_any_dtype(df[dt]):
        df["tx_hour"]=df[dt].dt.hour; df["tx_dayofweek"]=df[dt].dt.dayofweek
    if "amount" in df.columns:
        df["amount_log1p"]=np.log1p(df["amount"].clip(lower=0))
    return df

# ---------------- encode / split ----------------
def _infer_cols(df: pd.DataFrame, cfg: Dict[str, Any], target: str) -> Tuple[List[str], List[str]]:
    cat_cfg = cfg.get("preprocess", {}).get("categorical", []); num_cfg = cfg.get("preprocess", {}).get("numerical", [])
    cats = [c for c in (cat_cfg or df.select_dtypes(include=["object","category"]).columns) if c!=target and c in df.columns]
    nums = [c for c in (num_cfg or df.select_dtypes(include=[np.number]).columns) if c!=target and c in df.columns]
    return nums, cats

def encode_scale(df: pd.DataFrame, target: str, cfg: Dict[str, Any], fit=True, enc=None, scaler=None):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    nums, cats = _infer_cols(df, cfg, target); _req(df, nums+cats+[target], "encode_scale")
    trf=[]
    if nums:
        steps=[("scaler", scaler or StandardScaler())] if cfg.get("preprocess",{}).get("scale",True) else []
        trf.append(("num", Pipeline(steps) if steps else "passthrough", nums))
    if cats:
        if cfg.get("preprocess",{}).get("encode",True):
            trf.append(("cat", enc or _make_onehot(), cats))
        else: trf.append(("cat","passthrough",cats))
    ct = ColumnTransformer(trf, remainder="drop")
    X = ct.fit_transform(df[nums+cats]) if fit else ct.transform(df[nums+cats])
    y = df[target].values
    try: fn = list(ct.get_feature_names_out())
    except Exception: fn = [*nums, *cats]
    return X, y, {"column_transformer": ct, "feature_names": fn, "numerical_cols": nums, "categorical_cols": cats}

def stratified_split(X: np.ndarray, y: np.ndarray, s: SplitCfg):
    from sklearn.model_selection import train_test_split
    strat = y if s.stratify else None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=(s.val_size+s.test_size), random_state=s.random_state, stratify=strat)
    strat_tmp = ytmp if s.stratify else None
    t_ratio = s.test_size / (s.val_size + s.test_size)
    Xv, Xte, yv, yte = train_test_split(Xtmp, ytmp, test_size=t_ratio, random_state=s.random_state, stratify=strat_tmp)
    return Xtr, ytr, Xv, yv, Xte, yte

def apply_smote_if_needed(X: np.ndarray, y: np.ndarray, cfg: Dict[str, Any]):
    want = cfg.get("imbalance", {}).get("ml", {}).get("use_smote", False)
    if not want or not _HAS_SMOTE:
        if want and not _HAS_SMOTE: _LOG.warning("SMOTE requested but imblearn not installed. Skipping.")
        return X, y
    k = cfg.get("imbalance", {}).get("ml", {}).get("smote_k_neighbors", 5)
    _LOG.info("Applying SMOTE (k=%s)", k)
    sm = SMOTE(k_neighbors=k, random_state=cfg.get("split", {}).get("random_state", 42))
    return sm.fit_resample(X, y)

# ---------------- public builders ----------------
def make_tabular_dataset(cfg: Optional[Dict[str, Any]] = None):
    """Load → clean → merge → features → encode/scale → split → (SMOTE train)."""
    cfg = load_config() if cfg is None else cfg
    dp = resolve_data_paths(cfg); raw = load_raw_tables(dp)
    fe = feature_engineering(merge_all({**raw, "transactions": clean_transactions(raw["transactions"], cfg)}, cfg), cfg)
    target = cfg.get("preprocess", {}).get("target", "fraud_label")
    X, y, arts = encode_scale(fe, target, cfg, fit=True)
    Xtr, ytr, Xv, yv, Xte, yte = stratified_split(X, y, SplitCfg(**cfg.get("split", {})))
    Xtr, ytr = apply_smote_if_needed(Xtr, ytr, cfg)
    save_artifacts(arts, cfg)
    return Xtr, ytr, Xv, yv, Xte, yte, arts

def make_autoencoder_splits(cfg: Optional[Dict[str, Any]] = None):
    """Autoencoder: train trên NORMAL, test = NORMAL_holdout + FRAUD (cùng transformer)."""
    cfg = load_config() if cfg is None else cfg
    dp = resolve_data_paths(cfg); raw = load_raw_tables(dp)
    fe = feature_engineering(merge_all({**raw, "transactions": clean_transactions(raw["transactions"], cfg)}, cfg), cfg)
    target = cfg.get("preprocess", {}).get("target", "fraud_label")
    normal = fe[fe[target]==0].copy(); fraud = fe[fe[target]==1].copy()
    Xn, yn, arts = encode_scale(normal, target, cfg, fit=True)

    from sklearn.model_selection import train_test_split
    s = SplitCfg(**cfg.get("split", {}))
    Xtr, Xhold, ytr, _ = train_test_split(Xn, yn, test_size=s.test_size, random_state=s.random_state, stratify=yn if s.stratify else None)

    ct = arts["column_transformer"]; nums, cats = arts["numerical_cols"], arts["categorical_cols"]
    Xf = ct.transform(fraud[nums+cats]) if len(nums+cats) else np.empty((len(fraud),0))
    Xte = np.vstack([Xhold, Xf])
    yte = np.concatenate([np.zeros(len(Xhold),dtype=int), np.ones(len(Xf),dtype=int)])

    save_artifacts(arts, cfg)
    return Xtr, Xte, yte, arts

def build_graph_dataset(cfg: Optional[Dict[str, Any]] = None):
    """GNN: giữ ALL fraud + sample normal theo tỉ lệ; dựng nodes/edges theo config; trả về DGL graph nếu có."""
    cfg = load_config() if cfg is None else cfg
    dp = resolve_data_paths(cfg); raw = load_raw_tables(dp)
    fe = feature_engineering(merge_all({**raw, "transactions": clean_transactions(raw["transactions"], cfg)}, cfg), cfg)
    target = cfg.get("preprocess", {}).get("target", "fraud_label")
    dt = cfg.get("preprocess", {}).get("datetime_col")

    ratio = float(cfg.get("imbalance", {}).get("gnn", {}).get("normal_sample_ratio", 0.2))
    fraud, normal = fe[fe[target]==1], fe[fe[target]==0]
    keep = int(max(1, ratio*len(normal)))
    normal_s = normal.sample(n=keep, random_state=cfg.get("split",{}).get("random_state",42)) if keep<len(normal) else normal
    ws = pd.concat([fraud, normal_s]).reset_index(drop=True)

    node_type = cfg.get("gnn", {}).get("node_type", "card")
    ecfg = cfg.get("gnn", {}).get("edges", {})
    by_client, by_merchant, tw = bool(ecfg.get("by_client", True)), bool(ecfg.get("by_merchant", True)), ecfg.get("time_window_minutes")

    cid = next((c for c in ["card_id","cardId","pan_id"] if c in ws.columns), None)
    uid = next((c for c in ["client_id","user_id","customer_id"] if c in ws.columns), None)
    mid = next((c for c in ["merchant_id","m_id","merchant"] if c in ws.columns), None)
    tid = next((c for c in ["transaction_id","tx_id","id"] if c in ws.columns), None)

    if node_type=="transaction":
        if not tid: raise KeyError("transaction_id missing")
        key = tid; nodes = ws[[key, target]].drop_duplicates().reset_index(drop=True)
    elif node_type=="card":
        if not cid: raise KeyError("card_id missing")
        key = cid; nodes = ws.groupby(cid)[target].max().reset_index().rename(columns={cid:key})
    else:
        raise ValueError("gnn.node_type must be 'transaction' or 'card'")

    id2i = {nid:i for i, nid in enumerate(nodes[key].tolist())}
    edges=set()

    def connect(series: pd.Series):
        idxs=[id2i.get(x) for x in series if x in id2i]; idxs=[i for i in idxs if i is not None]
        for i in range(len(idxs)):
            for j in range(i+1,len(idxs)): edges.add((idxs[i], idxs[j]))

    if by_client and uid and uid in ws.columns:
        for _,g in ws.groupby(uid): connect(g[key].dropna().tolist())
    if by_merchant and mid and mid in ws.columns:
        for _,g in ws.groupby(mid): connect(g[key].dropna().tolist())

    if node_type=="transaction" and tw and dt and dt in ws.columns:
        ws=ws.sort_values(dt)
        grp = ws.groupby(cid or mid); win = pd.to_timedelta(int(tw), "m")
        for _,g in grp:
            g=g[[key,dt]].dropna().sort_values(dt); left=0; t=g[dt].values; ids=g[key].values
            for right in range(len(g)):
                while t[right]-t[left] > np.timedelta64(win.value,'ns'): left+=1
                win_ids=[id2i.get(x) for x in ids[left:right+1] if id2i.get(x) is not None]
                for i in range(len(win_ids)):
                    for j in range(i+1,len(win_ids)): edges.add((win_ids[i],win_ids[j]))

    edges_df = pd.DataFrame(list(edges), columns=["src","dst"])
    if not edges_df.empty:
        edges_df = pd.concat([edges_df, edges_df.rename(columns={"src":"dst","dst":"src"})], ignore_index=True).drop_duplicates()

    nodes_df = nodes.copy()
    feats = cfg.get("gnn", {}).get("features", {}).get("node_features", [])
    if feats:
        if node_type=="card" and cid:
            agg = {c:"mean" for c in feats if c in ws.columns}
            if agg:
                nf = ws.groupby(cid).agg(agg).reset_index().rename(columns={cid:key})
                nodes_df = nodes_df.merge(nf, on=key, how="left")
        else:
            cols=[c for c in feats if c in ws.columns]+[key]
            nf = ws[cols].drop_duplicates(subset=[key])
            nodes_df = nodes_df.merge(nf, on=key, how="left")

    result = {"nodes": nodes_df, "edges": edges_df, "mapping": id2i, "graph": None}
    if _HAS_DGL and not edges_df.empty:
        g = dgl.graph((torch.tensor(edges_df["src"].values), torch.tensor(edges_df["dst"].values)))
        fcols=[c for c in nodes_df.columns if c not in {key, target} and pd.api.types.is_numeric_dtype(nodes_df[c])]
        if fcols:
            g.ndata["feat"]=torch.tensor(nodes_df[fcols].fillna(0).values, dtype=torch.float32)
        g.ndata["label"]=torch.tensor(nodes_df[target].values, dtype=torch.long)
        result["graph"]=g
    return result

# ---------------- artifacts (save/load) ----------------
def save_artifacts(artifacts: dict, cfg: dict, logger=_LOG) -> None:
    art = cfg.get("artifacts", {})
    if not art.get("save_transformers", True): return
    d = _p(art.get("dir", paths()["artifacts_dir"])); d.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump(artifacts.get("column_transformer"), d / "column_transformer.joblib")
        (d / "feature_names.json").write_text(
            json.dumps(artifacts.get("feature_names", []), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info("Saved preprocessing artifacts to %s", d)
    except Exception as e:
        logger.warning("Could not save artifacts: %s", e)

def load_artifacts(cfg: dict) -> dict:
    d = _p(cfg.get("artifacts", {}).get("dir", paths()["artifacts_dir"]))
    import joblib
    ct = joblib.load(d / "column_transformer.joblib")
    feats = json.loads((d / "feature_names.json").read_text(encoding="utf-8"))
    return {"column_transformer": ct, "feature_names": feats}
