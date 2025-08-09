#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from src import load_config
from src.data_loader import resolve_data_paths, load_raw_tables
from src import get_logger

_LOG = get_logger("make_sample")


def main():
    ap = argparse.ArgumentParser(description="Create a small sampled dataset (keep all fraud + sample normal).")
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--outdir", default="data/sample", help="Output dir")
    ap.add_argument("--normal-ratio", type=float, default=0.02, help="Fraction of normal transactions to keep")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    dpaths = resolve_data_paths(cfg)
    tables = load_raw_tables(dpaths)

    tx = tables["transactions"].copy()
    lb = tables["labels"].copy()
    users = tables["users"].copy()
    cards = tables["cards"].copy()
    merchants = tables["merchants"].copy()

    # Guess id columns quickly
    tid = next((c for c in ["transaction_id","tx_id","id"] if c in tx.columns), None)
    if not tid: raise KeyError("transactions: missing transaction_id")
    lid = next((c for c in [tid,"transaction_id","tx_id","id"] if c in lb.columns), None)
    if not lid: raise KeyError("labels: missing transaction id column")
    label_col = lb.columns[-1]

    merged = tx.merge(lb[[lid, label_col]], left_on=tid, right_on=lid, how="left").drop(columns=[lid])
    merged[label_col] = merged[label_col].fillna(0).astype(int)

    fraud = merged[merged[label_col]==1]
    normal = merged[merged[label_col]==0]
    n_keep = int(max(1, args.normal_ratio * len(normal)))
    normal_s = normal.sample(n=n_keep, random_state=args.random_state) if n_keep < len(normal) else normal
    sub = pd.concat([fraud, normal_s], axis=0).reset_index(drop=True)

    # Filter side tables to referenced keys
    cid = next((c for c in ["card_id","cardId","pan_id"] if c in sub.columns), None)
    uid = next((c for c in ["client_id","user_id","customer_id"] if c in sub.columns), None)
    mid = next((c for c in ["merchant_id","m_id","merchant"] if c in sub.columns), None)

    if cid and "card_id" in cards.columns:
        cards = cards[cards["card_id"].isin(sub[cid].dropna().unique())]
    if uid and "client_id" in users.columns:
        users = users[users["client_id"].isin(sub[uid].dropna().unique())]
    if mid and "merchant_id" in merchants.columns:
        merchants = merchants[merchants["merchant_id"].isin(sub[mid].dropna().unique())]

    # Write out
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    tx_cols = [c for c in tx.columns if c in sub.columns]
    tx_out = sub[tx_cols].copy()
    tx_out.to_csv(out / "transactions.csv", index=False)
    # labels
    lb_out = sub[[tid, label_col]].rename(columns={tid: "transaction_id"})
    lb_out.to_csv(out / "fraud_labels.csv", index=False)
    # side tables
    users.to_json(out / "users.json", orient="records")
    cards.to_json(out / "cards.json", orient="records")
    merchants.to_json(out / "merchant.json", orient="records")

    _LOG.info("Sample created in %s  | tx=%d (fraud=%d, normal=%d)", out, len(sub), len(fraud), len(normal_s))


if __name__ == "__main__":
    main()
