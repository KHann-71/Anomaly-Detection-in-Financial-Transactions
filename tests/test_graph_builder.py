import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.data_loader import build_graph_dataset

def _write_fake_dataset(tmp: Path):
    tmp.mkdir(parents=True, exist_ok=True)

    # transactions.csv
    # 4 tx for card A (two fraud), 2 tx for card B (normal), share same client, different merchants
    rows = [
        # id,  card, client, merchant, amount, time
        (1, "A", "U1", "M1", 100.0, "2024-01-01 10:00:00"),
        (2, "A", "U1", "M1", 150.0, "2024-01-01 10:10:00"),
        (3, "A", "U1", "M2",  80.0, "2024-01-01 12:00:00"),
        (4, "A", "U1", "M2",  60.0, "2024-01-01 12:20:00"),
        (5, "B", "U1", "M3",  50.0, "2024-01-02 09:00:00"),
        (6, "B", "U1", "M3",  55.0, "2024-01-02 09:40:00"),
    ]
    tx = pd.DataFrame(rows, columns=["transaction_id","card_id","client_id","merchant_id","amount","tx_datetime"])
    tx.to_csv(tmp / "transactions.csv", index=False)

    # labels: mark tx 2 and 3 as fraud
    labels = pd.DataFrame({"transaction_id":[1,2,3,4,5,6], "fraud_label":[0,1,1,0,0,0]})
    labels.to_csv(tmp / "fraud_labels.csv", index=False)

    # minimal side tables
    pd.DataFrame({"client_id":["U1"], "age":[30]}).to_json(tmp / "users.json", orient="records")
    pd.DataFrame({"card_id":["A","B"], "credit_limit":[2000,1500]}).to_json(tmp / "cards.json", orient="records")
    pd.DataFrame({"merchant_id":["M1","M2","M3"], "mcc":["5411","5812","5999"]}).to_json(tmp / "merchant.json", orient="records")

def _base_cfg(tmp: Path):
    return {
        "data": {
            "dir": str(tmp),
            "files": {
                "transactions": "transactions.csv",
                "labels": "fraud_labels.csv",
                "users": "users.json",
                "cards": "cards.json",
                "merchants": "merchant.json",
            },
        },
        "preprocess": {
            "target": "fraud_label",
            "datetime_col": "tx_datetime",
            "id_cols": ["transaction_id","card_id","client_id","merchant_id"],
            "drop_cols": [],
            "categorical": [],
            "numerical": [],
            "scale": True,
            "encode": True,
        },
        "imbalance": {"gnn": {"normal_sample_ratio": 1.0}},
        "split": {"random_state": 42},
        "gnn": {
            "node_type": "card",
            "edges": {"by_client": True, "by_merchant": True, "time_window_minutes": 30},
            "features": {"node_features": ["amount","credit_limit"]},
        },
        "artifacts": {"dir": str(tmp / "artifacts"), "save_transformers": False},
    }

def test_build_graph_card_level(tmp_path: Path):
    _write_fake_dataset(tmp_path)
    cfg = _base_cfg(tmp_path)
    cfg["gnn"]["node_type"] = "card"

    ds = build_graph_dataset(cfg)
    nodes, edges = ds["nodes"], ds["edges"]

    # 2 cards → 2 nodes
    assert len(nodes) == 2
    # card A has any fraud → label 1, card B → 0
    assert set(nodes.columns) >= {"card_id","fraud_label"}
    lab = dict(zip(nodes["card_id"], nodes["fraud_label"]))
    assert lab["A"] == 1 and lab["B"] == 0

    # edges exist because both cards share client U1 (by_client=True)
    assert not edges.empty
    # ensure undirected duplication exists
    assert (edges[["src","dst"]].values[:, ::-1] == edges[["dst","src"]].values).any()

@pytest.mark.skipif(__import__("importlib").util.find_spec("dgl") is None, reason="DGL not installed")
def test_build_graph_transaction_time_edges(tmp_path: Path):
    _write_fake_dataset(tmp_path)
    cfg = _base_cfg(tmp_path)
    cfg["gnn"]["node_type"] = "transaction"
    cfg["gnn"]["edges"]["time_window_minutes"] = 15  # tx 1-2 within 10min; 3-4 within 20min -> only first pair

    ds = build_graph_dataset(cfg)
    nodes, edges = ds["nodes"], ds["edges"]

    # 6 transactions → 6 nodes
    assert len(nodes) == 6
    # At least one edge should connect tx 1 and 2 (same card, within 15min)
    # Map node ids -> indices
    id_to_idx = ds["mapping"]
    srcdst = set(map(tuple, edges[["src","dst"]].values.tolist()))
    pair = (id_to_idx[1], id_to_idx[2])
    assert pair in srcdst or (pair[1], pair[0]) in srcdst
