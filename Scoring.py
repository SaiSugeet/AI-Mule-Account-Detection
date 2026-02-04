# Scoring.py  (drop into utils/Scoring.py recommended)
"""
Robust scoring module for the Mule Detection demo.
This version auto-detects the project root whether the file sits in utils/ or project root.
"""

from pathlib import Path
import os, json
import joblib
import pandas as pd
import numpy as np
import networkx as nx
from datetime import timedelta

# Determine BASE_DIR robustly:
# If this file is inside a "utils" folder, project root is parent of utils.
# Otherwise project root is the parent of this file.
_THIS_FILE = Path(__file__).resolve()
if _THIS_FILE.parent.name.lower() == "utils":
    BASE_DIR = _THIS_FILE.parent.parent
else:
    BASE_DIR = _THIS_FILE.parent

RAW_DIR = BASE_DIR / "raw_data"
GRAPHS_DIR = BASE_DIR / "graphs"
MODEL_DIR = BASE_DIR / "model"
FEATURES_CACHE = BASE_DIR / "data" / "features_accounts.csv"

MODEL_PATH = MODEL_DIR / "model.pkl"
POLICY_PATH = MODEL_DIR / "policy.json"

_model = None
_policy = None
_features_median = None

def _ensure_model_loaded():
    global _model, _policy, _features_median
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run training notebook first.")
        _model = joblib.load(MODEL_PATH)
    if _policy is None:
        if POLICY_PATH.exists():
            with open(POLICY_PATH, "r") as f:
                _policy = json.load(f)
        else:
            _policy = {"allow_thresh": 0.3, "otp_thresh": 0.65, "chosen_thr": 0.5}
    if _features_median is None and FEATURES_CACHE.exists():
        try:
            df = pd.read_csv(FEATURES_CACHE)
            _features_median = df.median(numeric_only=True)
        except Exception:
            _features_median = None

def _load_transactions():
    tx_path = RAW_DIR / "simulated_transactions.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"Transactions CSV not found at {tx_path}. Run the simulation notebook first.")
    tx = pd.read_csv(tx_path)
    if "timestamp" in tx.columns:
        tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    return tx

def _compute_account_features(tx_df, account_id, as_of_time=None):
    if as_of_time is None:
        as_of_time = tx_df["timestamp"].max()
    window_24h = as_of_time - timedelta(days=1)
    window_7d = as_of_time - timedelta(days=7)
    epsilon = 1e-9

    in_24 = tx_df[(tx_df["receiver"] == account_id) & (tx_df["timestamp"] >= window_24h)]
    in_7 = tx_df[(tx_df["receiver"] == account_id) & (tx_df["timestamp"] >= window_7d)]
    out_24 = tx_df[(tx_df["sender"] == account_id) & (tx_df["timestamp"] >= window_24h)]
    out_7 = tx_df[(tx_df["sender"] == account_id) & (tx_df["timestamp"] >= window_7d)]

    feat = {}
    feat["account_id"] = account_id
    feat["inbound_count_24h"] = int(in_24.shape[0])
    feat["inbound_sum_24h"] = float(in_24["amount"].sum()) if in_24.shape[0] > 0 else 0.0
    feat["inbound_unique_senders_24h"] = int(in_24["sender"].nunique())

    feat["inbound_count_7d"] = int(in_7.shape[0])
    feat["inbound_sum_7d"] = float(in_7["amount"].sum()) if in_7.shape[0] > 0 else 0.0
    feat["inbound_unique_senders_7d"] = int(in_7["sender"].nunique())

    feat["outbound_count_24h"] = int(out_24.shape[0])
    feat["outbound_sum_24h"] = float(out_24["amount"].sum()) if out_24.shape[0] > 0 else 0.0
    feat["outbound_unique_receivers_24h"] = int(out_24["receiver"].nunique())

    feat["outbound_count_7d"] = int(out_7.shape[0])
    feat["outbound_sum_7d"] = float(out_7["amount"].sum()) if out_7.shape[0] > 0 else 0.0
    feat["outbound_unique_receivers_7d"] = int(out_7["receiver"].nunique())

    feat["pct_forwarded_7d"] = feat["outbound_sum_7d"] / (feat["inbound_sum_7d"] + epsilon)

    recent = tx_df[tx_df["timestamp"] >= window_7d]
    in_times = sorted(list(recent[recent["receiver"] == account_id]["timestamp"]))
    out_times = sorted(list(recent[recent["sender"] == account_id]["timestamp"]))
    avg_delay = -1.0
    if len(in_times) > 0 and len(out_times) > 0:
        delays = []
        j = 0
        for t_in in in_times:
            while j < len(out_times) and out_times[j] <= t_in:
                j += 1
            if j < len(out_times):
                delta = (pd.to_datetime(out_times[j]) - pd.to_datetime(t_in)).total_seconds()
                if delta >= 0:
                    delays.append(delta)
        if len(delays) > 0:
            avg_delay = float(np.mean(delays))
    feat["avg_forward_delay_seconds"] = avg_delay

    recent_edges = recent.groupby(["sender", "receiver"]).size().reset_index(name="weight")
    G = nx.DiGraph()
    for _, r in recent_edges.iterrows():
        G.add_edge(r["sender"], r["receiver"], weight=int(r["weight"]))
    feat["in_degree_7d"] = int(G.in_degree(account_id)) if account_id in G.nodes() else 0
    feat["out_degree_7d"] = int(G.out_degree(account_id)) if account_id in G.nodes() else 0
    try:
        pr = nx.pagerank(G, weight="weight") if len(G) > 0 else {}
        feat["pagerank_7d"] = float(pr.get(account_id, 0.0))
    except Exception:
        feat["pagerank_7d"] = 0.0

    feat["count_in_out_ratio_7d"] = (feat["inbound_count_7d"] + epsilon) / (feat["outbound_count_7d"] + epsilon)
    feat["many_inbound_senders_24h_flag"] = int(feat["inbound_unique_senders_24h"] >= 10)
    feat["burstiness_24h_vs_7d"] = feat["inbound_count_24h"] / (((feat["inbound_count_7d"] / 7.0) + epsilon))

    return feat

def score_transaction(sender, receiver, amount, as_of_time=None):
    _ensure_model_loaded()
    tx_df = _load_transactions()
    if as_of_time is None:
        as_of_time = tx_df["timestamp"].max()

    receiver_feat = _compute_account_features(tx_df, receiver, as_of_time=as_of_time)

    if FEATURES_CACHE.exists():
        header = list(pd.read_csv(FEATURES_CACHE, nrows=0).columns)
        ordered_cols = [c for c in header if c not in ("account_id","is_mule")]
    else:
        ordered_cols = [k for k in receiver_feat.keys() if k != "account_id"]

    X_row = pd.DataFrame([{c: receiver_feat.get(c, 0) for c in ordered_cols}])

    p = _model.predict_proba(X_row) if hasattr(_model, "predict_proba") else None
    if p is None:
        proba = float(_model.predict(X_row)[0])
    else:
        proba = float(p[:,1][0]) if p.shape[1] > 1 else float(p[:,0][0])

    allow_t = _policy.get("allow_thresh", 0.3)
    otp_t = _policy.get("otp_thresh", 0.65)
    if proba < allow_t:
        decision = "ALLOW"
    elif proba < otp_t:
        decision = "OTP"
    else:
        decision = "HOLD"

    reasons = []
    try:
        if _features_median is not None:
            diffs = []
            for f in ordered_cols:
                val = receiver_feat.get(f, 0)
                med = float(_features_median.get(f, 0.0)) if f in _features_median.index else 0.0
                diffs.append((f, float(val - med)))
            diffs_sorted = sorted(diffs, key=lambda x: -abs(x[1]))
            reasons = [f for f, d in diffs_sorted[:3]]
        else:
            kv = [(k, abs(v)) for k, v in receiver_feat.items() if k != "account_id"]
            kv_sorted = sorted(kv, key=lambda x: -x[1])[:3]
            reasons = [k for k, v in kv_sorted]
    except Exception:
        reasons = ["rapid receive-and-forward behaviour", "many incoming senders", "high network centrality"]

    ego_json = None
    candidate = GRAPHS_DIR / f"ego_example_{receiver}.json"
    if candidate.exists():
        ego_json = str(candidate)
    else:
        examples = [p for p in GRAPHS_DIR.glob("ego_example_*.json")] if GRAPHS_DIR.exists() else []
        ego_json = str(examples[0]) if examples else None

    return {
        "receiver": receiver,
        "score": proba,
        "decision": decision,
        "reasons": reasons,
        "ego_graph_json": ego_json,
        "features": receiver_feat
    }

if __name__ == "__main__":
    _ensure_model_loaded()
    tx_df = _load_transactions()
    sample_recv = tx_df["receiver"].iloc[0]
    print(score_transaction("A000000", sample_recv, 100.0))
