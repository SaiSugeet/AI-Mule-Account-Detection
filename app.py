# app/streamlit_app.py (polished version)
import streamlit as st
import pandas as pd
import json
import os
import tempfile
from pathlib import Path
from Scoring import score_transaction    # your Scoring.py at project root
from pyvis.network import Network
import streamlit.components.v1 as components
from functools import lru_cache
import math

# ---------------- page config ----------------
st.set_page_config(page_title="Mule Guard Demo", layout="wide", initial_sidebar_state="expanded")

BASE = Path(".")
RAW = BASE / "raw_data"
MODEL = BASE / "model"
GRAPHS = BASE / "graphs"

# ---------------- Theme CSS snippets ----------------
THEMES = {
    "Light": {
        "bg": "#f7fafc",
        "card": "#ffffff",
        "text": "#0f172a",
        "accent": "#0ea5a4"
    },
    "Dark": {
        "bg": "#0b1220",
        "card": "#0f172a",
        "text": "#e6eef6",
        "accent": "#7c3aed"
    },
    "Teal Accent": {
        "bg": "#e6fffa",
        "card": "#ffffff",
        "text": "#064e3b",
        "accent": "#06b6d4"
    }
}

def apply_theme(theme_name):
    t = THEMES.get(theme_name, THEMES["Light"])
    css = f"""
    <style>
    .stApp {{ background: {t['bg']}; color: {t['text']}; }}
    .card {{ background: {t['card']}; padding: 14px; border-radius: 10px; box-shadow: 0 6px 18px rgba(15,23,42,0.08); }}
    .muted {{ color: #6b7280; }}
    .accent {{ color: {t['accent']}; font-weight: 700; }}
    .score-bar {{ height: 18px; border-radius: 9px; background: linear-gradient(90deg, {t['accent']}, #ef4444); }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ---------------- sidebar controls ----------------
st.sidebar.title("Demo Controls")
theme_choice = st.sidebar.radio("Theme", ["Light", "Dark", "Teal Accent"])
apply_theme(theme_choice)

# load labels if present
try:
    labels_df = pd.read_csv(RAW / "account_labels.csv")
    mule_ids = labels_df[labels_df["is_mule"]==1]["account_id"].astype(str).tolist()
    normal_ids = labels_df[labels_df["is_mule"]==0]["account_id"].astype(str).tolist()
except Exception:
    mule_ids, normal_ids = [], []

st.sidebar.markdown("### Presets")
preset = st.sidebar.selectbox("Pick a preset", ["-- manual --", "Example: normal (A000002)"] + ([f"MULE: {m}" for m in mule_ids[:8]] if mule_ids else []))

st.sidebar.markdown("---")
st.sidebar.markdown("### Decision thresholds")
allow_t = st.sidebar.slider("Allow (score <)", 0.0, 0.6, 0.30, 0.01)
otp_t = st.sidebar.slider("OTP (score <)", allow_t + 0.01, 1.0, 0.65, 0.01)
st.sidebar.markdown("Live thresholds only affect the UI — policy.json remains unchanged unless you save it.")
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick run")
default_sender = "A000001"
default_receiver = "A000002"
if preset and preset.startswith("MULE: "):
    default_receiver = preset.split(": ")[1]
elif preset.startswith("Example: normal"):
    default_receiver = "A000002"

sender = st.sidebar.text_input("Sender", value=default_sender)
receiver = st.sidebar.text_input("Receiver", value=default_receiver)
amount = st.sidebar.number_input("Amount (₹)", value=1000.0, step=100.0)
run_btn = st.sidebar.button("Score transaction")

# ---------------- helpers ----------------
def decision_from_score(score, allow=0.3, otp=0.65):
    if score < allow:
        return "ALLOW"
    if score < otp:
        return "OTP"
    return "HOLD"

def badge_html(decision):
    color = {"ALLOW":"#16a34a","OTP":"#f59e0b","HOLD":"#dc2626"}.get(decision,"#6b7280")
    return f"<div style='display:inline-block;padding:8px 12px;border-radius:8px;background:{color};color:white;font-weight:700'>{decision}</div>"

# small LRU cache for scoring to speed demo interactions
@lru_cache(maxsize=256)
def cached_score(sender, receiver, amount):
    return score_transaction(sender, receiver, float(amount))

# read feature importances if available
def load_top_features(n=10):
    fi_path = MODEL / "feature_importances.csv"
    if fi_path.exists():
        df = pd.read_csv(fi_path).sort_values("importance", ascending=False).head(n)
        return df
    return None

# ---------------- layout ----------------
st.markdown("<div class='card'><h2 style='margin:0'>Mule Guard — Demo Dashboard</h2><p class='muted' style='margin:0'>Simulated UPI transaction scoring • RandomForest model</p></div>", unsafe_allow_html=True)
st.write("")

col_main, col_side = st.columns([2.5, 1])

with col_main:
    st.subheader("Transaction simulator")
    st.write("Simulate a transaction and score the receiver account. Use presets for fast demo flows.")

    if run_btn:
        try:
            out = cached_score(sender.strip(), receiver.strip(), amount)
        except Exception as e:
            st.error("Scoring error: " + str(e))
            out = None

        if out:
            score = float(out.get("score", 0.0))
            decision = decision_from_score(score, allow=allow_t, otp=otp_t)
            # top row: score + badge + reasons
            c1, c2, c3 = st.columns([2,1,2])
            with c1:
                st.markdown("<div style='font-size:14px;color:gray'>Risk score</div>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='margin:0'>{score:.3f}</h2>", unsafe_allow_html=True)
                # visual bar (width proportional to score)
                pct = max(0.0, min(1.0, score))
                w = int(pct * 100)
                bar_html = f"<div style='background:#e5e7eb;border-radius:9px;padding:3px;'><div style='width:{w}%;background:linear-gradient(90deg, #06b6d4, #ef4444);height:18px;border-radius:6px'></div></div>"
                st.markdown(bar_html, unsafe_allow_html=True)
            with c2:
                st.markdown("<div style='font-size:14px;color:gray'>Decision</div>", unsafe_allow_html=True)
                st.markdown(badge_html(decision), unsafe_allow_html=True)
            with c3:
                st.markdown("<div style='font-size:14px;color:gray'>Top reasons</div>", unsafe_allow_html=True)
                for r in out.get("reasons", []):
                    st.write("- " + r)

            st.markdown("#### Receiver account features (snapshot)")
            feat_df = pd.DataFrame([out["features"]]).T.rename(columns={0: "value"})
            st.dataframe(feat_df, height=320)

            # Ego graph (try to render if exists)
            ego_path = out.get("ego_graph_json")
            if ego_path and os.path.exists(ego_path):
                st.markdown("#### Network snapshot (ego graph)")
                try:
                    with open(ego_path, "r", encoding="utf-8") as f:
                        ego_json = json.load(f)
                    net = Network(height="420px", width="100%", directed=True)
                    for n in ego_json.get("nodes", []):
                        nid = n.get("id")
                        is_mule = n.get("is_mule", 0)
                        color = "red" if is_mule==1 else "lightblue"
                        net.add_node(nid, label=nid, title=str(n), color=color)
                    for e in ego_json.get("edges", []):
                        net.add_edge(e.get("source"), e.get("target"))
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                    net.save_graph(tmp.name)
                    html = open(tmp.name, 'r', encoding='utf-8').read()
                    components.html(html, height=460, scrolling=True)
                except Exception as e:
                    st.write("Could not render graph:", e)
            else:
                st.info("No ego graph snapshot found for this receiver (optional).")

with col_side:
    st.subheader("Inspector & model info")
    fi = load_top_features(10)
    if fi is not None:
        st.markdown("**Top features**")
        for _, row in fi.iterrows():
            nm = row["feature"]
            val = row["importance"]
            st.write(f"- **{nm}** — {val:.3f}")
    else:
        st.write("No feature_importances.csv found.")

    st.markdown("---")
    st.markdown("**Quick scenario**")
    if mule_ids:
        if st.button("Score first mule"):
            sample = mule_ids[0]
            sample_out = cached_score(sender, sample, amount)
            st.write(f"Scored mule: {sample}")
            st.write(sample_out)

st.markdown("---")
st.caption("Created by: Chai-fi")
