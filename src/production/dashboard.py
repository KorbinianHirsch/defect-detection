"""Production monitoring dashboard.

Reads the SQLite event log the worker writes to and shows live production
metrics: throughput, pass/fail counts, a score trend (drift proxy), and a
review queue of recently rejected parts with their heatmaps -- what a
human on the line would actually look at, instead of a per-image upload
form.

Run:
    streamlit run src/production/dashboard.py
"""

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.production.db import connect

st.set_page_config(page_title="Production Monitor", page_icon="🏭", layout="wide")
st.title("🏭 Production Line Monitor")
st.caption(
    "Simulated inline QC: parts arrive from `simulator.py`, get scored by the worker "
    "automatically, and are sorted into accepted/ or rejected/ -- no manual upload."
)

REFRESH_SECONDS = 3


def load_events() -> pd.DataFrame:
    with connect() as conn:
        df = pd.read_sql_query("SELECT * FROM events ORDER BY id ASC", conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


df = load_events()

if df.empty:
    st.info(
        "No parts processed yet. In two other terminals, run:\n\n"
        "```\npython -m src.production.worker --category bottle\n"
        "python -m src.production.simulator --category bottle\n```"
    )
    time.sleep(REFRESH_SECONDS)
    st.rerun()

total = len(df)
rejected = int(df["is_anomaly"].sum())
accepted = total - rejected

c1, c2, c3, c4 = st.columns(4)
c1.metric("Parts processed", total)
c2.metric("Accepted", accepted)
c3.metric("Rejected", rejected)
c4.metric("Avg latency", f"{df['latency_ms'].mean():.1f} ms")

st.subheader("Score trend (drift signal)")
threshold = df["threshold"].iloc[-1]
st.line_chart(df[["timestamp", "score"]].set_index("timestamp"))
st.caption(
    f"Current threshold: {threshold:.2f}. A sustained rise in the score of *accepted* parts "
    "-- even while staying under threshold -- can indicate drift (lighting change, camera "
    "aging, a new product variant) worth investigating before it crosses the line for real."
)

if df["true_label"].notna().any():
    st.subheader("Accuracy vs. ground truth")
    st.caption(
        "Only possible here because this simulation replays labeled benchmark data -- "
        "a real line wouldn't have ground truth at inference time."
    )
    df["true_anomaly"] = (df["true_label"] != "good").astype(int)
    tp = int(((df["true_anomaly"] == 1) & (df["is_anomaly"] == 1)).sum())
    fn = int(((df["true_anomaly"] == 1) & (df["is_anomaly"] == 0)).sum())
    fp = int(((df["true_anomaly"] == 0) & (df["is_anomaly"] == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("True positives", tp)
    a2.metric("Missed defects", fn)
    a3.metric("Precision", f"{precision:.2f}" if precision == precision else "n/a")
    a4.metric("Recall", f"{recall:.2f}" if recall == recall else "n/a")

st.subheader("Review queue: recently rejected parts")
recent_rejects = df[df["is_anomaly"] == 1].tail(6).iloc[::-1]
if recent_rejects.empty:
    st.write("No rejects yet.")
else:
    cols = st.columns(3)
    for i, (_, row) in enumerate(recent_rejects.iterrows()):
        with cols[i % 3]:
            sorted_path = Path(row["sorted_path"])
            heatmap_path = sorted_path.with_name(sorted_path.stem + "_heatmap.png")
            caption = f"score={row['score']:.1f} (true={row['true_label']})"
            if heatmap_path.exists():
                st.image(str(heatmap_path), caption=caption, use_container_width=True)
            else:
                st.write(f"{caption} -- no heatmap saved")

time.sleep(REFRESH_SECONDS)
st.rerun()
