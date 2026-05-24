"""
Page 1 — Launch Dashboard
Mirrors the original Dash app (spaceX_IVADash_LaunchRecords.py)
using Streamlit + Plotly Express.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.express as px

from utils.data_loader import load_dash_data

st.set_page_config(page_title="Launch Dashboard", page_icon="📊", layout="wide")

st.title("📊 SpaceX Launch Records Dashboard")
st.caption("Equivalent of the original Dash interactive dashboard")

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_dash_data()
min_payload = float(df["Payload Mass (kg)"].min())
max_payload = float(df["Payload Mass (kg)"].max())

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    site_options = ["All Sites"] + sorted(df["Launch Site"].unique().tolist())
    selected_site = st.selectbox("Launch Site", site_options, index=0)

    payload_range = st.slider(
        "Payload Mass (kg)",
        min_value=0,
        max_value=10_000,
        value=(int(min_payload), int(max_payload)),
        step=500,
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
low, high = payload_range
mask_payload = (df["Payload Mass (kg)"] >= low) & (df["Payload Mass (kg)"] <= high)

if selected_site == "All Sites":
    filtered_df = df[mask_payload]
else:
    filtered_df = df[(df["Launch Site"] == selected_site) & mask_payload]

# ── KPI row ───────────────────────────────────────────────────────────────────
total = len(filtered_df)
successes = int(filtered_df["class"].sum())
rate = successes / total * 100 if total else 0.0

k1, k2, k3 = st.columns(3)
k1.metric("Launches shown", total)
k2.metric("Successful landings", successes)
k3.metric("Success rate", f"{rate:.1f}%")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
col_pie, col_scatter = st.columns([1, 2])

with col_pie:
    st.subheader("Launch Outcome")
    if selected_site == "All Sites":
        pie_df = (
            df.groupby(["Launch Site", "class"])
            .size()
            .reset_index(name="count")
        )
        # successful launches by site
        success_by_site = (
            df[df["class"] == 1]
            .groupby("Launch Site")
            .size()
            .reset_index(name="count")
        )
        fig_pie = px.pie(
            success_by_site,
            values="count",
            names="Launch Site",
            title="Total Successful Launches by Site",
            hole=0.35,
        )
    else:
        outcome_df = (
            filtered_df.groupby("class")
            .size()
            .reset_index(name="count")
        )
        outcome_df["Outcome"] = outcome_df["class"].map(
            {1: "Success", 0: "Failure"}
        )
        fig_pie = px.pie(
            outcome_df,
            values="count",
            names="Outcome",
            title=f"Launch Outcome — {selected_site}",
            color="Outcome",
            color_discrete_map={"Success": "#2ecc71", "Failure": "#e74c3c"},
            hole=0.35,
        )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_scatter:
    st.subheader("Payload Mass vs. Launch Outcome")
    scatter_title = (
        f"Payload vs. Outcome — {selected_site}"
        if selected_site != "All Sites"
        else "Payload vs. Outcome — All Sites"
    )
    fig_scatter = px.scatter(
        filtered_df,
        x="Payload Mass (kg)",
        y="class",
        color="Booster Version Category",
        title=scatter_title,
        labels={"class": "Landing Outcome (1=Success, 0=Failure)"},
        hover_data=["Launch Site", "Booster Version Category"],
    )
    fig_scatter.update_yaxes(tickvals=[0, 1], ticktext=["Failure", "Success"])
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── Findings ──────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📝 Key Observations & Findings"):
    st.markdown(
        """
- **Largest successful launch count:** KSC LC-39A accounts for ~41.7 % of all successful launches.
- **Highest site success rate:** KSC LC-39A at ≈ 76.9 % success rate.
- **Best payload range:** 0 – 5,300 kg correlates with the highest success rates.
- **Worst payload range:** 1,000 – 5,000 kg at CCAFS LC-40 shows only ~26.9 % success rate.
- **Best booster version:** FT-series boosters achieve the highest overall landing success rate.
        """
    )
