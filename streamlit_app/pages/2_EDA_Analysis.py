"""
Page 2 — EDA Analysis
Reproduces the key visualisations from:
  • spaceX_datawranglingEDA.ipynb
  • spaceX_datavisualization.ipynb
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_eda_data

st.set_page_config(page_title="EDA Analysis", page_icon="🔬", layout="wide")

st.title("🔬 Exploratory Data Analysis")
st.caption(
    "Visual analysis from spaceX_datawranglingEDA.ipynb & spaceX_datavisualization.ipynb"
)

df = load_eda_data()

# ── Sidebar section selector ──────────────────────────────────────────────────
with st.sidebar:
    st.header("EDA Section")
    section = st.radio(
        "Choose analysis",
        [
            "Dataset Overview",
            "Flight Number Trends",
            "Launch Site Analysis",
            "Orbit Analysis",
            "Payload vs. Success",
            "Year-over-Year Trend",
        ],
    )

# ── Dataset Overview ──────────────────────────────────────────────────────────
if section == "Dataset Overview":
    st.subheader("Dataset Overview")

    r1, r2, r3 = st.columns(3)
    r1.metric("Total Records", len(df))
    r2.metric("Features", df.shape[1])
    r3.metric(
        "Missing Values",
        int(df.isnull().sum().sum()),
    )

    st.dataframe(df.head(20), width="stretch")

    st.subheader("Missing Values (% per column)")
    missing = (df.isnull().sum() / len(df) * 100).reset_index()
    missing.columns = ["Column", "Missing %"]
    missing = missing[missing["Missing %"] > 0].sort_values("Missing %", ascending=True)
    if missing.empty:
        st.success("No missing values found.")
    else:
        fig = px.bar(missing, x="Missing %", y="Column", orientation="h")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Column Types")
    dtypes_df = df.dtypes.reset_index()
    dtypes_df.columns = ["Column", "Dtype"]
    dtypes_df["Dtype"] = dtypes_df["Dtype"].astype(str)
    st.dataframe(dtypes_df, width="stretch")

# ── Flight Number Trends ──────────────────────────────────────────────────────
elif section == "Flight Number Trends":
    st.subheader("Flight Number vs. Payload Mass (coloured by outcome)")
    fig = px.scatter(
        df,
        x="FlightNumber",
        y="PayloadMass",
        color=df["Class"].map({1: "Success", 0: "Failure"}),
        color_discrete_map={"Success": "#2ecc71", "Failure": "#e74c3c"},
        labels={
            "FlightNumber": "Flight Number",
            "PayloadMass": "Payload Mass (kg)",
            "color": "Outcome",
        },
        title="Flight Number vs. Payload Mass",
    )
    st.plotly_chart(fig, width="stretch")
    st.info(
        "As flight number increases, the first stage is more likely to land successfully — "
        "reflecting SpaceX's iterative improvements over time."
    )

# ── Launch Site Analysis ──────────────────────────────────────────────────────
elif section == "Launch Site Analysis":
    st.subheader("Flight Number vs. Launch Site")
    fig1 = px.strip(
        df,
        x="FlightNumber",
        y="LaunchSite",
        color=df["Class"].map({1: "Success", 0: "Failure"}),
        color_discrete_map={"Success": "#2ecc71", "Failure": "#e74c3c"},
        labels={"FlightNumber": "Flight Number", "LaunchSite": "Launch Site", "color": "Outcome"},
        title="Flight Number vs. Launch Site",
    )
    st.plotly_chart(fig1, width="stretch")

    st.subheader("Payload Mass vs. Launch Site")
    fig2 = px.strip(
        df,
        x="PayloadMass",
        y="LaunchSite",
        color=df["Class"].map({1: "Success", 0: "Failure"}),
        color_discrete_map={"Success": "#2ecc71", "Failure": "#e74c3c"},
        labels={"PayloadMass": "Payload Mass (kg)", "LaunchSite": "Launch Site", "color": "Outcome"},
        title="Payload Mass vs. Launch Site",
    )
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Launch Count per Site")
    site_counts = df["LaunchSite"].value_counts().reset_index()
    site_counts.columns = ["Launch Site", "Count"]
    fig3 = px.bar(site_counts, x="Launch Site", y="Count", title="Launches per Site", color="Launch Site")
    st.plotly_chart(fig3, width="stretch")

# ── Orbit Analysis ────────────────────────────────────────────────────────────
elif section == "Orbit Analysis":
    st.subheader("Success Rate by Orbit Type")
    orbit_stats = (
        df.groupby("Orbit")["Class"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "Successes", "count": "Total"})
    )
    orbit_stats["Success Rate (%)"] = (orbit_stats["Successes"] / orbit_stats["Total"] * 100).round(1)
    orbit_stats = orbit_stats.sort_values("Success Rate (%)", ascending=False)

    fig = px.bar(
        orbit_stats,
        x="Orbit",
        y="Success Rate (%)",
        color="Success Rate (%)",
        color_continuous_scale="RdYlGn",
        title="Landing Success Rate per Orbit Type",
        text="Success Rate (%)",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Launch Count by Orbit")
    orbit_counts = df["Orbit"].value_counts().reset_index()
    orbit_counts.columns = ["Orbit", "Count"]
    fig2 = px.pie(orbit_counts, values="Count", names="Orbit", title="Distribution of Launches by Orbit", hole=0.3)
    st.plotly_chart(fig2, width="stretch")

# ── Payload vs. Success ───────────────────────────────────────────────────────
elif section == "Payload vs. Success":
    st.subheader("Payload Mass Distribution by Outcome")
    fig = px.box(
        df,
        x=df["Class"].map({1: "Success", 0: "Failure"}),
        y="PayloadMass",
        color=df["Class"].map({1: "Success", 0: "Failure"}),
        color_discrete_map={"Success": "#2ecc71", "Failure": "#e74c3c"},
        labels={"x": "Outcome", "PayloadMass": "Payload Mass (kg)"},
        title="Payload Mass Distribution — Success vs. Failure",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Payload Mass vs. Success (by Booster Version)")
    if "BoosterVersion" in df.columns or "Booster Version" in df.columns:
        bv_col = "BoosterVersion" if "BoosterVersion" in df.columns else "Booster Version"
        fig2 = px.scatter(
            df,
            x="PayloadMass",
            y="Class",
            color=bv_col,
            title="Payload Mass vs. Success by Booster Version",
            labels={"PayloadMass": "Payload Mass (kg)", "Class": "Outcome (1=Success)"},
        )
        st.plotly_chart(fig2, width="stretch")

# ── Year-over-Year Trend ──────────────────────────────────────────────────────
elif section == "Year-over-Year Trend":
    st.subheader("Annual Success Rate Trend")
    if "Date" in df.columns:
        df["Year"] = pd.to_datetime(df["Date"]).dt.year
    elif "date" in df.columns:
        df["Year"] = pd.to_datetime(df["date"]).dt.year
    else:
        # Derive year from FlightNumber order as a proxy
        st.warning("No date column found — showing running average by flight number instead.")
        df_sorted = df.sort_values("FlightNumber")
        df_sorted["Rolling Success"] = (
            df_sorted["Class"].expanding().mean() * 100
        )
        fig = px.line(
            df_sorted,
            x="FlightNumber",
            y="Rolling Success",
            title="Cumulative Success Rate over Flights",
            labels={"Rolling Success": "Cumulative Success Rate (%)", "FlightNumber": "Flight Number"},
        )
        st.plotly_chart(fig, width="stretch")
        st.stop()

    yearly = (
        df.groupby("Year")["Class"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "Successes", "count": "Total"})
    )
    yearly["Success Rate (%)"] = (yearly["Successes"] / yearly["Total"] * 100).round(1)

    fig = px.line(
        yearly,
        x="Year",
        y="Success Rate (%)",
        markers=True,
        title="Year-over-Year Landing Success Rate",
    )
    fig.update_traces(line_color="#2ecc71")
    st.plotly_chart(fig, width="stretch")
