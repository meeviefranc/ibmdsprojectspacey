"""
SpaceX Falcon 9 Landing Prediction — Streamlit App
Home / Overview page
"""
import sys
import os

# Ensure utils/ is importable when running: streamlit run app.py
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(
    page_title="SpaceX Falcon 9 Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hero section ──────────────────────────────────────────────────────────────
st.title("🚀 SpaceX Falcon 9 — First Stage Landing Prediction")
st.caption("IBM Data Science Capstone Project · Space Y Competitive Analysis")

st.image(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud"
    "/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/lab_v2/images/landing_1.gif",
    width="stretch",
)

st.markdown(
    """
## Project Overview

SpaceX advertises Falcon 9 rocket launches at **\\$62 million** per launch, far below competitors
who charge upward of **\\$165 million**. The key differentiator is the reusability of Falcon 9's
first stage booster.

This application analyses every recorded SpaceX launch to help **Space Y** — a hypothetical
competitor — understand:

- Which launch sites have the best success rates
- How payload mass and orbit type influence landing outcomes
- Where launch sites are located and their proximity to key geographic features
- Which machine-learning model best predicts whether the first stage will land successfully
""",
    unsafe_allow_html=False,
)

# ── Quick-stat cards ──────────────────────────────────────────────────────────
from utils.data_loader import load_dash_data  # noqa: E402

try:
    df = load_dash_data()
    total = len(df)
    successes = int(df["class"].sum())
    success_rate = successes / total * 100
    sites = df["Launch Site"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Launches", total)
    c2.metric("Successful Landings", successes)
    c3.metric("Overall Success Rate", f"{success_rate:.1f}%")
    c4.metric("Launch Sites", sites)
except Exception:
    st.info("Summary statistics will appear here once data loads.")

st.subheader("Key Findings")
st.markdown(
    """
- The best ML model achieves **>83% test accuracy**, meaning it correctly predicts first-stage
  landing outcomes more than 4 out of 5 times.
- The models identify clear predictors: **KSC LC-39A** as the most reliable site (76.9% historical
  success), **FT-series boosters** as the best-performing hardware, and a **0–5,300 kg payload
  range** as the sweet spot for successful recoveries.
- The strong correlation between **flight number and success rate** also confirms that landing
  probability is not random — it's learnable from structured features.

These insights can help Space Y optimize launch parameters and site selection to maximize their chances of successful landings.

SpaceX charges $62M per launch versus competitors at $165M+. 
The savings come almost entirely from recovering and reusing the first-stage booster. 
This makes landing outcome prediction directly valuable to Space Y in two ways:

1. **Bid competitively** — if SpaceX's first stage is likely to fail on a given launch profile, their cost rises toward $165M. Space Y can use that insight to undercut them.
2. **Understand cost drivers** — the ML models reveal which features (launch site, payload mass, orbit type, booster version, flight number) most influence landing success.
"""
)

st.divider()

# ── Navigation guide ──────────────────────────────────────────────────────────
st.subheader("Navigate the app")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
### 📊 Launch Dashboard
Interactive pie chart and scatter plot — filter by launch site and payload range.
        """
    )

with col2:
    st.markdown(
        """
### 🔬 EDA Analysis
Flight-number trends, orbit distributions, payload mass, and landing outcome breakdowns.
        """
    )

with col3:
    st.markdown(
        """
### 🗺️ Map View
Folium map of all launch sites with success/failure markers and proximity distances.
        """
    )

with col4:
    st.markdown(
        """
### 🤖 ML Predictions
Train and compare Logistic Regression, SVM, Decision Tree, and KNN classifiers.
        """
    )

st.divider()
st.caption(
    "Data sourced from the SpaceX API and IBM Skills Network open datasets. "
    "Original notebooks preserved in the repository root."
)
