"""
Data loading utilities for the SpaceX Streamlit app.
Caches data to avoid repeated network requests.
"""
import pandas as pd
import streamlit as st

DATASETS = {
    "dash": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv",
    "part2": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv",
    "part3": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv",
    "geo": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv",
}


@st.cache_data(show_spinner="Loading SpaceX launch data…")
def load_dash_data() -> pd.DataFrame:
    """Dashboard dataset: per-launch records with site, class, payload."""
    df = pd.read_csv(DATASETS["dash"])
    return df


@st.cache_data(show_spinner="Loading EDA dataset…")
def load_eda_data() -> pd.DataFrame:
    """EDA / visualisation dataset (dataset_part_2)."""
    df = pd.read_csv(DATASETS["part2"])
    return df


@st.cache_data(show_spinner="Loading ML feature dataset…")
def load_ml_features() -> pd.DataFrame:
    """One-hot encoded feature matrix for ML (dataset_part_3)."""
    df = pd.read_csv(DATASETS["part3"])
    return df


@st.cache_data(show_spinner="Loading geo dataset…")
def load_geo_data() -> pd.DataFrame:
    """Geo-enriched dataset with Lat/Long columns."""
    df = pd.read_csv(DATASETS["geo"])
    df = df[["Launch Site", "Lat", "Long", "class"]]
    return df
