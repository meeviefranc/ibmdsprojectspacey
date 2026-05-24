"""
Page 3 — Map View
Reproduces the Folium-based analysis from spaceX_dataViz_IVAFolium.ipynb
using folium + streamlit-folium.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, MousePosition
from folium.features import DivIcon
from streamlit_folium import st_folium

from utils.data_loader import load_geo_data

st.set_page_config(page_title="Map View", page_icon="🗺️", layout="wide")

st.title("🗺️ Launch Sites — Interactive Map")
st.caption("Reproduced from spaceX_dataViz_IVAFolium.ipynb")

df = load_geo_data()

launch_sites = df.groupby("Launch Site", as_index=False).first()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Map Options")
    show_clusters = st.checkbox("Cluster launch markers", value=True)
    show_distances = st.checkbox("Show distance lines to proximities", value=True)
    map_tile = st.selectbox(
        "Map tile",
        ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"],
        index=0,
    )
    selected_site_map = st.selectbox(
        "Focus on site",
        ["All Sites"] + sorted(launch_sites["Launch Site"].tolist()),
    )

# ── Build Folium map ──────────────────────────────────────────────────────────
if selected_site_map != "All Sites":
    row = launch_sites[launch_sites["Launch Site"] == selected_site_map].iloc[0]
    center = [row["Lat"], row["Long"]]
    zoom = 11
else:
    center = [29.559684888503615, -95.0830971930759]  # NASA JSC, Houston TX
    zoom = 4

site_map = folium.Map(location=center, zoom_start=zoom, tiles=map_tile)

# Add mouse position display
MousePosition().add_to(site_map)

# ── Marker colours ────────────────────────────────────────────────────────────
SUCCESS_COLOUR = "green"
FAILURE_COLOUR = "red"
SITE_COLOUR = "blue"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ── Task 1: Mark all launch sites ─────────────────────────────────────────────
for _, site_row in launch_sites.iterrows():
    site_lat = site_row["Lat"]
    site_lon = site_row["Long"]
    site_name = site_row["Launch Site"]

    # Site label
    folium.map.Marker(
        [site_lat, site_lon],
        icon=DivIcon(
            icon_size=(250, 36),
            icon_anchor=(0, 0),
            html=f'<div style="font-size:12px; font-weight:bold; color:navy;">{site_name}</div>',
        ),
    ).add_to(site_map)

    folium.Circle(
        [site_lat, site_lon],
        radius=1000,
        color=SITE_COLOUR,
        fill=True,
        fill_opacity=0.2,
        popup=site_name,
    ).add_to(site_map)

# ── Task 2: Success/failure markers for each launch ───────────────────────────
marker_layer = MarkerCluster() if show_clusters else site_map

site_filter = (
    [selected_site_map] if selected_site_map != "All Sites" else df["Launch Site"].unique().tolist()
)

for _, row in df[df["Launch Site"].isin(site_filter)].iterrows():
    colour = SUCCESS_COLOUR if row["class"] == 1 else FAILURE_COLOUR
    outcome_label = "Success" if row["class"] == 1 else "Failure"
    folium.CircleMarker(
        location=[row["Lat"], row["Long"]],
        radius=5,
        color=colour,
        fill=True,
        fill_color=colour,
        fill_opacity=0.8,
        popup=folium.Popup(
            f"<b>{row['Launch Site']}</b><br>Outcome: {outcome_label}",
            max_width=200,
        ),
    ).add_to(marker_layer)

if show_clusters:
    marker_layer.add_to(site_map)

# ── Task 3: Proximity distances (for selected or first site) ──────────────────
if show_distances:
    PROXIMITIES = {
        "Coastline (approx.)": (28.56230197, -80.57735665),
        "City (Titusville FL)": (28.61217, -80.80777),
        "Highway (US-1)": (28.56367, -80.57049),
        "Railway": (28.57206, -80.58526),
    }

    focus_site_row = (
        launch_sites[launch_sites["Launch Site"] == selected_site_map].iloc[0]
        if selected_site_map != "All Sites"
        else launch_sites.iloc[0]
    )
    s_lat, s_lon = focus_site_row["Lat"], focus_site_row["Long"]
    focus_name = focus_site_row["Launch Site"]

    for prox_name, (p_lat, p_lon) in PROXIMITIES.items():
        dist_km = _haversine_km(s_lat, s_lon, p_lat, p_lon)
        folium.PolyLine(
            locations=[[s_lat, s_lon], [p_lat, p_lon]],
            color="purple",
            weight=2,
            dash_array="5 5",
            tooltip=f"{focus_name} → {prox_name}: {dist_km:.2f} km",
        ).add_to(site_map)

        folium.Marker(
            [p_lat, p_lon],
            icon=DivIcon(
                icon_size=(220, 30),
                icon_anchor=(0, 0),
                html=f'<div style="font-size:11px; color:purple;">{prox_name} ({dist_km:.1f} km)</div>',
            ),
        ).add_to(site_map)

# ── Render map ────────────────────────────────────────────────────────────────
st_folium(site_map, width="100%", height=600)

# ── Site stats table ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Launch Site Summary")

site_stats = (
    df.groupby("Launch Site")["class"]
    .agg(Total="count", Successes="sum")
    .reset_index()
)
site_stats["Success Rate (%)"] = (
    site_stats["Successes"] / site_stats["Total"] * 100
).round(1)

st.dataframe(site_stats, use_container_width=True, hide_index=True)

with st.expander("📝 Geographic Insights"):
    st.markdown(
        """
- All four active launch sites are located near coastlines to allow safe rocket trajectories over open water.
- KSC LC-39A and CCAFS LC-40 / SLC-40 are clustered on Florida's east coast (Cape Canaveral).
- VAFB SLC-4E is on California's Pacific coast, used primarily for polar-orbit missions.
- Proximity to highways and railways facilitates logistics of heavy rocket components.
        """
    )
