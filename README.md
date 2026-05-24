<div align="center">

# 🚀 Winning the Space Race with Data Science

### IBM Data Science Capstone Project

![Falcon 9 Landing](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/lab_v2/images/landing_1.gif)

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.22+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![Folium](https://img.shields.io/badge/Folium-0.17+-77B829?style=for-the-badge&logo=leaflet&logoColor=white)](https://python-visualization.github.io/folium/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

</div>

---

## 📖 Project Overview

As a data scientist at the fictional startup **Space Y**, I was tasked with analysing every historical **SpaceX Falcon 9** launch to build a competitive intelligence tool. SpaceX can offer launches at just **\$62 million** — versus the industry average of **\$165 million** — largely because they recover and reuse Falcon 9's first-stage booster.

This project answers one critical question:

> **Can we predict whether Falcon 9's first stage will land successfully — and therefore estimate launch cost?**

The full pipeline covers data collection, wrangling, SQL analysis, geospatial visualisation, and a multi-model machine learning comparison, all surfaced through a modern **Streamlit** web application.

### Why Machine Learning?

SpaceX charges **$62M** per launch versus competitors at **$165M+**. The savings come almost entirely from recovering and reusing the first-stage booster. This makes landing outcome prediction directly valuable to Space Y in two ways:

1. **Bid competitively** — if SpaceX's first stage is likely to fail on a given launch profile, their cost rises toward $165M. Space Y can use that insight to undercut them.
2. **Understand cost drivers** — the ML models reveal which features (launch site, payload mass, orbit type, booster version, flight number) most influence landing success.

Four algorithms — Logistic Regression, SVM, Decision Tree, and KNN — are compared because no single model is guaranteed to fit best. GridSearchCV finds the optimal hyperparameters for each, and the winner is selected based on test accuracy and confusion matrix analysis. The confusion matrix is especially important here: a **false positive** (predicting a successful landing that fails) carries real financial consequences for any competitive bid.

---

## ✨ Live App — Streamlit UI

The project uses Streamlit UI to demonstrate the analysis results in a **multi-page Streamlit application**.

| Page | Description |
|------|-------------|
| 🏠 **Home** | Project overview with live KPI metrics (total launches, success rate, sites) |
| 📊 **Launch Dashboard** | Interactive pie chart & scatter plot — filter by site and payload range |
| 🔬 **EDA Analysis** | 6-section exploratory analysis: flight trends, orbit types, payload distributions, year-over-year success rate |
| 🗺️ **Map View** | Folium map with clustered success/failure markers, site labels, and proximity distance lines |
| 🤖 **ML Predictions** | Train & compare Logistic Regression, SVM, Decision Tree, and KNN with GridSearchCV; view confusion matrices and classification reports |

### Running the app

```bash
cd streamlit_app
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🗂️ Repository Structure

```
📦 ibmdsprojectspace
├── 📓 spaceX_dataCollection.ipynb          # SpaceX API data collection & wrangling
├── 📓 spaceX_webscraping.ipynb             # BeautifulSoup scraping of Falcon 9 wiki
├── 📓 spaceX_datawranglingEDA.ipynb        # EDA & training label engineering
├── 📓 spaceX_EDAWithSQL.ipynb             # SQL-based analysis via IBM Db2
├── 📓 spaceX_datavisualization.ipynb       # Feature engineering & matplotlib/seaborn plots
├── 📓 spaceX_dataViz_IVAFolium.ipynb       # Interactive geospatial maps with Folium
├── 📓 spaceX_ML_PredictiveAnalysis.ipynb   # ML pipeline: LR, SVM, Decision Tree, KNN
├── 🐍 spaceX_IVADash_LaunchRecords.py      # Original Dash dashboard (preserved)
│
└── 📁 streamlit_app/                       # ✨ Modern Streamlit UI
    ├── app.py                              # Home page entry point
    ├── requirements.txt                    # Pinned dependencies
    ├── .streamlit/
    │   └── config.toml                     # Theme & server configuration
    ├── pages/
    │   ├── 1_Launch_Dashboard.py           # Interactive launch records dashboard
    │   ├── 2_EDA_Analysis.py               # Exploratory data analysis
    │   ├── 3_Map_View.py                   # Folium geospatial map
    │   └── 4_ML_Predictions.py            # Machine learning model comparison
    └── utils/
        └── data_loader.py                  # Cached dataset loaders
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.14 |
| **Web UI** | Streamlit 1.35+, streamlit-folium |
| **Data** | Pandas 3.0, NumPy |
| **Visualisation** | Plotly Express, Folium |
| **Machine Learning** | scikit-learn (LR, SVM, Decision Tree, KNN, GridSearchCV) |
| **Original Dashboard** | Dash, Plotly |
| **Data Sources** | SpaceX REST API v4, IBM Skills Network open datasets |

---

## 📊 Key Findings

- **KSC LC-39A** accounts for ~41.7% of all successful launches with a **76.9% success rate** — the highest of any site.
- Success rate correlates strongly with **flight number** — SpaceX consistently improved over time.
- Optimal payload range for success is **0 – 5,300 kg**.
- **FT-series boosters** achieve the highest overall landing success rate.
- All four active launch sites sit on coastlines, enabling safe over-ocean trajectories.
- The best ML model achieves **>83% test accuracy** in predicting first-stage landing outcome.

---

## 📓 Original Notebooks

| Notebook | Description |
|----------|-------------|
| `spaceX_dataCollection` | REST API requests to SpaceX v4, data normalisation, helper functions for booster/launchpad/payload/core enrichment |
| `spaceX_webscraping` | BeautifulSoup scraping of the Falcon 9 Wikipedia launch history table into a Pandas DataFrame |
| `spaceX_datawranglingEDA` | Outcome classification, training label creation (`Class` column), missing-value analysis |
| `spaceX_EDAWithSQL` | IBM Db2 SQL queries — launch counts, payload statistics, success rates per orbit and site |
| `spaceX_datavisualization` | Feature engineering, `catplot` and scatter visualisations with seaborn and matplotlib |
| `spaceX_dataViz_IVAFolium` | Folium `MarkerCluster`, `MousePosition`, `DivIcon` — launch site mapping and proximity distance calculations |
| `spaceX_ML_PredictiveAnalysis` | `StandardScaler`, `train_test_split`, 10-fold `GridSearchCV` across LR / SVM / Decision Tree / KNN |
| `spaceX_IVADash_LaunchRecords` | Original interactive Dash app with dropdown, range slider, pie chart, and scatter callbacks |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ibmdsprojectspace.git
cd ibmdsprojectspace

# (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
cd streamlit_app
pip install -r requirements.txt

# Launch the app
python -m streamlit run app.py
```

---

## 📜 License

This project was completed as part of the **IBM Data Science Professional Certificate** capstone on Coursera. Data is sourced from the [SpaceX REST API](https://github.com/r-spacex/SpaceX-API) and IBM Skills Network open datasets.

---

<div align="center">
Made with ❤️ and Python
</div>
