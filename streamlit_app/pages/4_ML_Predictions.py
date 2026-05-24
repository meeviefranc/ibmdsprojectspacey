"""
Page 4 — ML Predictions
Reproduces the machine-learning pipeline from spaceX_ML_PredictiveAnalysis.ipynb.
Trains Logistic Regression, SVM, Decision Tree, and KNN classifiers with
GridSearchCV and visualises results interactively.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from utils.data_loader import load_eda_data, load_ml_features

st.set_page_config(page_title="ML Predictions", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning — First Stage Landing Prediction")
st.caption("Reproduced from spaceX_ML_PredictiveAnalysis.ipynb")

# ── Load data ─────────────────────────────────────────────────────────────────
data = load_eda_data()
X_df = load_ml_features()

Y = data["Class"].to_numpy()
X = X_df.to_numpy()

# Normalise
scaler = preprocessing.StandardScaler()
X_std = scaler.fit_transform(X)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Settings")

    test_size = st.slider("Test split (%)", 10, 40, 20, 5) / 100
    random_state = st.number_input("Random seed", value=42, min_value=0, max_value=9999, step=1)

    st.subheader("Algorithms")
    use_lr = st.checkbox("Logistic Regression", value=True)
    use_svm = st.checkbox("Support Vector Machine", value=True)
    use_dt = st.checkbox("Decision Tree", value=True)
    use_knn = st.checkbox("K-Nearest Neighbours", value=True)

    run_btn = st.button("▶ Train & Evaluate", type="primary", use_container_width=True)

# ── Train ─────────────────────────────────────────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X_std, Y, test_size=test_size, random_state=int(random_state)
)

MODELS: dict = {}
if use_lr:
    MODELS["Logistic Regression"] = (
        LogisticRegression(max_iter=1000),
        {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    )
if use_svm:
    MODELS["SVM"] = (
        SVC(probability=True),
        {"C": [1, 4, 16, 64], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
    )
if use_dt:
    MODELS["Decision Tree"] = (
        DecisionTreeClassifier(),
        {"criterion": ["gini", "entropy"], "max_depth": [2, 4, 6, 8, 10], "min_samples_split": [2, 5, 10]},
    )
if use_knn:
    MODELS["KNN"] = (
        KNeighborsClassifier(),
        {"n_neighbors": [2, 4, 6, 8, 10], "weights": ["uniform", "distance"], "p": [1, 2]},
    )


@st.cache_data(show_spinner="Training models — this may take a moment…")
def _train_all(
    _X_train: np.ndarray,
    _Y_train: np.ndarray,
    _X_test: np.ndarray,
    _Y_test: np.ndarray,
    model_names: tuple,
    ts: float,
    rs: int,
) -> list[dict]:
    """Grid-search all selected models; return a list of result dicts."""
    results = []
    algo_map = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
        ),
        "SVM": (
            SVC(probability=True),
            {"C": [1, 4, 16, 64], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(),
            {"criterion": ["gini", "entropy"], "max_depth": [2, 4, 6, 8, 10], "min_samples_split": [2, 5, 10]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [2, 4, 6, 8, 10], "weights": ["uniform", "distance"], "p": [1, 2]},
        ),
    }
    for name in model_names:
        estimator, param_grid = algo_map[name]
        gs = GridSearchCV(estimator, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
        gs.fit(_X_train, _Y_train)
        best = gs.best_estimator_
        y_pred = best.predict(_X_test)
        acc = accuracy_score(_Y_test, y_pred)
        cm = confusion_matrix(_Y_test, y_pred).tolist()
        report = classification_report(_Y_test, y_pred, output_dict=True)
        results.append(
            {
                "name": name,
                "best_params": gs.best_params_,
                "cv_score": gs.best_score_,
                "test_accuracy": acc,
                "confusion_matrix": cm,
                "report": report,
                "y_pred": y_pred.tolist(),
            }
        )
    return results


# ── Show results ───────────────────────────────────────────────────────────────
if run_btn or st.session_state.get("ml_results"):
    if not MODELS:
        st.warning("Select at least one algorithm in the sidebar.")
        st.stop()

    with st.spinner("Training models…"):
        results = _train_all(
            X_train,
            Y_train,
            X_test,
            Y_test,
            tuple(MODELS.keys()),
            test_size,
            int(random_state),
        )
    st.session_state["ml_results"] = results

    # ── Summary comparison table ───────────────────────────────────────────
    st.subheader("Model Comparison")
    summary = pd.DataFrame(
        [
            {
                "Algorithm": r["name"],
                "CV Accuracy": f"{r['cv_score']*100:.2f}%",
                "Test Accuracy": f"{r['test_accuracy']*100:.2f}%",
                "Best Parameters": str(r["best_params"]),
            }
            for r in results
        ]
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Bar chart
    bar_df = pd.DataFrame(
        [{"Algorithm": r["name"], "Accuracy (%)": r["test_accuracy"] * 100} for r in results]
    )
    fig_bar = px.bar(
        bar_df.sort_values("Accuracy (%)", ascending=False),
        x="Algorithm",
        y="Accuracy (%)",
        color="Algorithm",
        title="Test Accuracy by Algorithm",
        text_auto=".2f",
        range_y=[0, 100],
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Per-model detail ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Detailed Results")

    tabs = st.tabs([r["name"] for r in results])
    for tab, result in zip(tabs, results):
        with tab:
            c1, c2 = st.columns(2)

            with c1:
                st.metric("Test Accuracy", f"{result['test_accuracy']*100:.2f}%")
                st.metric("Cross-Val Accuracy", f"{result['cv_score']*100:.2f}%")
                st.json(result["best_params"])

            with c2:
                # Confusion matrix heatmap
                cm_array = np.array(result["confusion_matrix"])
                labels = ["Did Not Land", "Landed"]
                fig_cm = ff.create_annotated_heatmap(
                    z=cm_array,
                    x=labels,
                    y=labels,
                    colorscale="Greens",
                    showscale=True,
                )
                fig_cm.update_layout(
                    title=f"Confusion Matrix — {result['name']}",
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            # Classification report
            report_df = pd.DataFrame(result["report"]).T.drop(
                columns=["support"], errors="ignore"
            )
            report_df = report_df.map(
                lambda v: f"{v:.3f}" if isinstance(v, float) else v
            )
            st.dataframe(report_df, use_container_width=True)

else:
    st.info("Configure model settings in the sidebar and press **▶ Train & Evaluate** to begin.")

    st.subheader("About the ML Pipeline")
    st.markdown(
        """
The pipeline closely mirrors the original notebook:

| Step | Detail |
|------|--------|
| **Features** | One-hot-encoded dataset (dataset_part_3) — orbit type, launch site, booster version, etc. |
| **Target** | `Class` (1 = successful first-stage landing, 0 = failure) |
| **Pre-processing** | `StandardScaler` normalisation |
| **Split** | Configurable train/test split (default 80 / 20) |
| **Hyper-parameter search** | 10-fold `GridSearchCV` |
| **Algorithms** | Logistic Regression, SVM, Decision Tree, KNN |
        """
    )
