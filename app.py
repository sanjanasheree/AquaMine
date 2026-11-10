"""
AquaMine – Data Mining & Predictive Analytics Lab
Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="AquaMine – Data Mining Lab", page_icon="💧", layout="wide")

P = lambda path, title, icon: st.Page(path, title=title, icon=icon)

pages = {
    "Home": [P("views/home.py", "Overview", "🏠")],
    "Main Project": [P("views/water_quality.py", "Groundwater Contamination Prediction", "💧")],
    "Visualisation": [
        P("views/normal_curve.py", "Normal Curve", "🔔"),
        P("views/contour.py", "Contour Plot", "🗺️"),
        P("views/histogram.py", "Histogram", "📊"),
        P("views/scatter.py", "Scatter Plot", "🔵"),
        P("views/plot3d.py", "3D Plot", "🧊"),
    ],
    "Classification": [
        P("views/decision_tree.py", "Decision Tree", "🌳"),
        P("views/loan_rf.py", "Loan Default – Random Forest", "🏦"),
        P("views/naive_bayes.py", "Naive Bayes", "🎲"),
        P("views/svm.py", "Support Vector Machine", "📐"),
    ],
    "Dimensionality Reduction": [P("views/pca_lda.py", "PCA, LDA & PCR", "🧭")],
    "Clustering & Regression": [
        P("views/kmeans.py", "Customer Grouping – K-Means", "👥"),
        P("views/mlr.py", "Multiple Linear Regression", "📈"),
    ],
}

with st.sidebar:
    st.markdown("### 💧 AquaMine")
    st.caption("Data Mining & Predictive Analytics Lab")

st.navigation(pages).run()
