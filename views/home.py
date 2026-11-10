import streamlit as st

st.title("💧 AquaMine — Data Mining & Predictive Analytics")
st.markdown(
    "An integrated platform featuring the **main project** — *Water Quality / Groundwater "
    "Contamination Prediction* — along with interactive modules for visualisation, classification, "
    "dimensionality reduction, clustering, and regression. Use the sidebar to open any module."
)

st.subheader("Main Project: Groundwater Contamination Prediction")
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown(
        """
**Problem.** Groundwater in many Indian districts carries fluoride, arsenic, nitrate and
bacterial contamination. Testing every well in a laboratory is slow and expensive.

**Approach.** A data-mining pipeline that learns, from cheap **field-sensor readings**
(pH, EC/TDS, turbidity, temperature, DO, hardness, depth, distance to sanitation, rainfall,
source type, land use, season), which sources are likely to breach **BIS IS 10500:2012** limits —
so labs can prioritise the riskiest sources first.

**What the module does**
1. Data understanding & EDA (distributions, correlations, district exceedance maps)
2. Water Quality Index (Weighted Arithmetic WQI)
3. Model training & comparison — Random Forest, Gradient Boosting, Decision Tree, Logistic Regression, SVM, Naive Bayes, k-NN
4. Evaluation — accuracy, precision, recall, F1, ROC-AUC, confusion matrix, feature importance
5. Prediction — single-source form and batch CSV scoring with download
6. Hotspot clustering of sampling locations with K-Means
        """
    )
with c2:
    st.info(
        "**Two modelling modes**\n\n"
        "• **Sensor screening** – predict safety *before* lab testing (the real use case).\n\n"
        "• **Full lab model** – uses lab results too; shows the upper bound when all data are known."
    )
    st.warning("The bundled dataset is **synthetic**, generated to mimic documented Indian "
               "contamination belts. Replace it with real data (e.g. CGWB / state PHED) via "
               "the CSV upload on the project page.")

st.subheader("Analytics & Machine Learning Modules")
mods = [
    ("🔔 Normal Curve", "PDF, CDF, shaded probability, z-scores, sample vs theoretical"),
    ("🗺️ Contour Plot", "Line & filled contours of 2-variable functions, with labels"),
    ("📊 Histogram", "Binning rules, density, KDE overlay, descriptive statistics"),
    ("🔵 Scatter Plot", "Correlation, regression line, colour/size encodings"),
    ("🧊 3D Plot", "Surface, wireframe, 3D scatter, helix — static & interactive"),
    ("🌳 Decision Tree", "Gini/entropy, depth control, tree diagram & rules"),
    ("🏦 Loan Default – Random Forest", "Credit-risk classification with prediction form"),
    ("🎲 Naive Bayes", "Gaussian NB, priors & likelihoods, decision regions"),
    ("📐 SVM", "Linear/RBF/poly kernels, C & gamma, support vectors"),
    ("🧭 PCA, LDA & PCR", "Unsupervised vs supervised projection; principal component regression"),
    ("👥 K-Means", "Customer grouping, elbow & silhouette, segment profiles"),
    ("📈 Multiple Linear Regression", "Coefficients, p-values, R², residual diagnostics"),
]
cols = st.columns(3)
for i, (t, d) in enumerate(mods):
    cols[i % 3].markdown(f"**{t}**  \n{d}")

st.divider()
st.caption("Each page follows a structured workflow: Aim → Theory/Algorithm → Interactive Output → Result. "
           "Every chart has a PNG download button and every dataset a CSV download.")
