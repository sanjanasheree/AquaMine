# AquaMine — Data Mining & Predictive Analytics Lab

An integrated Streamlit web application containing:

* **Main project – Water Quality / Groundwater Contamination Prediction**
Implemented data mining techniques and visualizations, including Normal Curve, Contour Plot, Histogram, Scatter Plot, 3D Plot, Decision Tree, Random Forest for Loan Default Prediction, Naive Bayes, Support Vector Machine (SVM), PCA, LDA, PCR, K-Means Clustering for Customer Segmentation, and Multiple Linear Regression.

---

## 1. Quick start

Requires **Python 3.9 or newer**.

```bash
# Windows: double-click run_windows.bat
# Linux / macOS:
./run_linux_mac.sh

# or manually
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. Use the sidebar to switch modules.

---

## 2. Folder structure

```
AquaMine_DataMining_Lab/
├── app.py                  # entry point + sidebar navigation
├── core/
│   ├── datasets.py         # synthetic dataset generators, BIS limits, WQI
│   └── ui.py               # shared plotting / metrics / upload helpers
├── views/
│   ├── home.py
│   ├── water_quality.py    # MAIN PROJECT
│   ├── normal_curve.py     contour.py     histogram.py   scatter.py   plot3d.py
│   ├── decision_tree.py    loan_rf.py     naive_bayes.py svm.py
│   ├── pca_lda.py          kmeans.py      mlr.py
├── data/                   # CSV copies of all built-in datasets
├── generate_data.py        # regenerates the CSVs in data/
├── requirements.txt
├── run_windows.bat / run_linux_mac.sh
└── .streamlit/config.toml  # theme
```

---

## 3. Main project – Groundwater Contamination Prediction

**Problem.** Fluoride, arsenic, nitrate and bacterial contamination of groundwater affects millions in
India. Laboratory-testing every well is slow and costly. The goal is to **predict which sources are
unsafe before testing**, so that lab capacity is spent on the riskiest sources first.

**Data (`data/groundwater_quality.csv`, 2 000 samples, 12 districts).**

| Group | Columns |
|---|---|
| Location / context | District, State, Source_Type, Land_Use, Season |
| Field-sensor readings (cheap) | pH, EC, TDS, Turbidity, Temperature, Dissolved_Oxygen, Total_Hardness, Well_Depth_m, Dist_to_Sanitation_m, Rainfall_mm |
| Laboratory results (costly) | Fluoride, Arsenic, Nitrate, Iron, Chloride, Sulphate, Total_Coliform |
| Labels | Potability (Safe/Unsafe), Risk_Level (Low/Moderate/High), Exceedance_Count, WQI |

Labels are derived from **BIS IS 10500:2012** permissible limits. A sample is *Unsafe* when any
health-critical limit is breached (F⁻ > 1.5 mg/L, As > 0.01 mg/L, NO₃⁻ > 45 mg/L, coliform detected) or two
or more limits are exceeded.

> The bundled data are **synthetic**, generated to mimic documented contamination belts (fluoride in
> Nalgonda/Anantapur/Dharmapuri, arsenic in Murshidabad/Ballia/Bhojpur, nitrate in Bathinda, salinity in
> Barmer, microbial risk in shallow open wells). They are for teaching/demonstration. Replace them with
> real survey data (CGWB, state PHED, Jal Jeevan Mission WQMIS) through **Upload CSV** in the sidebar – use
> the same column names (download the built-in CSV as a template). If your file has the lab columns but no
> labels, the app computes Potability, Risk_Level and WQI automatically.

**Tabs on the project page**

1. **Data** – preview, descriptive statistics, BIS limits table, CSV download.
2. **EDA** – parameter distributions vs BIS limits, exceedance rates, unsafe % by district,
   correlation matrix, risk by source type, coliform by season.
3. **WQI** – Weighted Arithmetic Water Quality Index, class distribution, district summary.
4. **Train & Evaluate** – choose *Sensor screening* (no lab inputs) or *Full lab model*, target
   Potability or Risk_Level; compares Random Forest, Gradient Boosting, Decision Tree, Logistic
   Regression, SVM, Naive Bayes and k-NN (accuracy, precision, recall, F1, ROC-AUC, 5-fold CV);
   confusion matrix, ROC curve, permutation feature importance.
5. **Predict** – form for a single water source (prediction + probability + rule-based flags +
   recommended action) and batch scoring of an uploaded CSV with downloadable results.
6. **Hotspot Clusters** – K-Means on contamination profiles (fluoride-dominant, arsenic-dominant,
   microbial, low-risk…) and their distribution across districts.

On the built-in data the sensor-only screening model reaches roughly 0.72 accuracy / 0.69 F1 (no fluoride
or arsenic measurement available), while the full lab model exceeds 0.95 — the gap shows the trade-off
between cost and certainty, which is the core finding of the project.

---

## 4. Lab exercises

| Page | What it demonstrates |
|---|---|
| Normal Curve | PDF with ±σ markers, shaded P(a<X<b), CDF, sample histogram vs curve, Q–Q plot, multiple curves |
| Contour Plot | Line contours with labels, filled contours with colour bar, optional 3-D surface with projected contours |
| Histogram | Generated / dataset / uploaded data, bin rules (Sturges, √n, Freedman–Diaconis, Scott), KDE, box plot, frequency table |
| Scatter Plot | Colour & size encodings, linear/quadratic fit, Pearson & Spearman, scatter-plot matrix |
| 3D Plot | Surface, wireframe, 3-D scatter, helix, 3-D bar chart – static (Matplotlib) or interactive (Plotly) |
| Decision Tree | Gini/entropy, depth & leaf controls, tree diagram, IF–THEN rules, accuracy vs depth |
| Loan Default – Random Forest | Credit-risk dataset, trees/depth/features controls, adjustable threshold, OOB score, ROC, importance, new-applicant form |
| Naive Bayes | Gaussian NB, priors, class means, likelihood curves, decision regions, worked posterior |
| SVM | 2-D explorer with margin & support vectors, C/γ grid, kernel comparison, grid search |
| PCA, LDA & PCR | Scree plot, biplot, loadings, LDA projection, PCA vs LDA, principal-component regression (with PLS comparison) |
| Customer Grouping – K-Means | Elbow, silhouette, cluster plot with centroids, silhouette plot, segment profiles & marketing actions, 3-D view |
| Multiple Linear Regression | Equation, coefficients with SE/t/p/CI, R², adj. R², RMSE, VIF, residual diagnostics, Durbin–Watson, prediction form |

Most classification pages also accept **your own CSV** (choose the target column).

---

## 5. Notes

* All random processes use fixed seeds, so results are reproducible for lab records.
* Built-in scikit-learn datasets used: Iris, Wine, Breast Cancer, Diabetes (bundled with scikit-learn, no internet needed).
* Tested with Python 3.11, Streamlit 1.64, scikit-learn 1.8, pandas 3.0.
