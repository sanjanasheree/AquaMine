import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from core.ui import (header, theory, result, show_dl, dataset_picker, prepare_xy, plot_confusion, metric_row,
                     decision_boundary, split, PALETTE)
from core.datasets import water_quality_data

header("Naive Bayes Classification", "To classify data using the Gaussian Naive Bayes classifier and interpret "
       "the class priors and likelihoods learned by the model.", "🎲")
theory(r"""
**Bayes' theorem:**  P(C | x) = P(x | C)·P(C) / P(x)

**Naive assumption:** features are conditionally independent given the class, so
P(x | C) = Π P(xᵢ | C).  **Gaussian NB** models each P(xᵢ | C) as a normal curve with the class mean μ and
variance σ² of that feature.

**Algorithm:** 1. Estimate prior P(C) = nᶜ/n. 2. For each class and feature compute μ and σ².
3. For a new sample compute log P(C) + Σ log N(xᵢ; μ, σ²) for every class. 4. Predict the class with
the largest posterior. Fast, needs little data, works well for text and as a baseline.
""")


def sk(loader):
    b = loader(as_frame=True)
    f = b.frame.copy()
    f["target"] = f["target"].map(dict(enumerate(b.target_names)))
    return f


wq = water_quality_data()[["pH", "TDS", "Turbidity", "Total_Hardness", "Fluoride", "Nitrate", "Well_Depth_m",
                           "Dist_to_Sanitation_m", "Potability"]]
BUILTIN = {"Iris flowers": (sk(load_iris), "target"), "Wine cultivars": (sk(load_wine), "target"),
           "Breast cancer (diagnosis)": (sk(load_breast_cancer), "target"),
           "Groundwater potability": (wq, "Potability")}
name = st.selectbox("Built-in dataset", list(BUILTIN))
df, target = dataset_picker(*BUILTIN[name], key="nb")
X, y = prepare_xy(df, target)

c = st.columns(2)
ts = c[0].slider("Test size", 0.1, 0.5, 0.3, 0.05)
vs = c[1].select_slider("var_smoothing", [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3], 1e-9)
Xtr, Xte, ytr, yte = split(X, y, ts)
nb = GaussianNB(var_smoothing=vs).fit(Xtr, ytr)
pred = nb.predict(Xte)
cv = cross_val_score(GaussianNB(var_smoothing=vs), X, y, cv=5)
metric_row({"Test accuracy": accuracy_score(yte, pred), "Train accuracy": nb.score(Xtr, ytr),
            "5-fold CV": cv.mean(), "Classes": len(nb.classes_), "Features": X.shape[1]})

l, r = st.columns(2)
with l:
    fig, ax = plt.subplots(figsize=(4.8, 4))
    plot_confusion(yte, pred, list(nb.classes_), ax)
    show_dl(fig, "nb_cm")
with r:
    st.markdown("**Class priors P(C)**")
    st.dataframe(pd.DataFrame({"Class": nb.classes_, "Prior": nb.class_prior_.round(4),
                               "Train count": nb.class_count_.astype(int)}), hide_index=True,
                 width="stretch")
    st.markdown("**Class-conditional means μ (first 8 features)**")
    st.dataframe(pd.DataFrame(nb.theta_, index=nb.classes_, columns=X.columns).iloc[:, :8].round(3),
                 width="stretch")

st.subheader("Likelihood curves P(xᵢ | C)")
feat = st.selectbox("Feature", list(X.columns))
j = list(X.columns).index(feat)
fig, ax = plt.subplots(figsize=(9, 3.2))
xs = np.linspace(X[feat].min(), X[feat].max(), 400)
for i, cl in enumerate(nb.classes_):
    ax.plot(xs, stats.norm.pdf(xs, nb.theta_[i, j], np.sqrt(nb.var_[i, j])), lw=2, color=PALETTE[i], label=cl)
    ax.hist(Xtr.loc[np.asarray(ytr) == cl, feat], bins=25, density=True, alpha=0.18, color=PALETTE[i])
ax.set_xlabel(feat); ax.set_title(f"Gaussian likelihoods of {feat} per class"); ax.legend()
show_dl(fig, "nb_likelihood")

st.subheader("Decision regions on two features")
c = st.columns(2)
fx = c[0].selectbox("Feature 1", list(X.columns), index=0)
fy = c[1].selectbox("Feature 2", list(X.columns), index=min(1, X.shape[1] - 1))
X2 = X[[fx, fy]].values
nb2 = GaussianNB().fit(X2, y)
fig, ax = plt.subplots(figsize=(7, 4.4))
idx = np.random.default_rng(0).choice(len(X2), min(800, len(X2)), replace=False)
decision_boundary(nb2, X2[idx], np.asarray(y)[idx], ax, f"Naive Bayes decision regions (acc {nb2.score(X2, y):.3f})")
ax.set_xlabel(fx); ax.set_ylabel(fy)
show_dl(fig, "nb_regions")

with st.expander("Classification report"):
    st.text(classification_report(yte, pred, zero_division=0))
with st.expander("Worked posterior for the first test sample"):
    x0 = Xte.iloc[[0]]
    post = pd.DataFrame({"Class": nb.classes_, "log prior": np.log(nb.class_prior_),
                         "log likelihood": nb.predict_joint_log_proba(x0)[0] - np.log(nb.class_prior_),
                         "Posterior P(C|x)": nb.predict_proba(x0)[0]}).round(4)
    st.dataframe(post, hide_index=True)
    st.write(f"Predicted: **{nb.predict(x0)[0]}**, actual: **{yte.iloc[0]}**")

result(f"Gaussian Naive Bayes achieved a test accuracy of {accuracy_score(yte, pred):.3f} "
       f"(5-fold CV {cv.mean():.3f} ± {cv.std():.3f}) on the {name} data.")
