import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             classification_report)
from core.ui import (header, theory, result, show_dl, csv_download, dataset_picker, prepare_xy, plot_confusion,
                     plot_roc, metric_row, split, PALETTE)
from core.datasets import loan_data

header("Loan Default Risk Prediction – Random Forest", "To predict whether a loan applicant will default using a "
       "Random Forest ensemble classifier and identify the most influential risk factors.", "🏦")
theory("""
**Random Forest** = an ensemble of many decision trees built with **bagging** and **random feature selection**.
1. Draw *B* bootstrap samples from the training data.
2. Grow a tree on each sample; at each split consider only a random subset of *m* features (√p by default).
3. Predict by **majority vote** (probability = fraction of trees voting "Yes").
4. Out-of-bag (OOB) samples give a free validation estimate.

Averaging de-correlated trees reduces variance → better generalisation than a single tree.
For credit risk, **recall on the "Yes" (default) class** matters: a missed defaulter costs more than a
rejected good applicant. The decision threshold can be lowered to catch more defaulters.
""")

df, target = dataset_picker(loan_data(), "Default", key="loan")
st.dataframe(df.head(100), width="stretch", height=220)
csv_download(df, "loan_applicants")
X, y = prepare_xy(df, target)
classes = sorted(y.unique())
pos = "Yes" if "Yes" in classes else classes[-1]

c = st.columns(6)
n_est = c[0].slider("Trees", 10, 500, 200, 10)
depth = c[1].slider("Max depth (0 = none)", 0, 30, 12)
mf = c[2].selectbox("Max features", ["sqrt", "log2", "all"])
msl = c[3].slider("Min samples leaf", 1, 20, 3)
bal = c[4].selectbox("Class weight", ["balanced", "none"])
ts = c[5].slider("Test size", 0.1, 0.4, 0.25, 0.05)


@st.cache_resource(show_spinner="Training Random Forest…")
def fit(X, y, n_est, depth, mf, msl, bal, ts):
    Xtr, Xte, ytr, yte = split(X, y, ts)
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth or None, max_features=None if mf == "all" else mf,
                                min_samples_leaf=msl, class_weight=None if bal == "none" else bal,
                                oob_score=True, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    return rf, Xtr, Xte, ytr, yte


rf, Xtr, Xte, ytr, yte = fit(X, y, n_est, depth, mf, msl, bal, ts)
proba = rf.predict_proba(Xte)
pi = list(rf.classes_).index(pos)
thr = st.slider(f"Decision threshold for '{pos}'", 0.05, 0.95, 0.5, 0.05) if len(classes) == 2 else 0.5
pred = np.where(proba[:, pi] >= thr, pos, [c for c in rf.classes_ if c != pos][0]) if len(classes) == 2 \
    else rf.predict(Xte)

avg = "binary" if len(classes) == 2 else "macro"
kw = {"pos_label": pos} if avg == "binary" else {}
metric_row({"Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred, average=avg, **kw),
            "Recall": recall_score(yte, pred, average=avg, **kw), "F1": f1_score(yte, pred, average=avg, **kw),
            "ROC-AUC": roc_auc_score(yte == pos, proba[:, pi]) if len(classes) == 2 else
            roc_auc_score(yte, proba, multi_class="ovr"),
            "OOB score": rf.oob_score_})

l, m, r = st.columns(3)
with l:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    plot_confusion(yte, pred, list(rf.classes_), ax)
    show_dl(fig, "loan_cm")
with m:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    plot_roc(yte, proba, list(rf.classes_), ax)
    show_dl(fig, "loan_roc")
with r:
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.barh(imp.index, imp.values, color=PALETTE[0]); ax.set_title("Feature importance (MDI)")
    ax.tick_params(axis="y", labelsize=7)
    show_dl(fig, "loan_importance")

l, r = st.columns(2)
with l:
    st.markdown("**Accuracy vs number of trees**")
    ns = [5, 10, 25, 50, 100, 200, 300]
    sc = [RandomForestClassifier(n, max_depth=depth or None, min_samples_leaf=msl, random_state=42, n_jobs=-1)
          .fit(Xtr, ytr).score(Xte, yte) for n in ns]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.plot(ns, sc, "o-"); ax.set_xscale("log"); ax.set_xlabel("n_estimators"); ax.set_ylabel("Test accuracy")
    show_dl(fig, "loan_trees_curve")
with r:
    st.markdown("**Risk score distribution**")
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for i, c in enumerate(rf.classes_):
        ax.hist(proba[np.asarray(yte) == c, pi], bins=30, alpha=0.6, label=f"Actual {c}", color=PALETTE[i])
    ax.axvline(thr, color="k", ls="--", label="threshold"); ax.set_xlabel(f"P({pos})"); ax.legend()
    show_dl(fig, "loan_score_dist")

with st.expander("Classification report & 5-fold cross-validation"):
    st.text(classification_report(yte, pred, zero_division=0))
    cv = cross_val_score(RandomForestClassifier(100, max_depth=depth or None, random_state=42, n_jobs=-1), X, y, cv=5)
    st.write(f"5-fold CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}  ({', '.join(f'{v:.3f}' for v in cv)})")

if target == "Default" and set(["Credit_Score", "Annual_Income"]).issubset(df.columns):
    st.subheader("Check a new applicant")
    with st.form("loan_form"):
        c = st.columns(4)
        v = dict(Age=c[0].number_input("Age", 21, 70, 35),
                 Annual_Income=c[1].number_input("Annual income (₹)", 50000, 10_000_000, 600000, 10000),
                 Employment_Years=c[2].number_input("Employment years", 0.0, 45.0, 5.0),
                 Credit_Score=c[3].number_input("Credit score", 300, 900, 700),
                 Loan_Amount=c[0].number_input("Loan amount (₹)", 10000, 50_000_000, 800000, 10000),
                 Loan_Term_Months=c[1].selectbox("Term (months)", [12, 24, 36, 60, 84], 2),
                 Interest_Rate=c[2].number_input("Interest rate %", 5.0, 30.0, 12.5),
                 Debt_to_Income=c[3].number_input("Debt-to-income", 0.0, 1.0, 0.3),
                 Num_Credit_Lines=c[0].number_input("Credit lines", 1, 20, 4),
                 Previous_Default=c[1].selectbox("Previous default", [0, 1]),
                 Loan_Purpose=c[2].selectbox("Purpose", sorted(df.Loan_Purpose.unique())),
                 Home_Ownership=c[3].selectbox("Home ownership", sorted(df.Home_Ownership.unique())))
        ok = st.form_submit_button("Assess risk", type="primary")
    if ok:
        row = pd.get_dummies(pd.DataFrame([v])).reindex(columns=X.columns, fill_value=0).astype(float)
        p = rf.predict_proba(row)[0, pi]
        band = "High" if p >= 0.6 else "Medium" if p >= thr else "Low"
        (st.error if p >= thr else st.success)(
            f"### Default probability: {p:.1%} → **{'Likely to DEFAULT' if p >= thr else 'Likely to REPAY'}** "
            f"(risk band: {band})")

result(f"Random Forest with {n_est} trees predicted loan default with accuracy {accuracy_score(yte, pred):.3f} and "
       f"recall {recall_score(yte, pred, average=avg, **kw):.3f} at threshold {thr:.2f}; the most important factor was "
       f"**{pd.Series(rf.feature_importances_, index=X.columns).idxmax()}**.")
