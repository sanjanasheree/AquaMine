import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from core.ui import (split, header, theory, result, show_dl, dataset_picker, prepare_xy, plot_confusion, metric_row,
                     PALETTE)
from core.datasets import water_quality_data, loan_data

header("Decision Tree Classifier", "To build a decision-tree classifier, visualise the tree, extract its IF–THEN "
       "rules and evaluate its accuracy.", "🌳")
theory(r"""
A decision tree recursively splits the data on the attribute/threshold that gives the **purest** child nodes.

- **Entropy** H(S) = −Σ pᵢ log₂ pᵢ · **Information gain** = H(S) − Σ (|Sᵥ|/|S|)·H(Sᵥ)  (ID3 / C4.5)
- **Gini index** G(S) = 1 − Σ pᵢ²  (CART, used by scikit-learn)

**Algorithm (CART):** 1. Start with all training samples at the root. 2. For every feature and threshold compute
the impurity decrease. 3. Split on the best one. 4. Repeat for each child until max depth / minimum samples /
pure node. 5. Leaf label = majority class. **Pruning** (max_depth, min_samples_leaf, ccp_alpha) controls overfitting.
""")

iris = load_iris(as_frame=True).frame
iris["target"] = iris["target"].map(dict(enumerate(load_iris().target_names)))
wq = water_quality_data()[["pH", "TDS", "Turbidity", "Total_Hardness", "Well_Depth_m", "Dist_to_Sanitation_m",
                           "Source_Type", "Season", "Potability"]]
BUILTIN = {"Iris flowers": (iris, "target"), "Groundwater potability (sensor features)": (wq, "Potability"),
           "Loan default": (loan_data(), "Default")}
name = st.selectbox("Built-in dataset", list(BUILTIN))
df, target = dataset_picker(*BUILTIN[name], key="dt")
X, y = prepare_xy(df, target)

c = st.columns(5)
crit = c[0].selectbox("Criterion", ["gini", "entropy", "log_loss"])
depth = c[1].slider("Max depth", 1, 15, 3)
mss = c[2].slider("Min samples split", 2, 50, 2)
msl = c[3].slider("Min samples leaf", 1, 30, 1)
ts = c[4].slider("Test size", 0.1, 0.5, 0.3, 0.05)

Xtr, Xte, ytr, yte = split(X, y, ts)
clf = DecisionTreeClassifier(criterion=crit, max_depth=depth, min_samples_split=mss, min_samples_leaf=msl,
                             random_state=42).fit(Xtr, ytr)
pred = clf.predict(Xte)
acc_tr, acc_te = accuracy_score(ytr, clf.predict(Xtr)), accuracy_score(yte, pred)
metric_row({"Train accuracy": acc_tr, "Test accuracy": acc_te, "Leaves": int(clf.get_n_leaves()),
            "Depth": int(clf.get_depth())})

w = min(30, 4 + 2.2 * clf.get_n_leaves())
fig, ax = plt.subplots(figsize=(max(10, w), 2 + 1.6 * clf.get_depth()))
plot_tree(clf, feature_names=list(X.columns), class_names=list(clf.classes_), filled=True, rounded=True,
          fontsize=7 if clf.get_n_leaves() < 20 else 5, ax=ax, impurity=True, proportion=False)
ax.set_title(f"Decision tree ({crit}, depth={clf.get_depth()})")
show_dl(fig, "decision_tree")

l, r = st.columns(2)
with l:
    fig, ax = plt.subplots(figsize=(4.8, 4))
    plot_confusion(yte, pred, list(clf.classes_), ax)
    show_dl(fig, "dt_confusion")
with r:
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.barh(imp.index, imp.values, color=PALETTE[0]); ax.set_title("Feature importance (impurity decrease)")
    show_dl(fig, "dt_importance")

st.subheader("Effect of tree depth (overfitting check)")
depths = range(1, 16)
tr_s, te_s = [], []
for d in depths:
    m = DecisionTreeClassifier(criterion=crit, max_depth=d, random_state=42).fit(Xtr, ytr)
    tr_s.append(m.score(Xtr, ytr)); te_s.append(m.score(Xte, yte))
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(depths, tr_s, "o-", label="Train"); ax.plot(depths, te_s, "s-", label="Test")
ax.axvline(depth, color="grey", ls=":"); ax.set_xlabel("max_depth"); ax.set_ylabel("Accuracy"); ax.legend()
best_d = int(np.argmax(te_s) + 1)
ax.set_title(f"Accuracy vs depth (best test accuracy at depth {best_d})")
show_dl(fig, "dt_depth_curve")

with st.expander("IF–THEN rules extracted from the tree"):
    st.code(export_text(clf, feature_names=list(X.columns), max_depth=depth), language="text")
with st.expander("Classification report"):
    st.text(classification_report(yte, pred, zero_division=0))

result(f"A decision tree ({crit}, depth {clf.get_depth()}, {clf.get_n_leaves()} leaves) was built on the "
       f"{name if 'Upload' not in st.session_state.get('dt_src', '') else 'uploaded'} data with a test accuracy "
       f"of {acc_te:.3f} (train {acc_tr:.3f}). Best generalisation was observed at depth {best_d}.")
