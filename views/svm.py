import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, make_moons, make_circles
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from core.ui import (header, theory, result, show_dl, dataset_picker, prepare_xy, plot_confusion, metric_row,
                     split, PALETTE)
from core.datasets import water_quality_data

header("Support Vector Machine Classification", "To classify data with a Support Vector Machine, compare linear, "
       "polynomial and RBF kernels, and visualise the maximum-margin boundary and support vectors.", "📐")
theory(r"""
An SVM finds the hyperplane **w·x + b = 0** that separates classes with the **maximum margin** 2/‖w‖.
- **Support vectors** are the training points lying on or inside the margin – they alone define the boundary.
- **Soft margin (C):** small C → wider margin, more misclassification allowed; large C → narrow margin, risk of overfit.
- **Kernel trick:** replaces dot products with K(x, x′) to obtain non-linear boundaries:
  linear x·x′ · polynomial (γx·x′ + r)ᵈ · RBF exp(−γ‖x − x′‖²).
- **γ (RBF):** larger γ → more wiggly boundary.

**Algorithm:** 1. Scale features. 2. Choose kernel, C, γ. 3. Solve the quadratic optimisation (SMO).
4. Classify new points by the sign of the decision function. Multi-class uses one-vs-one.
""")

mode = st.radio("Mode", ["2-D boundary explorer", "Full dataset classification"], horizontal=True)

c = st.columns(4)
kernel = c[0].selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
C = c[1].select_slider("C", [0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 1000], 1)
gamma = c[2].select_slider("γ (gamma)", ["scale", 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10], "scale")
degree = c[3].slider("Degree (poly)", 2, 6, 3, disabled=kernel != "poly")

if mode.startswith("2-D"):
    ds = st.selectbox("Toy dataset", ["Moons", "Circles", "Iris (petal length vs width)", "Linearly separable blobs"])
    rng = np.random.default_rng(0)
    if ds == "Moons":
        X, y = make_moons(300, noise=0.22, random_state=0)
    elif ds == "Circles":
        X, y = make_circles(300, noise=0.1, factor=0.45, random_state=0)
    elif ds.startswith("Iris"):
        ir = load_iris(); X, y = ir.data[:, 2:], ir.target_names[ir.target]
    else:
        X = np.r_[rng.normal([-2, -2], 1, (150, 2)), rng.normal([2, 2], 1, (150, 2))]; y = np.r_[[0] * 150, [1] * 150]
    y = np.asarray(y).astype(str)
    X = StandardScaler().fit_transform(X)
    m = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree).fit(X, y)

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .6, X[:, 0].max() + .6, 300),
                         np.linspace(X[:, 1].min() - .6, X[:, 1].max() + .6, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    cls = list(m.classes_)
    Z = np.array([cls.index(p) for p in m.predict(grid)]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8, 5))
    from matplotlib.colors import ListedColormap
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=ListedColormap(PALETTE[:len(cls)]), levels=np.arange(len(cls) + 1) - .5)
    if len(cls) == 2:
        D = m.decision_function(grid).reshape(xx.shape)
        ax.contour(xx, yy, D, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
    for i, cl in enumerate(cls):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], s=22, color=PALETTE[i], edgecolor="white", lw=0.4, label=cl)
    sv = m.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=90, facecolors="none", edgecolors="k", lw=1, label="Support vectors")
    ax.legend(fontsize=8); ax.grid(False)
    ax.set_title(f"SVM ({kernel}, C={C}, γ={gamma}) – accuracy {m.score(X, y):.3f}, {len(sv)} support vectors")
    show_dl(fig, "svm_boundary")
    if kernel == "linear" and len(cls) == 2:
        w = m.coef_[0]; b = m.intercept_[0]
        st.latex(rf"\text{{Hyperplane: }} {w[0]:.3f}\,x_1 + {w[1]:.3f}\,x_2 + ({b:.3f}) = 0,\quad "
                 rf"\text{{margin}} = 2/\lVert w\rVert = {2 / np.linalg.norm(w):.3f}")
    metric_row({"Training accuracy": m.score(X, y), "Support vectors": len(sv),
                "SV per class": " / ".join(map(str, m.n_support_))})

    st.subheader("Effect of C and γ (RBF)")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for a, (cc, gg) in zip(axes.ravel(), [(0.1, 0.5), (1, 0.5), (10, 0.5), (100, 0.5), (1, 0.1), (1, 1), (1, 5), (1, 20)]):
        mm = SVC(kernel="rbf", C=cc, gamma=gg).fit(X, y)
        Zm = np.array([cls.index(p) for p in mm.predict(grid)]).reshape(xx.shape)
        a.contourf(xx, yy, Zm, alpha=0.25, cmap=ListedColormap(PALETTE[:len(cls)]))
        for i, cl in enumerate(cls):
            a.scatter(X[y == cl, 0], X[y == cl, 1], s=5, color=PALETTE[i])
        a.set_title(f"C={cc}, γ={gg}  acc={mm.score(X, y):.2f}", fontsize=8); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    show_dl(fig, "svm_grid")
    result(f"SVM with {kernel} kernel separated the {ds} data with training accuracy {m.score(X, y):.3f} using "
           f"{len(sv)} support vectors.")

else:
    bc = load_breast_cancer(as_frame=True).frame
    bc["target"] = bc["target"].map({0: "malignant", 1: "benign"})
    wq = water_quality_data()[["pH", "TDS", "Turbidity", "Total_Hardness", "Well_Depth_m", "Dist_to_Sanitation_m",
                               "Rainfall_mm", "Source_Type", "Season", "Potability"]]
    BUILTIN = {"Breast cancer diagnosis": (bc, "target"), "Groundwater potability": (wq, "Potability")}
    name = st.selectbox("Built-in dataset", list(BUILTIN))
    df, target = dataset_picker(*BUILTIN[name], key="svm")
    X, y = prepare_xy(df, target)
    Xtr, Xte, ytr, yte = split(X, y, 0.25)
    pipe = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    metric_row({"Test accuracy": accuracy_score(yte, pred), "Train accuracy": pipe.score(Xtr, ytr),
                "Support vectors": int(pipe[-1].support_vectors_.shape[0])})
    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(4.8, 4))
        plot_confusion(yte, pred, list(pipe.classes_), ax)
        show_dl(fig, "svm_cm")
    with r:
        st.markdown("**Kernel comparison (test accuracy)**")
        rows = []
        for k in ["linear", "poly", "rbf", "sigmoid"]:
            p = make_pipeline(StandardScaler(), SVC(kernel=k, C=C)).fit(Xtr, ytr)
            rows.append({"Kernel": k, "Test accuracy": p.score(Xte, yte), "Support vectors": p[-1].n_support_.sum()})
        kt = pd.DataFrame(rows)
        st.dataframe(kt.round(4), hide_index=True, width="stretch")
        fig, ax = plt.subplots(figsize=(4.8, 2.4))
        ax.bar(kt.Kernel, kt["Test accuracy"], color=PALETTE[0]); ax.set_ylim(0, 1)
        show_dl(fig, "svm_kernels")
    if st.button("Run grid search (C × γ, 5-fold CV)"):
        gs = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                          {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]}, cv=5, n_jobs=-1)
        gs.fit(Xtr, ytr)
        tab = pd.DataFrame(gs.cv_results_["mean_test_score"].reshape(4, 4), index=[0.1, 1, 10, 100],
                           columns=["scale", 0.01, 0.1, 1])
        tab.index.name = "C \\ γ"
        st.dataframe(tab.round(4), width="stretch")
        st.write(f"Best: {gs.best_params_} → CV accuracy {gs.best_score_:.4f}, test accuracy {gs.score(Xte, yte):.4f}")
    with st.expander("Classification report"):
        st.text(classification_report(yte, pred, zero_division=0))
    result(f"SVM ({kernel} kernel, C={C}) classified the {name} data with test accuracy {accuracy_score(yte, pred):.3f}.")
