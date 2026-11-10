import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from core.ui import header, theory, result, show_dl, csv_download, PALETTE
from core.datasets import water_quality_data, house_data

header("PCA, LDA & Principal Component Regression", "To reduce the dimensionality of a dataset using Principal "
       "Component Analysis (unsupervised) and Linear Discriminant Analysis (supervised), compare the projections, "
       "and use principal components for regression (PCR).", "🧭")
theory(r"""
**PCA (unsupervised).** Finds orthogonal directions of **maximum variance**.
1. Standardise X. 2. Covariance matrix Σ = XᵀX/(n−1). 3. Eigen-decompose Σ → eigenvectors (loadings) and
eigenvalues (variance). 4. Sort by eigenvalue; keep *k* components (e.g. ≥ 90 % cumulative variance).
5. Scores Z = X·W.

**LDA (supervised).** Finds directions that **maximise between-class scatter Sᴮ relative to within-class scatter Sᵂ**:
maximise J(w) = wᵀSᴮw / wᵀSᵂw → eigenvectors of Sᵂ⁻¹Sᴮ. At most (c − 1) components for c classes.
Also a classifier.

**PCR.** Regress y on the first *k* principal components instead of the raw (possibly collinear) features.
Choose *k* by cross-validated error. **PLS** is the supervised counterpart: components chosen to covary with y.
""")


def sk(loader):
    b = loader(as_frame=True)
    f = b.frame.copy()
    f["target"] = f["target"].map(dict(enumerate(b.target_names)))
    return f


wq = water_quality_data()
wq_num = wq[["pH", "EC", "TDS", "Turbidity", "Temperature", "Dissolved_Oxygen", "Total_Hardness", "Well_Depth_m",
             "Fluoride", "Arsenic", "Nitrate", "Iron", "Chloride", "Sulphate", "Total_Coliform", "Risk_Level"]]
DS = {"Iris": (sk(load_iris), "target"), "Wine": (sk(load_wine), "target"),
      "Groundwater quality (Risk level)": (wq_num, "Risk_Level")}

tab_pca, tab_lda, tab_cmp, tab_pcr = st.tabs(["PCA", "LDA", "PCA vs LDA", "PCR (regression)"])

name = st.sidebar.selectbox("Dataset for PCA / LDA", list(DS))
up = st.sidebar.file_uploader("…or upload CSV", type="csv", key="pca_up")
if up:
    df = pd.read_csv(up)
    target = st.sidebar.selectbox("Class column", df.columns, index=len(df.columns) - 1)
else:
    df, target = DS[name]
Xdf = df.drop(columns=[target]).select_dtypes("number").dropna()
y = df.loc[Xdf.index, target].astype(str).to_numpy(dtype=str)
Xs = StandardScaler().fit_transform(Xdf)
classes = sorted(set(y))


def scatter2(ax, Z, title, xl, yl):
    for i, c in enumerate(classes):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1] if Z.shape[1] > 1 else np.random.default_rng(0).normal(0, .05, m.sum()),
                   s=14, alpha=0.7, color=PALETTE[i % 8], label=c, edgecolor="white", lw=0.3)
    ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=8)


with tab_pca:
    pca = PCA().fit(Xs)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    c = st.columns(4)
    c[0].metric("Original features", Xs.shape[1])
    c[1].metric("PCs for 90 % variance", k90)
    c[2].metric("PC1 variance", f"{ev[0]:.1%}")
    c[3].metric("PC1 + PC2", f"{cum[min(1, len(cum) - 1)]:.1%}")
    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.bar(range(1, len(ev) + 1), ev * 100, color=PALETTE[0], label="Individual")
        ax.plot(range(1, len(ev) + 1), cum * 100, "o-", color=PALETTE[1], label="Cumulative")
        ax.axhline(90, ls=":", color="grey"); ax.set_xlabel("Principal component"); ax.set_ylabel("% variance")
        ax.set_title("Scree plot"); ax.legend()
        show_dl(fig, "pca_scree")
    with r:
        Z = pca.transform(Xs)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        scatter2(ax, Z[:, :2], "Data projected on PC1–PC2", f"PC1 ({ev[0]:.1%})", f"PC2 ({ev[1]:.1%})")
        show_dl(fig, "pca_scores")
    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        scatter2(ax, Z[:, :2] / np.abs(Z[:, :2]).max(0), "Biplot (scores + loadings)", "PC1", "PC2")
        for j, f in enumerate(Xdf.columns):
            ax.arrow(0, 0, pca.components_[0, j], pca.components_[1, j], color="k", alpha=0.7, head_width=0.02)
            ax.text(pca.components_[0, j] * 1.12, pca.components_[1, j] * 1.12, f, fontsize=7)
        show_dl(fig, "pca_biplot")
    with r:
        ld = pd.DataFrame(pca.components_[:min(5, len(ev))].T, index=Xdf.columns,
                          columns=[f"PC{i + 1}" for i in range(min(5, len(ev)))])
        st.markdown("**Loadings (eigenvectors)**")
        st.dataframe(ld.round(3), width="stretch")
        st.markdown("**Eigenvalues**: " + ", ".join(f"{v:.3f}" for v in pca.explained_variance_[:8]))
    out = pd.DataFrame(Z[:, :k90], columns=[f"PC{i + 1}" for i in range(k90)]).assign(Class=y)
    csv_download(out, "pca_scores")

with tab_lda:
    if len(classes) < 2:
        st.warning("LDA needs at least two classes.")
    else:
        lda = LinearDiscriminantAnalysis().fit(Xs, y)
        Zl = lda.transform(Xs)
        cv = cross_val_score(LinearDiscriminantAnalysis(), Xs, y, cv=5)
        c = st.columns(3)
        c[0].metric("Discriminant components", Zl.shape[1])
        c[1].metric("LD1 separation share", f"{lda.explained_variance_ratio_[0]:.1%}")
        c[2].metric("LDA classifier 5-fold CV acc.", f"{cv.mean():.3f}")
        l, r = st.columns(2)
        with l:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            if Zl.shape[1] >= 2:
                scatter2(ax, Zl[:, :2], "Data projected on LD1–LD2", "LD1", "LD2")
            else:
                for i, c_ in enumerate(classes):
                    ax.hist(Zl[y == c_, 0], bins=30, alpha=0.6, color=PALETTE[i], label=c_)
                ax.set_title("Projection on LD1"); ax.legend()
            show_dl(fig, "lda_proj")
        with r:
            coef = pd.DataFrame(lda.scalings_[:, :min(3, Zl.shape[1])], index=Xdf.columns,
                                columns=[f"LD{i + 1}" for i in range(min(3, Zl.shape[1]))])
            st.markdown("**Discriminant coefficients (scalings)**")
            st.dataframe(coef.round(3), width="stretch")

with tab_cmp:
    pca2 = PCA(2).fit_transform(Xs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    scatter2(axes[0], pca2, "PCA – maximises variance (ignores labels)", "PC1", "PC2")
    if len(classes) >= 2:
        lz = LinearDiscriminantAnalysis().fit_transform(Xs, y)
        scatter2(axes[1], lz[:, :2], "LDA – maximises class separation", "LD1", "LD2" if lz.shape[1] > 1 else "jitter")
    show_dl(fig, "pca_vs_lda")
    from sklearn.neighbors import KNeighborsClassifier
    rows = []
    for nm, Zc in [("Original (all features)", Xs), ("PCA (2 comps)", pca2)] + \
                  ([("LDA (≤2 comps)", lz[:, :2])] if len(classes) >= 2 else []):
        rows.append({"Representation": nm, "k-NN 5-fold CV accuracy": cross_val_score(KNeighborsClassifier(5), Zc, y, cv=5).mean()})
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    st.caption("PCA is unsupervised and keeps the most variance; LDA uses the labels and usually separates classes "
               "better with fewer dimensions.")

with tab_pcr:
    rds = st.selectbox("Regression dataset", ["Diabetes progression (sklearn)", "House prices (MLR dataset)"])
    if rds.startswith("Diabetes"):
        d = load_diabetes(as_frame=True).frame; yt = "target"
    else:
        d = house_data(); yt = "Price_Lakh"
    Xr, yr = d.drop(columns=[yt]).values, d[yt].values
    p = Xr.shape[1]
    kf = KFold(10, shuffle=True, random_state=0)
    rows = []
    for k in range(1, p + 1):
        pcr = make_pipeline(StandardScaler(), PCA(k), LinearRegression())
        pls = make_pipeline(StandardScaler(), PLSRegression(k))
        rows.append({"Components": k,
                     "PCR RMSE": np.sqrt(-cross_val_score(pcr, Xr, yr, cv=kf, scoring="neg_mean_squared_error").mean()),
                     "PLS RMSE": np.sqrt(-cross_val_score(pls, Xr, yr, cv=kf, scoring="neg_mean_squared_error").mean())})
    tb = pd.DataFrame(rows)
    ols = np.sqrt(-cross_val_score(make_pipeline(StandardScaler(), LinearRegression()), Xr, yr, cv=kf,
                                   scoring="neg_mean_squared_error").mean())
    kbest = int(tb.loc[tb["PCR RMSE"].idxmin(), "Components"])
    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(tb.Components, tb["PCR RMSE"], "o-", label="PCR")
        ax.plot(tb.Components, tb["PLS RMSE"], "s-", label="PLS")
        ax.axhline(ols, ls="--", color="grey", label=f"OLS (all features) {ols:.2f}")
        ax.set_xlabel("Number of components"); ax.set_ylabel("10-fold CV RMSE"); ax.legend()
        ax.set_title("Choosing the number of components")
        show_dl(fig, "pcr_rmse")
    with r:
        k = st.slider("Components used in PCR", 1, p, kbest)
        pcr = make_pipeline(StandardScaler(), PCA(k), LinearRegression())
        yp = cross_val_predict(pcr, Xr, yr, cv=kf)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.scatter(yr, yp, s=12, alpha=0.6, color=PALETTE[0])
        ax.plot([yr.min(), yr.max()], [yr.min(), yr.max()], "k--")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted (CV)")
        ax.set_title(f"PCR with {k} PCs: R² = {r2_score(yr, yp):.3f}")
        show_dl(fig, "pcr_fit")
    st.dataframe(tb.round(3), hide_index=True, width="stretch")
    pcr.fit(Xr, yr)
    beta = pcr[0].scale_ ** -1 * (pcr[1].components_.T @ pcr[2].coef_)
    st.markdown("**PCR coefficients mapped back to the original (unscaled) features**")
    st.dataframe(pd.DataFrame({"Feature": d.drop(columns=[yt]).columns, "Coefficient": beta}).round(4),
                 hide_index=True)

result(f"PCA reduced {Xs.shape[1]} features to {k90} components retaining ≥90 % variance; LDA produced "
       f"{min(len(classes) - 1, Xs.shape[1])} discriminant axes that separate the {len(classes)} classes; "
       f"PCR achieved its lowest CV-RMSE with {kbest} components.")
