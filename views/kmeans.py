import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from core.ui import header, theory, result, show_dl, csv_download, PALETTE
from core.datasets import customer_data

header("Customer Grouping using K-Means Clustering", "To segment mall customers into groups with similar "
       "income and spending behaviour using K-Means, choose k with the elbow and silhouette methods, and "
       "profile each segment.", "👥")
theory(r"""
**K-Means** partitions *n* points into *k* clusters by minimising the within-cluster sum of squares
(WCSS / inertia) = Σₖ Σ_{x∈Cₖ} ‖x − μₖ‖².

**Algorithm (Lloyd):** 1. Choose k and initialise centroids (k-means++). 2. **Assign** each point to the nearest
centroid. 3. **Update** each centroid to the mean of its points. 4. Repeat 2–3 until assignments stop changing.

**Choosing k:** *Elbow* – plot WCSS vs k, pick the bend. *Silhouette* s = (b − a)/max(a, b) ∈ [−1, 1]; higher is
better. Davies–Bouldin (lower better) and Calinski–Harabasz (higher better) are extra checks.
""")

src = st.radio("Data", ["Built-in mall customers", "Upload CSV"], horizontal=True)
if src.startswith("Built"):
    df = customer_data()
else:
    f = st.file_uploader("Customer CSV", type="csv")
    if f is None:
        st.info("Upload a CSV — using the built-in data meanwhile."); df = customer_data()
    else:
        df = pd.read_csv(f)
st.dataframe(df.head(50), width="stretch", height=200)

num = list(df.select_dtypes("number").columns)
default = [c for c in ["Annual_Income_kINR", "Spending_Score"] if c in num] or num[:2]
feats = st.multiselect("Features for clustering", [c for c in num if "id" not in c.lower()], default=default)
scale = st.checkbox("Standardise features (recommended when units differ)", value=True)
if len(feats) < 2:
    st.warning("Select at least two features."); st.stop()

X = df[feats].dropna().values
Xs = StandardScaler().fit_transform(X) if scale else X

ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km_ = KMeans(k, n_init=10, random_state=42).fit(Xs)
    inertia.append(km_.inertia_); sil.append(silhouette_score(Xs, km_.labels_))
best_k = int(ks[int(np.argmax(sil))])

l, r = st.columns(2)
with l:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(ks, inertia, "o-"); ax.set_xlabel("k"); ax.set_ylabel("WCSS (inertia)"); ax.set_title("Elbow method")
    show_dl(fig, "km_elbow")
with r:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(ks, sil, "s-", color=PALETTE[1]); ax.axvline(best_k, ls=":", color="grey")
    ax.set_xlabel("k"); ax.set_ylabel("Mean silhouette"); ax.set_title(f"Silhouette method (best k = {best_k})")
    show_dl(fig, "km_silhouette")

k = st.slider("Number of clusters k", 2, 10, best_k)
km = KMeans(k, n_init=10, random_state=42).fit(Xs)
lab = km.labels_
cent = km.cluster_centers_
cent_orig = StandardScaler().fit(X).inverse_transform(cent) if scale else cent
c = st.columns(4)
c[0].metric("Silhouette", f"{silhouette_score(Xs, lab):.3f}")
c[1].metric("Davies–Bouldin", f"{davies_bouldin_score(Xs, lab):.3f}")
c[2].metric("Calinski–Harabasz", f"{calinski_harabasz_score(Xs, lab):.0f}")
c[3].metric("Iterations", km.n_iter_)

seg = df.loc[df[feats].dropna().index].copy()
seg["Cluster"] = lab


def seg_name(row):
    if not {"Annual_Income_kINR", "Spending_Score"}.issubset(row.index):
        return ""
    inc_med, sp_med = df["Annual_Income_kINR"].median(), 50
    hi_i = row["Annual_Income_kINR"] > inc_med * 1.2
    lo_i = row["Annual_Income_kINR"] < inc_med * 0.8
    hi_s, lo_s = row["Spending_Score"] > 62, row["Spending_Score"] < 38
    if hi_i and hi_s: return "Premium / Target"
    if hi_i and lo_s: return "Careful high-earners"
    if lo_i and hi_s: return "Impulsive spenders"
    if lo_i and lo_s: return "Budget-conscious"
    return "Average / Standard"


prof = seg.groupby("Cluster")[feats].mean().round(1)
prof["Size"] = seg.groupby("Cluster").size()
prof["Share %"] = (prof["Size"] / len(seg) * 100).round(1)
prof["Segment"] = prof.apply(seg_name, axis=1)
names = {i: f"C{i}" + (f" – {prof.loc[i, 'Segment']}" if prof.loc[i, "Segment"] else "") for i in prof.index}

fx, fy = feats[0], feats[1]
l, r = st.columns([3, 2])
with l:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for i in range(k):
        m = lab == i
        ax.scatter(X[m, feats.index(fx)], X[m, feats.index(fy)], s=24, color=PALETTE[i % 8], alpha=0.75,
                   edgecolor="white", lw=0.4, label=names[i])
    ax.scatter(cent_orig[:, 0], cent_orig[:, 1], marker="X", s=220, c="k", label="Centroids")
    ax.set_xlabel(fx); ax.set_ylabel(fy); ax.set_title(f"Customer segments (k = {k})"); ax.legend(fontsize=7)
    show_dl(fig, "km_clusters")
with r:
    sv = silhouette_samples(Xs, lab)
    fig, ax = plt.subplots(figsize=(4.6, 4.8))
    yl = 0
    for i in range(k):
        s = np.sort(sv[lab == i])
        ax.fill_betweenx(np.arange(yl, yl + len(s)), 0, s, color=PALETTE[i % 8], alpha=0.8)
        ax.text(-0.08, yl + len(s) / 2, f"C{i}", fontsize=8); yl += len(s) + 8
    ax.axvline(sv.mean(), color="k", ls="--"); ax.set_yticks([]); ax.set_xlabel("Silhouette coefficient")
    ax.set_title("Silhouette plot")
    show_dl(fig, "km_silplot")

st.subheader("Segment profiles")
st.dataframe(prof, width="stretch")

if len(feats) >= 3:
    f3 = feats[:3]
    fig = go.Figure()
    for i in range(k):
        m = lab == i
        fig.add_trace(go.Scatter3d(x=X[m, 0], y=X[m, 1], z=X[m, 2], mode="markers",
                                   marker=dict(size=3.5, color=PALETTE[i % 8]), name=names[i]))
    fig.update_layout(height=520, scene=dict(xaxis_title=f3[0], yaxis_title=f3[1], zaxis_title=f3[2]),
                      margin=dict(l=0, r=0, t=30, b=0), title="3-D view of clusters (drag to rotate)")
    st.plotly_chart(fig, width="stretch")
else:
    st.caption("Tip: add Age as a third feature to see an interactive 3-D cluster view.")

if "Gender" in seg:
    st.markdown("**Gender mix by cluster (%)**")
    st.dataframe((pd.crosstab(seg.Cluster, seg.Gender, normalize="index") * 100).round(1), width="stretch")

seg["Segment"] = seg["Cluster"].map(names)
csv_download(seg, "customer_segments")
st.subheader("Marketing suggestions")
TIPS = {"Premium / Target": "Loyalty programme, premium launches, personalised offers.",
        "Careful high-earners": "Value-focused messaging, quality guarantees, exclusive previews to convert spending.",
        "Impulsive spenders": "Flash sales, EMI / buy-now-pay-later, trend-led promotions.",
        "Budget-conscious": "Discounts, combo packs, festival-season bargains.",
        "Average / Standard": "Regular promotions, cross-selling, seasonal campaigns."}
for i in prof.index:
    if prof.loc[i, "Segment"]:
        st.markdown(f"- **{names[i]}** ({prof.loc[i, 'Size']} customers): {TIPS[prof.loc[i, 'Segment']]}")

result(f"K-Means grouped {len(seg)} customers into {k} segments using {', '.join(feats)} "
       f"(silhouette = {silhouette_score(Xs, lab):.3f}). The silhouette method suggested k = {best_k}.")
