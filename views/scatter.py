import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from core.ui import header, theory, result, show_dl, PALETTE
from core.datasets import water_quality_data, loan_data, customer_data, house_data

header("Scatter Plot", "To study the relationship between two numerical variables using a scatter plot, "
       "the correlation coefficient and a fitted regression line.", "🔵")
theory("""
A **scatter plot** places each observation at (x, y). Pattern → relationship:
upward = positive, downward = negative, no pattern = no linear relationship, curve = non-linear.

**Pearson r** = cov(x, y)/(σₓσᵧ), −1 ≤ r ≤ 1 · **Spearman ρ** = rank correlation (monotonic).
Least-squares line: ŷ = b₀ + b₁x, b₁ = r·σᵧ/σₓ.

**Algorithm:** 1. Load data. 2. Select X, Y (and optional colour / size variables). 3. `plt.scatter`.
4. Fit `numpy.polyfit` line; compute r, ρ, R². 5. Optional scatter-matrix for all pairs.
""")

DS = {"House prices": house_data(), "Mall customers": customer_data(),
      "Groundwater quality": water_quality_data(), "Loan applicants": loan_data()}
c = st.columns(5)
name = c[0].selectbox("Dataset", list(DS))
df = DS[name]
num = list(df.select_dtypes("number").columns)
cat = [None] + [x for x in df.columns if df[x].nunique() <= 12]
xcol = c[1].selectbox("X", num, index=0)
ycol = c[2].selectbox("Y", num, index=len(num) - 1)
hue = c[3].selectbox("Colour by", cat, index=0)
size = c[4].selectbox("Size by", [None] + num)

c = st.columns(3)
fit = c[0].selectbox("Fit", ["Linear", "Quadratic", "None"])
alpha = c[1].slider("Point opacity", 0.1, 1.0, 0.6)
nmax = c[2].slider("Max points shown", 100, len(df), min(1500, len(df)))
d = df.sample(nmax, random_state=0) if nmax < len(df) else df
x, y = d[xcol].values.astype(float), d[ycol].values.astype(float)
s = 15 if size is None else 10 + 120 * (d[size] - d[size].min()) / (np.ptp(d[size]) or 1)

fig, ax = plt.subplots(figsize=(8, 4.8))
if hue:
    for i, g in enumerate(sorted(d[hue].astype(str).unique())):
        m = d[hue].astype(str) == g
        ax.scatter(x[m], y[m], s=s if np.isscalar(s) else s[m], alpha=alpha, color=PALETTE[i % 8],
                   edgecolor="white", lw=0.3, label=g)
    ax.legend(title=hue, fontsize=8)
else:
    ax.scatter(x, y, s=s, alpha=alpha, color=PALETTE[0], edgecolor="white", lw=0.3)
if fit != "None":
    deg = 1 if fit == "Linear" else 2
    co = np.polyfit(x, y, deg)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, np.polyval(co, xs), color="k", lw=2,
            label=("ŷ = " + (f"{co[0]:.4f}x + {co[1]:.3f}" if deg == 1 else
                              f"{co[0]:.4g}x² + {co[1]:.4g}x + {co[2]:.3f}")))
    ax.legend(fontsize=8)
    r2 = 1 - np.sum((y - np.polyval(co, x)) ** 2) / np.sum((y - y.mean()) ** 2)
else:
    r2 = np.nan
ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(f"{ycol} vs {xcol}")
show_dl(fig, "scatter")

r, pr = stats.pearsonr(x, y)
rho, _ = stats.spearmanr(x, y)
c = st.columns(4)
c[0].metric("Pearson r", f"{r:.3f}")
c[1].metric("p-value", f"{pr:.2e}")
c[2].metric("Spearman ρ", f"{rho:.3f}")
c[3].metric("R² of fit", "–" if np.isnan(r2) else f"{r2:.3f}")

if st.checkbox("Show scatter-plot matrix (pair plot)"):
    cols = st.multiselect("Variables", num, default=num[:4])
    if len(cols) >= 2:
        k = len(cols)
        fig, axes = plt.subplots(k, k, figsize=(2.2 * k, 2.2 * k))
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                ax = axes[i, j]
                if i == j:
                    ax.hist(d[a], bins=25, color=PALETTE[0])
                else:
                    ax.scatter(d[b], d[a], s=4, alpha=0.4, color=PALETTE[0])
                ax.set_xticks([]); ax.set_yticks([])
                if i == k - 1: ax.set_xlabel(b, fontsize=7)
                if j == 0: ax.set_ylabel(a, fontsize=7)
        fig.tight_layout()
        show_dl(fig, "scatter_matrix")

strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
direction = "positive" if r > 0 else "negative"
result(f"The scatter plot of {ycol} against {xcol} shows a {strength} {direction} linear relationship "
       f"(Pearson r = {r:.3f}, Spearman ρ = {rho:.3f}).")
