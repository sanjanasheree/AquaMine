import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from core.ui import header, theory, result, show_dl, PALETTE
from core.datasets import water_quality_data, loan_data, customer_data

header("Histogram", "To visualise the frequency distribution of a numerical variable using a histogram and "
       "study the effect of bin size, density scaling and KDE.", "📊")
theory("""
A **histogram** divides the range of data into **bins** and draws a bar whose height is the count
(or density) of observations in each bin. It reveals shape (symmetric / skewed / bimodal), centre,
spread and outliers.

**Bin-width rules:** Sturges k = ⌈log₂n⌉+1 · Square-root k = ⌈√n⌉ ·
Freedman–Diaconis h = 2·IQR·n^(−1/3) · Scott h = 3.49·s·n^(−1/3).

**Algorithm:** 1. Load/generate data. 2. Choose bin rule. 3. `plt.hist(data, bins)`. 4. Optionally normalise to
density and overlay a kernel-density estimate. 5. Report descriptive statistics.
""")

src = st.radio("Data", ["Generated distribution", "Column from a project dataset", "Upload CSV"], horizontal=True)
if src == "Generated distribution":
    c = st.columns(3)
    dist = c[0].selectbox("Distribution", ["Normal", "Uniform", "Exponential (right-skewed)",
                                           "Left-skewed (beta)", "Bimodal", "Log-normal"])
    n = c[1].slider("n", 50, 20000, 1000)
    rng = np.random.default_rng(c[2].number_input("Seed", 0, 9999, 3))
    data = {"Normal": lambda: rng.normal(50, 10, n), "Uniform": lambda: rng.uniform(0, 100, n),
            "Exponential (right-skewed)": lambda: rng.exponential(15, n),
            "Left-skewed (beta)": lambda: rng.beta(6, 1.6, n) * 100,
            "Bimodal": lambda: np.r_[rng.normal(35, 6, n // 2), rng.normal(70, 8, n - n // 2)],
            "Log-normal": lambda: rng.lognormal(3, 0.5, n)}[dist]()
    label = dist
elif src == "Column from a project dataset":
    ds = {"Groundwater quality": water_quality_data(), "Loan applicants": loan_data(),
          "Mall customers": customer_data()}
    c = st.columns(2)
    name = c[0].selectbox("Dataset", list(ds))
    d = ds[name]
    col = c[1].selectbox("Column", d.select_dtypes("number").columns)
    data, label = d[col].dropna().values, col
else:
    f = st.file_uploader("CSV", type="csv")
    if f is None:
        st.stop()
    d = pd.read_csv(f)
    col = st.selectbox("Column", d.select_dtypes("number").columns)
    data, label = d[col].dropna().values, col

c = st.columns(4)
rule = c[0].selectbox("Bin rule", ["Manual", "sturges", "sqrt", "fd", "scott", "auto"])
nb = c[1].slider("Bins (manual)", 3, 150, 30, disabled=rule != "Manual")
density = c[2].checkbox("Density scale", True)
kde = c[3].checkbox("Overlay KDE", True)
bins = nb if rule == "Manual" else rule

l, r = st.columns([3, 2])
with l:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts, edges, _ = ax.hist(data, bins=bins, density=density, color=PALETTE[0], alpha=0.75,
                               edgecolor="white", label="Histogram")
    if kde:
        xs = np.linspace(data.min(), data.max(), 400)
        k = stats.gaussian_kde(data)(xs)
        if not density:
            k *= len(data) * (edges[1] - edges[0])
        ax.plot(xs, k, color=PALETTE[1], lw=2, label="KDE")
    ax.axvline(np.mean(data), color="k", ls="--", lw=1, label=f"Mean {np.mean(data):.2f}")
    ax.axvline(np.median(data), color=PALETTE[2], ls=":", lw=1.5, label=f"Median {np.median(data):.2f}")
    ax.set_xlabel(label); ax.set_ylabel("Density" if density else "Frequency")
    ax.set_title(f"Histogram of {label} ({len(edges) - 1} bins)"); ax.legend(fontsize=8)
    show_dl(fig, "histogram")
with r:
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.boxplot(data, vert=False, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=PALETTE[0], alpha=0.5))
    ax.set_title("Box plot (outlier check)"); ax.set_yticks([])
    show_dl(fig, "histogram_box")

sk = stats.skew(data)
st.dataframe(pd.DataFrame({
    "Statistic": ["Count", "Mean", "Median", "Mode (bin centre)", "Std dev", "Variance", "Min", "Q1", "Q3",
                  "Max", "Range", "IQR", "Skewness", "Kurtosis"],
    "Value": [len(data), np.mean(data), np.median(data),
              (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2, np.std(data, ddof=1),
              np.var(data, ddof=1), np.min(data), *np.percentile(data, [25, 75]), np.max(data), np.ptp(data),
              stats.iqr(data), sk, stats.kurtosis(data)]}).set_index("Statistic").astype(float).round(4).T,
             width="stretch")

st.subheader("Frequency table")
ft = pd.DataFrame({"Bin": [f"{edges[i]:.2f} – {edges[i + 1]:.2f}" for i in range(len(edges) - 1)],
                   "Frequency": np.histogram(data, edges)[0]})
ft["Relative %"] = (ft.Frequency / ft.Frequency.sum() * 100).round(2)
ft["Cumulative"] = ft.Frequency.cumsum()
st.dataframe(ft, width="stretch", hide_index=True, height=240)

shape = "approximately symmetric" if abs(sk) < 0.5 else ("right (positively) skewed" if sk > 0 else
                                                       "left (negatively) skewed")
result(f"Histogram of {label} plotted with {len(edges) - 1} bins; the distribution is {shape} "
       f"(skewness = {sk:.3f}), mean = {np.mean(data):.3f}, median = {np.median(data):.3f}.")
