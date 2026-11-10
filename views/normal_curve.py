import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from core.ui import header, theory, result, show_dl, PALETTE

header("Normal (Gaussian) Curve", "To plot the normal distribution curve, shade the probability between two "
       "values, and compare it with a random sample drawn from the same distribution.", "🔔")
theory(r"""
The normal probability density function is
$$f(x)=\frac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
- μ (mean) sets the centre, σ (standard deviation) sets the spread.
- **Empirical rule:** ≈68 % of values lie in μ±1σ, 95 % in μ±2σ, 99.7 % in μ±3σ.
- **z-score:** z = (x − μ)/σ ; P(a < X < b) = Φ(z_b) − Φ(z_a).

**Algorithm:** 1. Read μ, σ. 2. Generate x in μ±4σ. 3. Compute f(x) with `scipy.stats.norm.pdf`.
4. Plot the curve and shade P(a<X<b). 5. Draw a sample, plot its histogram and overlay the PDF.
""")

c = st.columns(4)
mu = c[0].number_input("Mean μ", value=50.0)
sigma = c[1].number_input("Std deviation σ", value=10.0, min_value=0.1)
a = c[2].number_input("Lower bound a", value=mu - sigma)
b = c[3].number_input("Upper bound b", value=mu + sigma)
a, b = min(a, b), max(a, b)

x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 600)
pdf = stats.norm.pdf(x, mu, sigma)
p = stats.norm.cdf(b, mu, sigma) - stats.norm.cdf(a, mu, sigma)

l, r = st.columns(2)
with l:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(x, pdf, lw=2.2, color=PALETTE[0])
    m = (x >= a) & (x <= b)
    ax.fill_between(x[m], pdf[m], alpha=0.35, color=PALETTE[1], label=f"P({a:.1f} < X < {b:.1f}) = {p:.4f}")
    for k in (-3, -2, -1, 0, 1, 2, 3):
        ax.axvline(mu + k * sigma, color="grey", lw=0.7, ls=":" if k else "--")
        ax.text(mu + k * sigma, pdf.max() * 1.03, f"{k:+d}σ" if k else "μ", ha="center", fontsize=8)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.set_title(f"Normal curve  N({mu:g}, {sigma:g}²)")
    ax.legend(loc="upper right", fontsize=8); ax.set_ylim(0, pdf.max() * 1.12)
    show_dl(fig, "normal_curve")
with r:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(x, stats.norm.cdf(x, mu, sigma), lw=2.2, color=PALETTE[2])
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("x"); ax.set_ylabel("Φ(x)"); ax.set_title("Cumulative distribution function")
    show_dl(fig, "normal_cdf")

st.subheader("Sample vs theoretical curve")
c = st.columns(3)
n = c[0].slider("Sample size", 20, 10000, 1000, step=10)
bins = c[1].slider("Bins", 5, 100, 30)
seed = c[2].number_input("Random seed", 0, 9999, 1)
sample = np.random.default_rng(seed).normal(mu, sigma, n)

l, r = st.columns(2)
with l:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(sample, bins=bins, density=True, alpha=0.55, color=PALETTE[0], edgecolor="white", label="Sample")
    ax.plot(x, pdf, color=PALETTE[1], lw=2.2, label="Theoretical PDF")
    ax.set_title(f"Histogram of {n} samples with normal curve"); ax.legend()
    show_dl(fig, "normal_sample")
with r:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    stats.probplot(sample, dist="norm", plot=ax)
    ax.get_lines()[0].set(markersize=3, color=PALETTE[0]); ax.get_lines()[1].set(color=PALETTE[1])
    ax.set_title("Normal Q–Q plot")
    show_dl(fig, "normal_qq")

st.subheader("Compare several normal curves")
fig, ax = plt.subplots(figsize=(10, 3.4))
for i, (m_, s_) in enumerate([(0, 1), (0, 0.5), (0, 2), (-2, 0.8)]):
    xx = np.linspace(-6, 6, 500)
    ax.plot(xx, stats.norm.pdf(xx, m_, s_), lw=2, label=f"μ={m_}, σ={s_}")
ax.legend(); ax.set_title("Effect of μ and σ on the curve")
show_dl(fig, "normal_compare")

sh, pv = stats.shapiro(sample[:5000])
c = st.columns(5)
c[0].metric("Sample mean", f"{sample.mean():.3f}")
c[1].metric("Sample std", f"{sample.std(ddof=1):.3f}")
c[2].metric("Skewness", f"{stats.skew(sample):.3f}")
c[3].metric("Kurtosis (excess)", f"{stats.kurtosis(sample):.3f}")
c[4].metric("Shapiro–Wilk p", f"{pv:.3f}")
result(f"The normal curve N({mu:g}, {sigma:g}²) was plotted. P({a:.2f} < X < {b:.2f}) = {p:.4f}. "
       f"The sample of {n} values closely follows the theoretical curve (Shapiro–Wilk p = {pv:.3f}).")
