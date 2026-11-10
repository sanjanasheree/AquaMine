import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from core.ui import header, theory, result, show_dl, csv_download, metric_row, PALETTE
from core.datasets import house_data, water_quality_data

header("Multiple Linear Regression", "To implement a Multiple Linear Regression model that predicts a continuous "
       "target from several independent variables, interpret its coefficients and validate its assumptions.", "📈")
theory(r"""
**Model:**  y = β₀ + β₁x₁ + β₂x₂ + … + βₚxₚ + ε

**Ordinary Least Squares:** β̂ = (XᵀX)⁻¹Xᵀy minimises Σ(y − ŷ)².

**Evaluation:** R² = 1 − SSE/SST · Adjusted R² = 1 − (1 − R²)(n − 1)/(n − p − 1) · RMSE · MAE.
**Significance:** t = β̂ⱼ / SE(β̂ⱼ), SE from σ̂²(XᵀX)⁻¹; overall F-test.
**Assumptions:** linearity, independent & normally distributed residuals with constant variance
(homoscedasticity), no severe multicollinearity (VIF < 5–10).

**Algorithm:** 1. Load data, choose y and predictors. 2. Split train/test. 3. Fit OLS. 4. Report coefficients,
p-values, R², RMSE. 5. Check residual plots, Q–Q plot and VIF. 6. Predict for new inputs.
""")

wq = water_quality_data()[["pH", "EC", "TDS", "Turbidity", "Temperature", "Total_Hardness", "Well_Depth_m",
                           "Chloride", "Sulphate", "Fluoride"]]
DS = {"House prices (₹ lakh)": (house_data(), "Price_Lakh"), "Groundwater: predict TDS": (wq, "TDS")}
src = st.radio("Data", list(DS) + ["Upload CSV"], horizontal=True)
if src == "Upload CSV":
    f = st.file_uploader("CSV", type="csv")
    if f is None:
        st.stop()
    df = pd.read_csv(f).select_dtypes("number").dropna()
    ycol = st.selectbox("Dependent variable (y)", df.columns, index=len(df.columns) - 1)
else:
    df, ycol = DS[src]
xopts = [c for c in df.columns if c != ycol]
default_x = [c for c in xopts if not (ycol == "TDS" and c == "EC")]
xcols = st.multiselect("Independent variables (X)", xopts, default=default_x)
if not xcols:
    st.warning("Pick at least one predictor."); st.stop()
ts = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

X, y = df[xcols].astype(float), df[ycol].astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=42)
lr = LinearRegression().fit(Xtr, ytr)
pred_tr, pred = lr.predict(Xtr), lr.predict(Xte)

# --- OLS statistics on training data
n, p = Xtr.shape
A = np.column_stack([np.ones(n), Xtr.values])
resid = ytr.values - pred_tr
sse = resid @ resid
sigma2 = sse / (n - p - 1)
cov = sigma2 * np.linalg.pinv(A.T @ A)
se = np.sqrt(np.diag(cov))
beta = np.r_[lr.intercept_, lr.coef_]
tval = beta / se
pval = 2 * stats.t.sf(np.abs(tval), n - p - 1)
r2 = r2_score(ytr, pred_tr)
adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
F = (r2 / p) / ((1 - r2) / (n - p - 1))
Fp = stats.f.sf(F, p, n - p - 1)
ci = stats.t.ppf(0.975, n - p - 1) * se

coef = pd.DataFrame({"Coefficient": beta, "Std. error": se, "t": tval, "p-value": pval,
                     "95% CI low": beta - ci, "95% CI high": beta + ci},
                    index=["Intercept"] + xcols)
coef["Significant (α=0.05)"] = np.where(coef["p-value"] < 0.05, "Yes", "No")

eq = f"{ycol} = {lr.intercept_:.4f} " + " ".join(f"{'+' if b >= 0 else '−'} {abs(b):.4f}·{c}" for b, c in zip(lr.coef_, xcols))
st.markdown("**Fitted equation**")
st.code(eq, language="text")
metric_row({"R² (train)": r2, "Adj. R²": adj, "R² (test)": r2_score(yte, pred),
            "RMSE (test)": float(np.sqrt(mean_squared_error(yte, pred))), "MAE (test)": mean_absolute_error(yte, pred),
            "F-stat p": f"{Fp:.2e}"})
st.dataframe(coef.style.format({c: "{:.4f}" for c in coef.columns[:-1]}).format({"p-value": "{:.2e}"}),
             width="stretch")

# --- VIF
if len(xcols) > 1:
    vif = {}
    for c in xcols:
        others = [o for o in xcols if o != c]
        r2c = LinearRegression().fit(X[others], X[c]).score(X[others], X[c])
        vif[c] = 1 / (1 - r2c) if r2c < 1 else np.inf
    vdf = pd.DataFrame({"VIF": vif}).round(2)
    vdf["Multicollinearity"] = pd.cut(vdf.VIF, [0, 5, 10, np.inf], labels=["Low", "Moderate", "High"])
    with st.expander("Variance Inflation Factors (multicollinearity check)", expanded=False):
        st.dataframe(vdf, width="stretch")

l, r = st.columns(2)
with l:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(yte, pred, s=14, alpha=0.6, color=PALETTE[0])
    lims = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
    ax.plot(lims, lims, "k--"); ax.set_xlabel(f"Actual {ycol}"); ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs predicted (test R² = {r2_score(yte, pred):.3f})")
    show_dl(fig, "mlr_actual_pred")
with r:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(pred_tr, resid, s=12, alpha=0.5, color=PALETTE[1]); ax.axhline(0, color="k")
    ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals"); ax.set_title("Residuals vs fitted (homoscedasticity)")
    show_dl(fig, "mlr_resid")
l, r = st.columns(2)
with l:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    stats.probplot(resid, dist="norm", plot=ax)
    ax.get_lines()[0].set(markersize=3, color=PALETTE[0]); ax.set_title("Normal Q–Q plot of residuals")
    show_dl(fig, "mlr_qq")
with r:
    std_coef = pd.Series(lr.coef_ * Xtr.std().values / ytr.std(), index=xcols).sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.barh(std_coef.index, std_coef.values, color=[PALETTE[1] if v < 0 else PALETTE[0] for v in std_coef])
    ax.set_title("Standardised coefficients (relative effect)")
    show_dl(fig, "mlr_std_coef")

cv = cross_val_score(LinearRegression(), X, y, cv=10, scoring="r2")
dw = np.sum(np.diff(resid) ** 2) / sse
sw_p = stats.shapiro(resid[:5000])[1]
st.markdown(f"**Diagnostics:** 10-fold CV R² = {cv.mean():.4f} ± {cv.std():.4f} · Durbin–Watson = {dw:.3f} "
            f"(≈2 ⇒ independent residuals) · Shapiro–Wilk p (residual normality) = {sw_p:.3f}")

st.subheader("Predict for new values")
with st.form("mlr_form"):
    cols = st.columns(min(4, len(xcols)))
    vals = {c: cols[i % len(cols)].number_input(c, value=float(round(X[c].median(), 3))) for i, c in enumerate(xcols)}
    go = st.form_submit_button("Predict", type="primary")
if go:
    yhat = lr.predict(pd.DataFrame([vals]))[0]
    st.success(f"Predicted **{ycol} = {yhat:.3f}**  (≈95 % prediction interval ± {1.96 * np.sqrt(sigma2):.3f})")

out = Xte.assign(Actual=yte, Predicted=pred.round(3), Residual=(yte - pred).round(3))
csv_download(out, "mlr_test_predictions")
sig = [c for c in xcols if coef.loc[c, "p-value"] < 0.05]
result(f"The MLR model explains {r2:.1%} of the variance in {ycol} on training data (adj. R² = {adj:.3f}) and "
       f"{r2_score(yte, pred):.1%} on test data (RMSE = {np.sqrt(mean_squared_error(yte, pred)):.3f}). "
       f"Significant predictors: {', '.join(sig) if sig else 'none'}.")
