import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans

from core.datasets import (water_quality_data, label_water, exceedances, wqi_class, BIS_LIMITS,
                           SENSOR_FEATURES, LAB_FEATURES, DISTRICTS, SOURCE_TYPES, LAND_USE, SEASONS)
from core.ui import (header, theory, result, show_dl, csv_download, plot_confusion, plot_roc,
                     metric_row, PALETTE)

header("Groundwater Contamination Prediction",
       "To build a data-mining pipeline that flags unsafe drinking-water sources from field-sensor "
       "readings, so that laboratory testing can be prioritised for the riskiest sources.", "💧")

theory("""
**Pipeline (CRISP-DM)**
1. **Data collection** – field readings (pH, EC, TDS, turbidity, temperature, DO, hardness, well depth,
   distance to sanitation, rainfall, source type, land use, season) + lab results
   (fluoride, arsenic, nitrate, iron, chloride, sulphate, total coliform).
2. **Labelling** – each sample is compared with **BIS IS 10500:2012** permissible limits.
   *Unsafe* if any health-critical parameter (F⁻ > 1.5, As > 0.01, NO₃⁻ > 45 mg/L, coliform > 0) is breached
   or two or more limits are exceeded. Risk level: High (health breach + ≥3 exceedances), Moderate, Low.
3. **Pre-processing** – one-hot encoding of categorical variables, z-score scaling of numerical ones.
4. **Modelling** – several classifiers trained on a stratified train/test split; 5-fold CV for stability.
5. **Evaluation** – accuracy, precision, recall (sensitivity to unsafe sources matters most),
   F1, ROC-AUC, confusion matrix, feature importance.
6. **Deployment** – single-source and batch prediction; contamination-profile clustering.

**Water Quality Index (Weighted Arithmetic):**  WQI = Σ(Wᵢ·Qᵢ)/ΣWᵢ, Qᵢ = 100·(Vᵢ−V_ideal)/(Sᵢ−V_ideal), Wᵢ = K/Sᵢ.
0–25 Excellent · 26–50 Good · 51–75 Poor · 76–100 Very poor · >100 Unfit.
""")

NUM_SENSOR = [f for f in SENSOR_FEATURES if f not in ("Source_Type", "Land_Use", "Season")]
CAT_SENSOR = ["Source_Type", "Land_Use", "Season"]


@st.cache_data
def builtin(n):
    return water_quality_data(n)


# ------------------------------------------------------------------ data source
with st.sidebar:
    st.markdown("#### Project data")
    src = st.radio("Source", ["Built-in synthetic data", "Upload CSV"], key="wq_src")
    n_samples = st.slider("Samples (built-in)", 500, 5000, 2000, 250) if src.startswith("Built") else None

df = builtin(n_samples) if n_samples else None
if src == "Upload CSV":
    up = st.sidebar.file_uploader("Water quality CSV", type="csv")
    st.sidebar.caption("Needs the sensor columns; lab columns are needed to (re)compute labels.")
    if up is None:
        st.info("Upload a CSV with the same column names as the built-in data (download it from the "
                "Data tab as a template). Showing the built-in data for now.")
        df = builtin(2000)
    else:
        df = pd.read_csv(up)
        missing = [c for c in SENSOR_FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()
        if "Potability" not in df.columns:
            if all(c in df.columns for c in LAB_FEATURES):
                df = label_water(df)
            else:
                st.error("No 'Potability' column and lab columns are incomplete – cannot label data.")
                st.stop()
        if "District" not in df.columns:
            df["District"] = "Unknown"

tabs = st.tabs(["📋 Data", "🔍 EDA", "🧪 WQI", "🤖 Train & Evaluate", "🎯 Predict", "🗺️ Hotspot Clusters"])

# ================================================================== DATA
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Samples", len(df))
    c[1].metric("Districts", df["District"].nunique())
    c[2].metric("Unsafe sources", f"{(df['Potability'] == 'Unsafe').mean():.1%}")
    c[3].metric("Mean WQI", f"{df['WQI'].mean():.1f}" if "WQI" in df else "–")
    st.dataframe(df.head(200), width="stretch", height=320)
    csv_download(df, "groundwater_quality_dataset")

    l, r = st.columns(2)
    with l:
        st.markdown("**Descriptive statistics**")
        st.dataframe(df.describe().T.round(3), width="stretch")
    with r:
        st.markdown("**BIS IS 10500:2012 limits used for labelling**")
        lim = pd.DataFrame([{"Parameter": k,
                             "Acceptable": str(v["acceptable"]) if not isinstance(v["acceptable"], tuple) else "6.5–8.5",
                             "Permissible": str(v["permissible"]) if not isinstance(v["permissible"], tuple) else "6.5–8.5",
                             "Unit": v["unit"]} for k, v in BIS_LIMITS.items()])
        st.dataframe(lim, width="stretch", hide_index=True)
        st.markdown("**Missing values**: " + (", ".join(f"{k}={v}" for k, v in df.isna().sum().items() if v)
                                               or "none"))

# ================================================================== EDA
with tabs[1]:
    params = [p for p in ["pH", "TDS", "Turbidity", "Total_Hardness", "Fluoride", "Arsenic", "Nitrate",
                          "Iron", "Chloride", "Sulphate", "Total_Coliform"] if p in df]
    exc = exceedances(df) if all(p in df for p in params) else None

    l, r = st.columns(2)
    with l:
        p = st.selectbox("Parameter distribution", params, index=params.index("Fluoride") if "Fluoride" in params else 0)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for i, lab in enumerate(["Safe", "Unsafe"]):
            ax.hist(df.loc[df.Potability == lab, p], bins=40, alpha=0.6, color=PALETTE[[2, 1][i]], label=lab)
        lim = BIS_LIMITS.get(p, {}).get("permissible")
        if isinstance(lim, tuple):
            for v in lim: ax.axvline(v, color="k", ls="--", lw=1)
        elif lim is not None:
            ax.axvline(lim, color="k", ls="--", lw=1, label=f"BIS limit ({lim})")
        ax.set_title(f"{p} by potability"); ax.set_xlabel(p); ax.legend()
        show_dl(fig, f"wq_hist_{p}")
    with r:
        if exc is not None:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            s = exc.mean().sort_values() * 100
            ax.barh(s.index, s.values, color=[PALETTE[1] if v > 10 else PALETTE[0] for v in s.values])
            for i, v in enumerate(s.values):
                ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
            ax.set_xlabel("% samples exceeding BIS permissible limit"); ax.set_title("Exceedance by parameter")
            show_dl(fig, "wq_exceedance")

    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        d = (df.groupby("District")["Potability"].apply(lambda s: (s == "Unsafe").mean() * 100)
             .sort_values())
        ax.barh(d.index, d.values, color=PALETTE[0])
        ax.set_xlabel("% unsafe samples"); ax.set_title("Unsafe sources by district")
        show_dl(fig, "wq_district")
    with r:
        num = df[[c for c in NUM_SENSOR + LAB_FEATURES if c in df]].corr()
        fig, ax = plt.subplots(figsize=(6.2, 5))
        im = ax.imshow(num, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(num)), num.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(num)), num.columns, fontsize=7)
        ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Correlation matrix")
        show_dl(fig, "wq_corr")

    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ct = pd.crosstab(df["Source_Type"], df["Risk_Level"], normalize="index")[["Low", "Moderate", "High"]] * 100
        ct.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE[2], PALETTE[3], PALETTE[1]], rot=0)
        ax.set_ylabel("%"); ax.set_title("Risk level by source type"); ax.legend(title="Risk")
        show_dl(fig, "wq_source_risk")
    with r:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        data = [df.loc[df.Season == s, "Total_Coliform"].gt(0).mean() * 100 for s in SEASONS]
        ax.bar(SEASONS, data, color=PALETTE[4])
        ax.set_ylabel("% samples with coliform"); ax.set_title("Bacterial contamination by season")
        show_dl(fig, "wq_season")

# ================================================================== WQI
with tabs[2]:
    if "WQI" not in df:
        st.warning("WQI needs lab parameters.")
    else:
        w = df.assign(WQI_Class=df["WQI"].apply(wqi_class))
        order = ["Excellent", "Good", "Poor", "Very poor", "Unfit for drinking"]
        l, r = st.columns(2)
        with l:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.hist(w["WQI"].clip(upper=200), bins=50, color=PALETTE[0])
            for v in (25, 50, 75, 100): ax.axvline(v, color="grey", ls=":", lw=1)
            ax.set_xlabel("WQI (clipped at 200)"); ax.set_title("WQI distribution")
            show_dl(fig, "wqi_hist")
        with r:
            vc = w["WQI_Class"].value_counts().reindex(order).fillna(0)
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.bar(vc.index, vc.values, color=[PALETTE[2], "#8bc34a", PALETTE[3], PALETTE[1], "#8b0000"])
            ax.tick_params(axis="x", rotation=20); ax.set_title("Samples by WQI class")
            show_dl(fig, "wqi_class")
        tbl = (w.groupby("District").agg(Mean_WQI=("WQI", "mean"), Median_WQI=("WQI", "median"),
                                         Pct_Unfit=("WQI", lambda s: (s > 100).mean() * 100),
                                         Samples=("WQI", "size")).round(1).sort_values("Mean_WQI", ascending=False))
        st.markdown("**District-wise WQI summary**")
        st.dataframe(tbl, width="stretch")

# ================================================================== TRAIN
MODELS = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
    "SVM (RBF)": lambda: SVC(probability=True, random_state=42),
    "Naive Bayes": lambda: GaussianNB(),
    "k-NN": lambda: KNeighborsClassifier(n_neighbors=15),
}


def make_pipe(model_name, num, cat):
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    return Pipeline([("pre", pre), ("clf", MODELS[model_name]())])


@st.cache_resource(show_spinner="Training models…")
def train_all(data, target, num, cat, names, test_size, seed):
    X, y = data[num + cat], data[target].astype(str)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    out = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for n in names:
        pipe = make_pipe(n, num, cat).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)
        avg = "binary" if y.nunique() == 2 else "macro"
        pos = "Unsafe" if "Unsafe" in set(y) else None
        kw = {"pos_label": pos} if avg == "binary" else {}
        auc_ = (roc_auc_score(yte == pipe.classes_[1], proba[:, 1]) if y.nunique() == 2
                else roc_auc_score(yte, proba, multi_class="ovr"))
        out[n] = dict(pipe=pipe, pred=pred, proba=proba,
                      Accuracy=accuracy_score(yte, pred),
                      Precision=precision_score(yte, pred, average=avg, zero_division=0, **kw),
                      Recall=recall_score(yte, pred, average=avg, zero_division=0, **kw),
                      F1=f1_score(yte, pred, average=avg, zero_division=0, **kw),
                      ROC_AUC=auc_,
                      CV_Accuracy=cross_val_score(make_pipe(n, num, cat), X, y, cv=cv, n_jobs=-1).mean())
    return out, Xte, yte


with tabs[3]:
    c = st.columns([2, 2, 1.3, 1])
    mode = c[0].radio("Feature set", ["Sensor screening (no lab tests)", "Full lab model (sensor + lab)"],
                      key="wq_mode")
    target = c[1].radio("Target", ["Potability (Safe / Unsafe)", "Risk_Level (Low / Moderate / High)"],
                        key="wq_target").split(" ")[0]
    test_size = c[2].slider("Test size", 0.1, 0.4, 0.25, 0.05)
    seed = c[3].number_input("Seed", 0, 9999, 42)
    chosen = st.multiselect("Models to compare", list(MODELS), default=list(MODELS))
    use_district = st.checkbox("Include District as a feature (location prior)", value=True)

    num = NUM_SENSOR + (LAB_FEATURES if mode.startswith("Full") else [])
    cat = CAT_SENSOR + (["District"] if use_district else [])
    num = [x for x in num if x in df]

    if not chosen:
        st.warning("Select at least one model."); st.stop()
    res, Xte, yte = train_all(df, target, num, cat, tuple(chosen), test_size, seed)
    st.session_state["wq_models"] = dict(res=res, num=num, cat=cat, target=target, mode=mode)

    comp = (pd.DataFrame({k: {m: v[m] for m in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "CV_Accuracy"]}
                          for k, v in res.items()}).T.sort_values("F1", ascending=False))
    best = comp.index[0]
    st.markdown(f"**Model comparison** (test set = {len(yte)} samples; best by F1 → **{best}**)")
    st.dataframe(comp.style.format("{:.3f}").highlight_max(axis=0, color="#cdeccd"), width="stretch")
    csv_download(comp.reset_index().rename(columns={"index": "Model"}), "wq_model_comparison")

    fig, ax = plt.subplots(figsize=(9, 3.4))
    comp[["Accuracy", "Recall", "F1", "ROC_AUC"]].plot(kind="bar", ax=ax, rot=0, width=0.8)
    ax.set_ylim(0, 1.05); ax.set_title("Model performance comparison"); ax.legend(ncol=4, loc="lower right")
    show_dl(fig, "wq_model_bar")

    sel = st.selectbox("Inspect model", list(comp.index), index=0)
    r = res[sel]
    metric_row({k: r[k] for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]})
    l, m = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        plot_confusion(yte, r["pred"], list(r["pipe"].classes_), ax, f"Confusion matrix – {sel}")
        show_dl(fig, "wq_cm")
    with m:
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        plot_roc(yte, r["proba"], list(r["pipe"].classes_), ax)
        show_dl(fig, "wq_roc")

    st.markdown("**Feature importance** (permutation importance on the test set – model-agnostic)")
    pi = permutation_importance(r["pipe"], Xte, yte, n_repeats=5, random_state=0, n_jobs=-1, scoring="f1_macro")
    imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(imp) + 1))
    ax.barh(imp.index, imp.values, xerr=pi.importances_std[np.argsort(pi.importances_mean)], color=PALETTE[0])
    ax.set_xlabel("Drop in macro-F1 when feature is shuffled"); ax.set_title(f"Feature importance – {sel}")
    show_dl(fig, "wq_importance")

    with st.expander("Classification report"):
        st.text(classification_report(yte, r["pred"], zero_division=0))

    result(f"{best} gave the best F1-score of {comp.loc[best, 'F1']:.3f} (accuracy {comp.loc[best, 'Accuracy']:.3f}, "
           f"ROC-AUC {comp.loc[best, 'ROC_AUC']:.3f}) using the "
           f"{'sensor-only' if mode.startswith('Sensor') else 'full sensor + lab'} feature set.")
    if mode.startswith("Sensor"):
        st.caption("Interpretation: sensor-only screening cannot see fluoride/arsenic directly, so accuracy is "
                   "lower than the full lab model – but it ranks sources by risk at near-zero cost, which is what "
                   "is needed to prioritise lab testing. Switch to *Full lab model* to see the upper bound.")

# ================================================================== PREDICT
with tabs[4]:
    store = st.session_state.get("wq_models")
    if not store:
        st.info("Open the **Train & Evaluate** tab first to train the models.")
    else:
        res, num, cat = store["res"], store["num"], store["cat"]
        mname = st.selectbox("Model used for prediction", list(res), key="wq_pred_model")
        pipe = res[mname]["pipe"]
        st.caption(f"Feature set: {store['mode']} · Target: {store['target']}")

        with st.form("wq_form"):
            st.markdown("**Enter readings for one water source**")
            vals = {}
            cols = st.columns(4)
            defaults = df[num].median()
            for i, f in enumerate(num):
                lo, hi = float(df[f].quantile(0.001)), float(df[f].quantile(0.999))
                vals[f] = cols[i % 4].number_input(f, value=float(round(defaults[f], 3)),
                                                   min_value=min(0.0, lo), max_value=hi * 3 + 1,
                                                   format="%.3f" if f == "Arsenic" else "%.2f")
            cc = st.columns(4)
            vals["Source_Type"] = cc[0].selectbox("Source_Type", SOURCE_TYPES)
            vals["Land_Use"] = cc[1].selectbox("Land_Use", LAND_USE)
            vals["Season"] = cc[2].selectbox("Season", SEASONS)
            if "District" in cat:
                vals["District"] = cc[3].selectbox("District", sorted(df["District"].unique()))
            go = st.form_submit_button("Predict", type="primary")
        if go:
            row = pd.DataFrame([vals])[num + cat]
            pred = pipe.predict(row)[0]
            proba = pd.Series(pipe.predict_proba(row)[0], index=pipe.classes_)
            bad = pred in ("Unsafe", "High")
            (st.error if bad else st.warning if pred == "Moderate" else st.success)(
                f"### Prediction: **{pred}**")
            st.bar_chart(proba)
            flags = []
            if not 6.5 <= vals["pH"] <= 8.5: flags.append("pH outside 6.5–8.5")
            if vals["TDS"] > 2000: flags.append("TDS above 2000 mg/L")
            if vals["Turbidity"] > 5: flags.append("Turbidity above 5 NTU")
            if vals["Dist_to_Sanitation_m"] < 15: flags.append("Source within 15 m of sanitation (coliform risk)")
            for f in LAB_FEATURES:
                if f in vals:
                    lim = BIS_LIMITS[f]["permissible"]
                    if vals[f] > lim: flags.append(f"{f} {vals[f]} > BIS limit {lim}")
            if flags:
                st.markdown("**Rule-based flags:** " + " · ".join(flags))
            st.markdown("**Recommended action:** " + (
                "Prioritise for laboratory testing (F⁻, As, NO₃⁻, coliform); advise residents to use an alternate "
                "source or treated water meanwhile." if bad else
                "Include in routine monitoring; re-test after the monsoon."))

        st.divider()
        st.markdown("**Batch prediction** – upload a CSV with the feature columns above")
        bf = st.file_uploader("CSV to score", type="csv", key="wq_batch")
        if bf is not None:
            bd = pd.read_csv(bf)
            miss = [c for c in num + cat if c not in bd]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                bd["Predicted"] = pipe.predict(bd[num + cat])
                pr = pipe.predict_proba(bd[num + cat])
                for i, c in enumerate(pipe.classes_):
                    bd[f"P({c})"] = pr[:, i].round(3)
                st.dataframe(bd, width="stretch")
                csv_download(bd, "wq_batch_predictions", "⬇ Download predictions")
        else:
            st.caption("Tip: download the dataset from the Data tab, delete the label columns, and upload it here.")

# ================================================================== CLUSTERS
with tabs[5]:
    st.markdown("Groups samples by their **contamination profile** (standardised lab parameters) using K-Means, "
                "revealing fluoride-dominant, arsenic-dominant, microbial and low-risk zones.")
    feats = [f for f in ["Fluoride", "Arsenic", "Nitrate", "Iron", "Total_Coliform", "TDS", "Turbidity"] if f in df]
    k = st.slider("Number of clusters (k)", 2, 8, 5, key="wq_k")
    Z = StandardScaler().fit_transform(np.log1p(df[feats] * np.where(np.array(feats) == "Arsenic", 1000, 1)))
    km = KMeans(k, n_init=10, random_state=0).fit(Z)
    lab = km.labels_
    prof = pd.DataFrame(km.cluster_centers_, columns=feats)

    def name(row):
        top = row.idxmax()
        return "Low-risk" if row.max() < 0.3 else {"Total_Coliform": "Microbial", "TDS": "Saline",
                                                   "Turbidity": "Turbid"}.get(top, f"{top}-dominant")
    names = [f"C{i}: {name(prof.iloc[i])}" for i in range(k)]
    dfc = df.assign(Cluster=[names[i] for i in lab])

    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(6, 0.5 * k + 1.5))
        im = ax.imshow(prof.values, cmap="RdYlGn_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(feats)), feats, rotation=35, ha="right"); ax.set_yticks(range(k), names)
        for i in range(k):
            for j in range(len(feats)):
                ax.text(j, i, f"{prof.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)
        ax.grid(False); fig.colorbar(im, ax=ax); ax.set_title("Cluster centroids (z-scores)")
        show_dl(fig, "wq_cluster_profile")
    with r:
        ct = pd.crosstab(dfc["District"], dfc["Cluster"], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ct.plot(kind="barh", stacked=True, ax=ax, color=PALETTE[:k], width=0.8)
        ax.set_xlabel("% of district samples"); ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
        ax.set_title("Contamination profile by district")
        show_dl(fig, "wq_cluster_district")
    summ = dfc.groupby("Cluster").agg(Samples=("Sample_ID", "size"),
                                      Pct_Unsafe=("Potability", lambda s: (s == "Unsafe").mean() * 100),
                                      **{f"Mean_{f}": (f, "mean") for f in feats}).round(3)
    st.dataframe(summ, width="stretch")
    csv_download(dfc, "wq_clustered_samples")
