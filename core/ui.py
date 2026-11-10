"""Shared UI helpers: page header, lab-record blocks, plotting style, dataset upload."""
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc

PALETTE = ["#1f6f8b", "#e07a5f", "#3d9970", "#f2a541", "#7b5ea7", "#c44569", "#2a9d8f", "#8d99ae"]

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "font.size": 9,
})


def header(title, aim, icon=""):
    st.title(f"{icon} {title}".strip())
    st.markdown(f"**Aim:** {aim}")


def theory(md):
    with st.expander("📘 Theory / Algorithm", expanded=False):
        st.markdown(md)


def result(text):
    st.success(f"**Result:** {text}")


def show(fig):
    st.pyplot(fig, clear_figure=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def show_dl(fig, name):
    """Show a matplotlib figure with a PNG download button (handy for lab records)."""
    png = show(fig)
    st.download_button("⬇ Download PNG", png, file_name=f"{name}.png", mime="image/png",
                       key=f"dl_{name}")


def csv_download(df, name, label="⬇ Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode(), file_name=f"{name}.csv",
                       mime="text/csv", key=f"csv_{name}")


def dataset_picker(default_df, default_target, key):
    """Use the built-in dataset or upload a CSV and choose its target column."""
    src = st.radio("Dataset", ["Built-in sample dataset", "Upload my own CSV"],
                   horizontal=True, key=f"{key}_src")
    if src == "Upload my own CSV":
        f = st.file_uploader("CSV file", type="csv", key=f"{key}_up")
        if f is None:
            st.info("Upload a CSV to continue — using the built-in dataset meanwhile.")
            return default_df, default_target
        df = pd.read_csv(f)
        target = st.selectbox("Target (class) column", df.columns, index=len(df.columns) - 1,
                              key=f"{key}_tgt")
        return df, target
    return default_df, default_target


def prepare_xy(df, target, drop=()):
    """One-hot encode categoricals, drop NA rows, return X (DataFrame), y (Series)."""
    d = df.drop(columns=[c for c in drop if c in df.columns]).dropna()
    y = d[target].astype(str)
    X = pd.get_dummies(d.drop(columns=[target]), drop_first=False)
    X = X.loc[:, X.nunique() > 1]
    return X.astype(float), y


def plot_confusion(y_true, y_pred, labels, ax, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title); ax.grid(False)
    thr = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontweight="bold")
    return cm


def plot_roc(y_true, proba, classes, ax):
    """Binary ROC (or one-vs-rest for multi-class)."""
    y_true = np.asarray(y_true)
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true == classes[1], proba[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true == c, proba[:, i])
            ax.plot(fpr, tpr, lw=1.6, label=f"{c} (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")


def metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for c, (k, v) in zip(cols, metrics.items()):
        c.metric(k, f"{v:.3f}" if isinstance(v, float) else v)


def decision_boundary(model, X2, y, ax, title, class_names=None):
    """2-D decision regions for any fitted classifier on two features."""
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    classes = list(model.classes_)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Zi = np.array([classes.index(z) for z in Z]).reshape(xx.shape)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(PALETTE[:len(classes)])
    ax.contourf(xx, yy, Zi, alpha=0.22, cmap=cmap, levels=np.arange(len(classes) + 1) - 0.5)
    for i, c in enumerate(classes):
        m = np.asarray(y) == c
        ax.scatter(X2[m, 0], X2[m, 1], s=18, color=PALETTE[i], edgecolor="white", lw=0.4,
                   label=str(c))
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(False)


def split(X, y, test_size, seed=42):
    """Stratified train/test split, falling back to a plain split for tiny classes."""
    from sklearn.model_selection import train_test_split
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=seed)
