import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from core.ui import header, theory, result, show_dl, PALETTE
from core.datasets import customer_data, water_quality_data

header("3D Plot", "To create three-dimensional surface, wireframe, scatter and line plots using Matplotlib "
       "(static) and Plotly (interactive, rotatable).", "🧊")
theory("""
Matplotlib's `mpl_toolkits.mplot3d` adds a third axis: `fig.add_subplot(projection='3d')`.
- **Surface** `plot_surface(X, Y, Z)` – coloured continuous surface over a meshgrid.
- **Wireframe** `plot_wireframe` – grid lines only.
- **3D scatter** `scatter(x, y, z)` – points with three coordinates (e.g. customer Age/Income/Spending).
- **3D line** `plot(x, y, z)` – parametric curves such as a helix x = cos t, y = sin t, z = t.

**Algorithm:** 1. Generate data / meshgrid. 2. Create 3D axes. 3. Call the plotting function.
4. Label axes, add colour bar, set viewing angle (`view_init(elev, azim)`).
""")

kind = st.radio("Plot type", ["Surface", "Wireframe", "3D Scatter", "3D Line (helix)", "Bar3D"], horizontal=True)
c = st.columns(4)
elev = c[0].slider("Elevation", 0, 90, 28)
azim = c[1].slider("Azimuth", -180, 180, -60)
cmap = c[2].selectbox("Colour map", ["viridis", "plasma", "coolwarm", "terrain", "magma"])
interactive = c[3].toggle("Interactive (Plotly)", value=False)

FUN = {"z = sin(√(x²+y²))": lambda X, Y: np.sin(np.sqrt(X ** 2 + Y ** 2)),
       "z = x·e^(−x²−y²)": lambda X, Y: X * np.exp(-X ** 2 - Y ** 2) * 6,
       "z = x² − y²": lambda X, Y: (X ** 2 - Y ** 2) / 10,
       "z = cos(x)·sin(y)": lambda X, Y: np.cos(X) * np.sin(Y)}

summary = ""
if kind in ("Surface", "Wireframe"):
    fn = st.selectbox("Function", list(FUN))
    rng_ = 6 if "√" in fn or "cos" in fn else 2.5
    X, Y = np.meshgrid(np.linspace(-rng_, rng_, 80), np.linspace(-rng_, rng_, 80))
    Z = FUN[fn](X, Y)
    summary = f"{kind.lower()} of {fn}"
    if interactive:
        fig = go.Figure(go.Surface(x=X, y=Y, z=Z, colorscale={"viridis": "Viridis", "plasma": "Plasma", "coolwarm": "RdBu_r", "terrain": "Earth", "magma": "Magma"}[cmap],
                                   showscale=True, hidesurface=kind == "Wireframe",
                                   contours={"x": {"show": kind == "Wireframe"}, "y": {"show": kind == "Wireframe"}}))
        fig.update_layout(height=560, title=fn, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
    else:
        fig = plt.figure(figsize=(8, 5.6)); ax = fig.add_subplot(111, projection="3d")
        if kind == "Surface":
            s = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True)
            fig.colorbar(s, ax=ax, shrink=0.55, pad=0.08)
        else:
            ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, color=PALETTE[0], linewidth=0.6)
        ax.set_title(fn); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.view_init(elev, azim)
        show_dl(fig, "plot3d_surface")

elif kind == "3D Scatter":
    ds = st.selectbox("Data", ["Mall customers (Age, Income, Spending)", "Groundwater (pH, TDS, Fluoride)"])
    if ds.startswith("Mall"):
        d = customer_data(); cols = ["Age", "Annual_Income_kINR", "Spending_Score"]; colour = d["Gender"]
    else:
        d = water_quality_data().sample(800, random_state=1); cols = ["pH", "TDS", "Fluoride"]; colour = d["Potability"]
    summary = f"3D scatter of {', '.join(cols)}"
    if interactive:
        fig = go.Figure()
        for i, g in enumerate(sorted(colour.unique())):
            m = colour == g
            fig.add_trace(go.Scatter3d(x=d.loc[m, cols[0]], y=d.loc[m, cols[1]], z=d.loc[m, cols[2]], mode="markers",
                                       marker=dict(size=3.5, color=PALETTE[i]), name=str(g)))
        fig.update_layout(height=560, scene=dict(xaxis_title=cols[0], yaxis_title=cols[1], zaxis_title=cols[2]),
                          margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, width="stretch")
    else:
        fig = plt.figure(figsize=(8, 5.6)); ax = fig.add_subplot(111, projection="3d")
        for i, g in enumerate(sorted(colour.unique())):
            m = colour == g
            ax.scatter(d.loc[m, cols[0]], d.loc[m, cols[1]], d.loc[m, cols[2]], s=10, color=PALETTE[i], label=g,
                       alpha=0.7)
        ax.set_xlabel(cols[0]); ax.set_ylabel(cols[1]); ax.set_zlabel(cols[2]); ax.legend(); ax.view_init(elev, azim)
        ax.set_title(summary)
        show_dl(fig, "plot3d_scatter")

elif kind == "3D Line (helix)":
    turns = st.slider("Turns", 1, 12, 5)
    t = np.linspace(0, turns * 2 * np.pi, 1500)
    xx, yy, zz = np.cos(t), np.sin(t), t / (2 * np.pi)
    summary = f"3D helix with {turns} turns"
    if interactive:
        fig = go.Figure(go.Scatter3d(x=xx, y=yy, z=zz, mode="lines", line=dict(width=6, color=zz, colorscale="Viridis")))
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, width="stretch")
    else:
        fig = plt.figure(figsize=(7, 5.6)); ax = fig.add_subplot(111, projection="3d")
        ax.plot(xx, yy, zz, lw=2, color=PALETTE[0])
        ax.plot(xx, yy, np.zeros_like(zz), lw=0.6, color="grey", alpha=0.5)
        ax.set_title("Helix: x = cos t, y = sin t, z = t/2π"); ax.view_init(elev, azim)
        show_dl(fig, "plot3d_helix")

else:
    d = water_quality_data()
    piv = (d.assign(U=(d.Potability == "Unsafe") * 100)
           .pivot_table(index="Source_Type", columns="Season", values="U", aggfunc="mean"))
    summary = "3D bar chart of % unsafe sources by source type and season"
    fig = plt.figure(figsize=(8, 5.6)); ax = fig.add_subplot(111, projection="3d")
    xs, ys = np.meshgrid(range(piv.shape[1]), range(piv.shape[0]))
    xs, ys, zs = xs.ravel(), ys.ravel(), piv.values.ravel()
    colors = plt.get_cmap(cmap)(zs / zs.max())
    ax.bar3d(xs, ys, np.zeros_like(zs), 0.6, 0.6, zs, color=colors, shade=True)
    ax.set_xticks(np.arange(piv.shape[1]) + 0.3, piv.columns, fontsize=7)
    ax.set_yticks(np.arange(piv.shape[0]) + 0.3, piv.index, fontsize=7)
    ax.set_zlabel("% unsafe"); ax.set_title("% unsafe water sources"); ax.view_init(elev, azim)
    show_dl(fig, "plot3d_bar")

result(f"A {summary} was plotted successfully" + (" (interactive – drag to rotate)." if interactive else "."))
