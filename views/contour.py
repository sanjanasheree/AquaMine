import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from core.ui import header, theory, result, show_dl

header("Contour Plot", "To represent a three-dimensional surface z = f(x, y) on a two-dimensional plane using "
       "contour lines (isolines) and filled contours.", "🗺️")
theory("""
A **contour line** joins all points (x, y) where f(x, y) has the same value. Closely spaced lines indicate a
steep surface; widely spaced lines indicate a gentle slope. Used for terrain maps, weather (isobars),
loss-surfaces of ML models and groundwater-level maps.

**Algorithm:** 1. Create 1-D arrays for x and y. 2. Build a grid with `numpy.meshgrid`.
3. Evaluate Z = f(X, Y). 4. `plt.contour` for lines (with `clabel` labels), `plt.contourf` for filled contours.
5. Add a colour bar.
""")

FUNCS = {
    "z = sin(x)·cos(y)": (lambda X, Y: np.sin(X) * np.cos(Y), (-2 * np.pi, 2 * np.pi)),
    "z = x² + y²  (bowl)": (lambda X, Y: X ** 2 + Y ** 2, (-3, 3)),
    "z = x² − y²  (saddle)": (lambda X, Y: X ** 2 - Y ** 2, (-3, 3)),
    "Two Gaussian hills": (lambda X, Y: np.exp(-((X - 1) ** 2 + (Y - 1) ** 2)) +
                           0.8 * np.exp(-((X + 1.2) ** 2 + (Y + 0.8) ** 2) / 0.6), (-3, 3)),
    "Rosenbrock function": (lambda X, Y: np.log1p((1 - X) ** 2 + 100 * (Y - X ** 2) ** 2), (-2, 2)),
    "z = sin(√(x²+y²))  (ripple)": (lambda X, Y: np.sin(np.sqrt(X ** 2 + Y ** 2)), (-10, 10)),
}

c = st.columns(4)
fname = c[0].selectbox("Function", list(FUNCS))
levels = c[1].slider("Number of levels", 4, 40, 15)
cmap = c[2].selectbox("Colour map", ["viridis", "plasma", "coolwarm", "terrain", "RdYlBu_r", "cividis"])
res = c[3].slider("Grid resolution", 30, 400, 200)

f, (lo, hi) = FUNCS[fname]
x = np.linspace(lo, hi, res)
y = np.linspace(lo, hi, res)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

l, r = st.columns(2)
with l:
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    cs = ax.contour(X, Y, Z, levels=levels, cmap=cmap, linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    ax.set_title(f"Contour lines: {fname}"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    show_dl(fig, "contour_lines")
with r:
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.35, alpha=0.6)
    fig.colorbar(cf, ax=ax, label="z")
    ax.set_title(f"Filled contour: {fname}"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    ax.grid(False)
    show_dl(fig, "contour_filled")

if st.checkbox("Show the same function as a 3-D surface with projected contours"):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, linewidth=0, rstride=max(1, res // 60), cstride=max(1, res // 60))
    ax.contour(X, Y, Z, levels=levels, cmap=cmap, offset=Z.min(), zdir="z")
    ax.set_title(fname); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    show_dl(fig, "contour_surface")

result(f"Contour and filled-contour plots of {fname} were drawn with {levels} levels over "
       f"[{lo:.2f}, {hi:.2f}]²; z ranges from {Z.min():.3f} to {Z.max():.3f}.")
