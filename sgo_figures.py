"""
Paper 3, step 4: journal-style figures (same style as Papers 1 and 2).

  Figure 1  Share of robotaxi crashes with an injury, by city (95% CI)     (sgo_ads_clean.csv)
  Figure 2  Adjusted odds of injury vs San Francisco, by city              (city_or.csv)
  Figure 3  What drives injury risk: mean |SHAP| by factor                 (shap_importance.csv)

Run after sgo_build.py, sgo_model.py and sgo_ml.py:
    python sgo_figures.py
Output: figures/FigureN.tif, .pdf, .png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.linewidth": 0.8, "legend.frameon": False, "savefig.dpi": 600,
})
MID = "#777777"
SHORT = {"San Francisco Bay Area": "San Francisco Bay Area"}


def clean(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=3, width=0.8)


def save(fig, name):
    kw = {"bbox_inches": "tight", "pad_inches": 0.05}
    fig.savefig(OUT / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **kw)
    fig.savefig(OUT / f"{name}.pdf", **kw)
    fig.savefig(OUT / f"{name}.png", dpi=200, **kw)
    plt.close(fig)
    print(f"Saved {name}")


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


# ---------- Figure 1: injury share by city ----------
d = pd.read_csv(BASE / "sgo_ads_clean.csv")
d = d[d["any_injury"].notna() & (d["metro"] != "Other")]
g = d.groupby("metro")["any_injury"].agg(k="sum", n="size").reset_index()
g["share"] = g["k"] / g["n"]
g["lo"], g["hi"] = wilson(g["k"], g["n"])
g = g.sort_values("share")
overall = d["any_injury"].mean()
fig, ax = plt.subplots(figsize=(3.5, 3.0))
y = np.arange(len(g))
ax.axvline(overall * 100, color=MID, linewidth=0.8, linestyle="--")
ax.errorbar(g["share"] * 100, y, xerr=[(g["share"] - g["lo"]) * 100, (g["hi"] - g["share"]) * 100],
            fmt="o", color="black", markersize=3.5, elinewidth=0.8, capsize=2)
ax.set_yticks(y)
ax.set_yticklabels([f"{m} (n={n:,})" for m, n in zip(g["metro"], g["n"])])
ax.set_xlabel("Robotaxi crashes with any injury (%, 95% CI)")
ax.text(overall * 100, len(g) - 0.4, f" All cities {overall:.1%}", color=MID, fontsize=7, va="bottom")
clean(ax)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
fig.tight_layout()
save(fig, "Figure1")

# ---------- Figure 2: adjusted odds ratios by city ----------
c = pd.read_csv(BASE / "city_or.csv").sort_values("Odds ratio")
fig, ax = plt.subplots(figsize=(3.5, 2.8))
y = np.arange(len(c))
ax.axvline(1, color=MID, linewidth=0.8, linestyle="--")
sig = (c["CI low"] > 1) | (c["CI high"] < 1)
ax.errorbar(c["Odds ratio"], y, xerr=[c["Odds ratio"] - c["CI low"], c["CI high"] - c["Odds ratio"]],
            fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
ax.scatter(c["Odds ratio"], y, s=18, zorder=3, edgecolor="black", linewidth=0.8,
           c=np.where(sig, "black", "white"))
ax.set_xscale("log")
ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4"])
ax.set_yticks(y)
ax.set_yticklabels(c["City"])
ax.set_xlabel("Adjusted odds of injury vs San Francisco Bay Area\n(odds ratio, 95% CI, log scale)")
clean(ax)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
fig.tight_layout()
save(fig, "Figure2")

# ---------- Figure 3: SHAP importance ----------
s = pd.read_csv(BASE / "shap_importance.csv").head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(3.5, 3.0))
y = np.arange(len(s))
ax.barh(y, s["Mean |SHAP|"], color="#555555", height=0.65)
ax.set_yticks(y)
ax.set_yticklabels(s["Factor"])
ax.set_xlabel("Mean absolute SHAP value (log-odds of injury)")
clean(ax)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
fig.tight_layout()
save(fig, "Figure3")
print(f"\nAll figures saved in: {OUT}")
