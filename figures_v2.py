"""
Robotaxi paper, version 2, step 4: figures.
  Figure 1  Injury share by metro area, passenger-service sample (95% Wilson CI)
  Figure 2  Adjusted odds of injury vs San Francisco: passenger-service sample (main) and all ADS reports
  Figure 3  Mean |SHAP| by factor, passenger-service sample
Run after build_v2.py, model_v2.py, ml_v2.py.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent / "outputs"
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.8, "legend.frameon": False, "savefig.dpi": 600})
MID = "#777777"


def clean(ax):
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, width=0.8)


def save(fig, name):
    kw = {"bbox_inches": "tight", "pad_inches": 0.05}
    fig.savefig(OUT / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **kw)
    fig.savefig(OUT / f"{name}.pdf", **kw)
    fig.savefig(OUT / f"{name}.png", dpi=220, **kw)
    plt.close(fig)


# Figure 1
d = pd.read_csv(BASE / "table2_descriptive_ps.csv")
allrow = d[d["Metro area"] == "All nine metro areas"].iloc[0]
g = d[d["Metro area"] != "All nine metro areas"].sort_values("Injury %")
fig, ax = plt.subplots(figsize=(3.5, 3.0))
y = np.arange(len(g))
ax.axvline(allrow["Injury %"], color=MID, lw=0.8, ls="--")
small = g["Crashes"] < 40
for yi, (_, r), s in zip(y, g.iterrows(), small):
    ax.errorbar(r["Injury %"], yi, xerr=[[r["Injury %"] - r["CI low"]], [r["CI high"] - r["Injury %"]]],
                fmt="o", color="black", mfc="white" if s else "black", ms=3.5, elinewidth=0.8, capsize=2)
ax.set_yticks(y)
ax.set_yticklabels([f"{m} (n={n:,})" for m, n in zip(g["Metro area"], g["Crashes"])])
ax.set_xlabel("Passenger-service robotaxi crashes with any injury (%, 95% CI)")
ax.text(allrow["Injury %"], len(g) - 0.4, f" All nine areas {allrow['Injury %']:.1f}%", color=MID, fontsize=7)
clean(ax)
fig.tight_layout()
save(fig, "Figure1")

# Figure 2
c = pd.read_csv(BASE / "table3_city_or.csv")
order = c[c["Sample"] == "PS"].sort_values("OR")["City"].tolist()
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.axvline(1, color=MID, lw=0.8, ls="--")
for off, lab, face, txt in [(0.15, "PS", "black", "Passenger-service sample (main)"),
                            (-0.15, "ALL", "white", "All ADS reports (sensitivity)")]:
    s = c[c["Sample"] == lab].set_index("City").loc[order]
    yy = np.arange(len(order)) + off
    ax.errorbar(s["OR"], yy, xerr=[s["OR"] - s["CI low"], s["CI high"] - s["OR"]], fmt="none",
                ecolor="black", elinewidth=0.8, capsize=1.5)
    ax.scatter(s["OR"], yy, s=16, zorder=3, edgecolor="black", lw=0.8, c=face, label=txt)
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlim(0.2, 6)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels(order)
ax.set_xlabel("Adjusted odds of injury vs San Francisco Bay Area\n(odds ratio, 95% CI, log scale)")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=1, fontsize=6.5, handletextpad=0.3)
clean(ax)
fig.tight_layout()
save(fig, "Figure2")

# Figure 3
s = pd.read_csv(BASE / "shap_importance_v2.csv").head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(3.5, 3.0))
yy = np.arange(len(s))
ax.barh(yy, s["Mean |SHAP|"], color="#555555", height=0.65)
ax.set_yticks(yy)
ax.set_yticklabels(s["Factor"])
ax.set_xlabel("Mean absolute SHAP value (log-odds of injury)")
clean(ax)
fig.tight_layout()
save(fig, "Figure3")
print("figures saved")
