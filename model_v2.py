"""
Robotaxi paper, version 2, step 2: models on the passenger-service sample, with sensitivity analyses.

Outcome: any alleged injury (minor or worse) in a crash. Logistic regression with robust (HC1)
standard errors. Cities with fewer than 40 passenger-service crashes (Dallas, Houston, Miami,
Las Vegas) are pooled as "Other newer markets" for the city model.

Samples
    PS   passenger-service robotaxi sample (main)        -> rules R1-R3 in build_v2.py
    ALL  all ADS crash reports (the earlier definition)   -> sensitivity
    RF   robotaxi fleets incl. crashes with a test driver -> sensitivity
Run: python model_v2.py   (after build_v2.py)
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "outputs"
ads = pd.read_csv(OUT / "ads_v2.csv", low_memory=False)
SMALL = ["Dallas", "Houston", "Miami", "Las Vegas"]


def prep(d):
    d = d[d["any_injury"].notna() & d["market"].isin(["Early", "Newer"])].copy()
    d["newer"] = (d["market"] == "Newer").astype(int)
    d["new_rules"] = (d["source_file"] == "current").astype(int)
    d["waymo"] = (d["company"] == "Waymo LLC").astype(int)
    d["freeway"] = (d["roadway"] == "Highway / Freeway").astype(int)
    d["intersection"] = (d["roadway"] == "Intersection").astype(int)
    d["parking"] = d["roadway"].isin(["Parking Lot", "Parking Garage"]).astype(int)
    d["partner"] = d["partner"].replace({"Pedestrian, cyclist or scooter": "VRU", "Motor vehicle": "Vehicle",
                                         "Object or animal": "Object or other", "Other": "Object or other"})
    d["metro5"] = d["metro"].where(~d["metro"].isin(SMALL), "Other newer markets")
    d["t"] = d["year"] * 12 + d["month"]
    d["years_in_market"] = (d["t"] - d.groupby("metro")["t"].transform("min")) / 12
    return d


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


# Freeway is not modelled: driverless passenger service almost never used freeways (0.3% of crashes, no injuries).
CONTROLS = "C(partner, Treatment('Vehicle')) + intersection + parking + sv_stationary + waymo"
samples = {
    "PS": prep(ads[ads["passenger_service"]]),
    "ALL": prep(ads),
    "RF": prep(ads[ads["r1_robotaxi_fleet"]]),
}
ps = samples["PS"]

# ---------- Table 2: descriptive by metro (PS) ----------
rows = []
for m, g in ps.groupby("metro"):
    k, n = int(g["any_injury"].sum()), len(g)
    lo, hi = wilson(k, n)
    rows.append({"Metro area": m, "Market": g["market"].iloc[0], "Crashes": n, "Injury crashes": k,
                 "Injury %": 100 * k / n, "CI low": 100 * lo, "CI high": 100 * hi,
                 "VRU crash %": 100 * (g["partner"] == "VRU").mean(),
                 "Robotaxi stopped %": 100 * g["sv_stationary"].mean(),
                 "Waymo %": 100 * g["waymo"].mean(),
                 "First / last crash": f"{g['year'].min()}-{g['year'].max()}"})
k, n = int(ps["any_injury"].sum()), len(ps)
lo, hi = wilson(k, n)
rows.append({"Metro area": "All nine metro areas", "Market": "", "Crashes": n, "Injury crashes": k,
             "Injury %": 100 * k / n, "CI low": 100 * lo, "CI high": 100 * hi,
             "VRU crash %": 100 * (ps["partner"] == "VRU").mean(),
             "Robotaxi stopped %": 100 * ps["sv_stationary"].mean(), "Waymo %": 100 * ps["waymo"].mean(),
             "First / last crash": f"{ps['year'].min()}-{ps['year'].max()}"})
desc = pd.DataFrame(rows).sort_values("Crashes", ascending=False)
desc.to_csv(OUT / "table2_descriptive_ps.csv", index=False)


# ---------- City model ----------
def city_model(d, label):
    m = smf.logit(f"any_injury ~ C(metro5, Treatment('San Francisco Bay Area')) + new_rules + {CONTROLS}",
                  data=d).fit(disp=0, cov_type="HC1")
    m0 = smf.logit(f"any_injury ~ new_rules + {CONTROLS}", data=d).fit(disp=0)
    lr = 2 * (m.llf - m0.llf)
    df_ = d["metro5"].nunique() - 1
    p_lr = chi2.sf(lr, df_)
    out = []
    for v in [x for x in m.params.index if x.startswith("C(metro5")]:
        name = v.split("[T.")[1].rstrip("]")
        b, se = m.params[v], m.bse[v]
        out.append({"Sample": label, "City": name, "OR": np.exp(b), "CI low": np.exp(b - 1.96 * se),
                    "CI high": np.exp(b + 1.96 * se), "p": m.pvalues[v], "Crashes": int((d["metro5"] == name).sum())})
    out = pd.DataFrame(out)
    out["p Holm"] = multipletests(out["p"], method="holm")[1]
    other = {}
    for v in m.params.index:
        if "VRU" in v or "Object" in v or v in ("sv_stationary", "parking", "intersection", "new_rules", "waymo"):
            b, se = m.params[v], m.bse[v]
            other[v] = (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[v])
    return out, p_lr, lr, df_, int(m.nobs), other


city_tabs, lr_rows, covars = [], [], {}
for lab, d in samples.items():
    t, p, lr, df_, nobs, other = city_model(d, lab)
    city_tabs.append(t)
    lr_rows.append({"Sample": lab, "LR chi2": lr, "df": df_, "p": p, "Crashes": nobs,
                    "Injury %": 100 * d["any_injury"].mean()})
    covars[lab] = other
city = pd.concat(city_tabs)
city.to_csv(OUT / "table3_city_or.csv", index=False)
lrt = pd.DataFrame(lr_rows)
lrt.to_csv(OUT / "city_lr_tests.csv", index=False)
cov = pd.DataFrame([{"Sample": s, "Term": k, "OR": v[0], "CI low": v[1], "CI high": v[2], "p": v[3]}
                    for s, dd in covars.items() for k, v in dd.items()])
cov.to_csv(OUT / "covariate_or.csv", index=False)


# ---------- Newer vs early markets, learning ----------
def fit_terms(formula, d, label, sample):
    m = smf.logit(formula, data=d).fit(disp=0, cov_type="HC1")
    r = []
    for v in ("newer", "years_in_market"):
        if v in m.params:
            b, se = m.params[v], m.bse[v]
            r.append({"Sample": sample, "Model": label, "Term": v, "OR": np.exp(b), "CI low": np.exp(b - 1.96 * se),
                      "CI high": np.exp(b + 1.96 * se), "p": m.pvalues[v], "Crashes": int(m.nobs)})
    return r


res = []
for lab, d in samples.items():
    res += fit_terms("any_injury ~ newer + new_rules", d, "Unadjusted (reporting period only)", lab)
    res += fit_terms(f"any_injury ~ newer + new_rules + {CONTROLS}", d, "Adjusted", lab)
    res += fit_terms(f"any_injury ~ newer + {CONTROLS}", d[d["new_rules"] == 0], "Adjusted, old reporting rules", lab)
    res += fit_terms(f"any_injury ~ newer + {CONTROLS}", d[d["new_rules"] == 1], "Adjusted, new reporting rules", lab)
    res += fit_terms(f"any_injury ~ newer + new_rules + {CONTROLS.replace(' + waymo', '')}", d[d["waymo"] == 1],
                     "Adjusted, Waymo only", lab)
    res += fit_terms(f"any_injury ~ newer + years_in_market + new_rules + {CONTROLS}", d,
                     "Adjusted + years in market", lab)
nt = pd.DataFrame(res)
nt.to_csv(OUT / "table4_newer_learning.csv", index=False)

# ---------- Extra sensitivity on PS: outcome moderate+ and excluding object/animal crashes ----------
extra = []
# Moderate or worse injuries are too rare (about 1.5%) for a stable city model; report counts only.
mod = ps[ps["moderate_plus"].notna()].groupby("metro5")["moderate_plus"].agg(["size", "sum", "mean"])
mod.to_csv(OUT / "moderate_plus_by_metro.csv")
d3 = ps[ps["partner"] != "Object or other"]
t, p, lr, df_, nobs, _ = city_model(d3, "PS, vehicle and VRU crashes only")
extra.append(t)
lr_rows.append({"Sample": "PS, vehicle and VRU only", "LR chi2": lr, "df": df_, "p": p, "Crashes": nobs,
                "Injury %": 100 * d3["any_injury"].mean()})
pd.concat(extra).to_csv(OUT / "sensitivity_city_or_extra.csv", index=False)
# Newer markets without Los Angeles (is the 'newer' gap only Los Angeles?)
noLA = []
for lab in ("PS", "ALL"):
    d = samples[lab]
    noLA += fit_terms(f"any_injury ~ newer + new_rules + {CONTROLS}", d[d["metro"] != "Los Angeles"],
                      "Adjusted, excluding Los Angeles", lab)
    noLA += fit_terms(f"any_injury ~ newer + {CONTROLS}", d[(d["metro"] != "Los Angeles") & (d["new_rules"] == 1)],
                      "Adjusted, excluding Los Angeles, new rules", lab)
nt = pd.concat([nt, pd.DataFrame(noLA)])
nt.to_csv(OUT / "table4_newer_learning.csv", index=False)
pd.DataFrame(lr_rows).to_csv(OUT / "city_lr_tests.csv", index=False)

pd.set_option("display.width", 220)
fmt = lambda r: f"{r['OR']:.2f} ({r['CI low']:.2f}-{r['CI high']:.2f})"
print(desc.round(1).to_string(index=False))
print("\nCity ORs vs San Francisco (HC1), by sample")
c2 = city.copy(); c2["OR (95% CI)"] = c2.apply(fmt, axis=1)
print(c2[["Sample", "City", "OR (95% CI)", "p", "p Holm", "Crashes"]].round(4).to_string(index=False))
print("\nLR tests"); print(pd.DataFrame(lr_rows).round(4).to_string(index=False))
print("\nCovariates"); cv = cov.copy(); cv["OR (95% CI)"] = cv.apply(fmt, axis=1)
print(cv[["Sample", "Term", "OR (95% CI)", "p"]].round(4).to_string(index=False))
print("\nNewer vs early / learning"); n2 = nt.copy(); n2["OR (95% CI)"] = n2.apply(fmt, axis=1)
print(n2[["Sample", "Model", "Term", "OR (95% CI)", "p", "Crashes"]].round(3).to_string(index=False))
e2 = pd.concat(extra); e2["OR (95% CI)"] = e2.apply(fmt, axis=1)
print("\nExtra sensitivity"); print(e2[["Sample", "City", "OR (95% CI)", "p", "Crashes"]].round(3).to_string(index=False))
print("\nModerate+ by metro (PS)"); print(mod.round(3).to_string())
print("\nShares by market and rules (PS)")
print(ps.groupby(["new_rules", "market"])["any_injury"].agg(["size", "mean"]).round(3).to_string())
print("\nVRU injury share PS:", round(ps.loc[ps.partner == 'VRU', 'any_injury'].mean(), 3),
      int((ps.partner == 'VRU').sum()))
