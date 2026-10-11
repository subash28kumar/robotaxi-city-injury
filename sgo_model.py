"""
Paper 3, step 2: are robotaxi crashes in newer cities more likely to injure someone?

Model (logistic regression, robust standard errors clustered by metro area):
    any injury ~ newer market + reporting rules + crash partner + roadway + robotaxi stopped/parked + company
Also asks: does the injury share fall the longer a robotaxi fleet has been in a city ("learning")?

NHTSA changed the reporting rules on June 16, 2025 (some minor property-damage crashes no longer
need a report). Every model includes the reporting-rule period, and results are also shown
separately for the old-rules and new-rules data.

Needs: sgo_ads_clean.csv (from sgo_build.py)
Run:   python sgo_model.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
d = pd.read_csv(BASE / "sgo_ads_clean.csv")
d = d[d["any_injury"].notna() & d["market"].isin(["Early", "Newer"])].copy()
d["newer"] = (d["market"] == "Newer").astype(int)
d["new_rules"] = (d["source_file"] == "current").astype(int)
d["waymo"] = (d["company"] == "Waymo LLC").astype(int)
d["freeway"] = (d["roadway"] == "Highway / Freeway").astype(int)
d["intersection"] = (d["roadway"] == "Intersection").astype(int)
d["parking"] = d["roadway"].isin(["Parking Lot", "Parking Garage"]).astype(int)
d["partner"] = d["partner"].replace({"Pedestrian, cyclist or scooter": "VRU", "Motor vehicle": "Vehicle",
                                     "Object or animal": "Object"})
# Months since the first reported robotaxi crash in that metro area (a simple measure of time in market)
d["t"] = d["year"] * 12 + d["month"]
d["months_in_market"] = d["t"] - d.groupby("metro")["t"].transform("min")
d["years_in_market"] = d["months_in_market"] / 12


def fit(formula, data, label):
    m = smf.logit(formula, data=data).fit(disp=0, cov_type="cluster", cov_kwds={"groups": data["metro"]})
    rows = []
    for v in [x for x in m.params.index if x in ("newer", "years_in_market", "new_rules", "sv_stationary")]:
        b, se = m.params[v], m.bse[v]
        rows.append({"Model": label, "Term": v, "Odds ratio": np.exp(b),
                     "CI low": np.exp(b - 1.96 * se), "CI high": np.exp(b + 1.96 * se),
                     "p": m.pvalues[v], "Crashes": int(m.nobs)})
    return rows


controls = "C(partner, Treatment('Vehicle')) + freeway + intersection + parking + sv_stationary + waymo"
res = []
res += fit(f"any_injury ~ newer + new_rules", d, "1. Unadjusted (rules only)")
res += fit(f"any_injury ~ newer + new_rules + {controls}", d, "2. Adjusted")
res += fit(f"any_injury ~ newer + {controls}", d[d["new_rules"] == 0], "3. Adjusted, old rules only")
res += fit(f"any_injury ~ newer + {controls}", d[d["new_rules"] == 1], "4. Adjusted, new rules only")
res += fit(f"any_injury ~ newer + new_rules + {controls}", d[d["waymo"] == 1], "5. Adjusted, Waymo only")
res += fit(f"any_injury ~ newer + years_in_market + new_rules + {controls}", d, "6. Adjusted + years in market")

# City-by-city adjusted odds of injury compared with San Francisco (robust HC1 standard errors;
# only 9 metro areas, so clustering by metro is not used for this model)
mc = smf.logit(f"any_injury ~ C(metro, Treatment('San Francisco Bay Area')) + new_rules + {controls}",
               data=d).fit(disp=0, cov_type="HC1")
city = []
for v in [x for x in mc.params.index if x.startswith("C(metro")]:
    b, se = mc.params[v], mc.bse[v]
    city.append({"City": v.split("[T.")[1].rstrip("]"), "Odds ratio": np.exp(b),
                 "CI low": np.exp(b - 1.96 * se), "CI high": np.exp(b + 1.96 * se), "p": mc.pvalues[v],
                 "Crashes": int((d["metro"] == v.split("[T.")[1].rstrip("]")).sum())})
city = pd.DataFrame(city).sort_values("Odds ratio")
city.to_csv(BASE / "city_or.csv", index=False)
m0 = smf.logit(f"any_injury ~ new_rules + {controls}", data=d).fit(disp=0)
from scipy.stats import chi2
lr = 2 * (mc.llf - m0.llf)
p_city = chi2.sf(lr, d["metro"].nunique() - 1)

out = pd.DataFrame(res)
out.to_csv(BASE / "model_results.csv", index=False)
show = out.copy()
show["Odds ratio (95% CI)"] = show.apply(lambda r: f"{r['Odds ratio']:.2f} ({r['CI low']:.2f}-{r['CI high']:.2f})", axis=1)
show["p"] = show["p"].round(3)
pd.set_option("display.width", 200)
print("\nInjury share by market and reporting rules")
print(d.groupby(["new_rules", "market"])["any_injury"].agg(["size", "mean"]).round(3).to_string())
print("\nOdds of injury (odds ratio above 1 = more likely)\n")
print(show[["Model", "Term", "Odds ratio (95% CI)", "p", "Crashes"]].to_string(index=False))
print("\nCity vs San Francisco, adjusted odds of injury\n")
print(city.round(3).to_string(index=False))
print(f"\nDo cities differ? Likelihood-ratio test p = {p_city:.2g}")
print("\nSaved model_results.csv, city_or.csv")
