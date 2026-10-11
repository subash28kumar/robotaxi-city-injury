"""
Paper 3, step 3: explainable machine learning (gradient boosting + SHAP).

Question: which crash factors best predict whether a robotaxi crash injured someone,
and how much does the city add once crash type is known?

  1. Compares a gradient boosting model (LightGBM) with logistic regression (5-fold cross-validation, AUC).
  2. Checks how much the city adds: model with city vs model without city.
  3. SHAP values: how much each factor pushes the injury risk up or down.

Needs: sgo_ads_clean.csv (from sgo_build.py)
        pip install lightgbm shap scikit-learn
Run:    python sgo_ml.py
Output: ml_auc.csv, shap_importance.csv, shap_city.csv
"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
SEED = 2026

d = pd.read_csv(BASE / "sgo_ads_clean.csv")
d = d[d["any_injury"].notna() & (d["metro"] != "Other")].reset_index(drop=True)
y = d["any_injury"].astype(int).values

X_base = pd.DataFrame({
    "Pedestrian, cyclist or scooter": (d["partner"] == "Pedestrian, cyclist or scooter").astype(int),
    "Object or animal": (d["partner"] == "Object or animal").astype(int),
    "Other partner": (d["partner"] == "Other").astype(int),
    "Robotaxi stopped or parked": d["sv_stationary"],
    "Robotaxi speed (mph)": d["sv_speed"].fillna(d["sv_speed"].median()),
    "Freeway": (d["roadway"] == "Highway / Freeway").astype(int),
    "Intersection": (d["roadway"] == "Intersection").astype(int),
    "Parking lot or garage": d["roadway"].isin(["Parking Lot", "Parking Garage"]).astype(int),
    "Night (8 pm-6 am)": d["hour"].isin(list(range(20, 24)) + list(range(0, 6))).astype(int),
    "Waymo": (d["company"] == "Waymo LLC").astype(int),
    "New reporting rules": (d["source_file"] == "current").astype(int),
})
cities = pd.get_dummies(d["metro"], prefix="", prefix_sep="").astype(int).drop(columns="San Francisco Bay Area")
X_city = pd.concat([X_base, cities], axis=1)
# LightGBM needs plain feature names; keep readable labels for the output tables
LABELS = {f"f{i}": c for i, c in enumerate(X_city.columns)}
SAFE = {c: k for k, c in LABELS.items()}
X_base = X_base.rename(columns=SAFE)
X_city = X_city.rename(columns=SAFE)
city_cols = [SAFE[c] for c in cities.columns]

GBM = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=30,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=SEED, verbose=-1)


def cv_auc(X, make):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = make().fit(X.iloc[tr], y[tr])
        pred[te] = m.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y, pred)


rows = []
for name, X in [("Without city", X_base), ("With city", X_city)]:
    rows.append({"Model": "Gradient boosting (LightGBM)", "Features": name,
                 "AUC": cv_auc(X, lambda: lgb.LGBMClassifier(**GBM))})
    rows.append({"Model": "Logistic regression", "Features": name,
                 "AUC": cv_auc(X, lambda: LogisticRegression(max_iter=2000, C=1.0))})
auc = pd.DataFrame(rows)
auc.to_csv(BASE / "ml_auc.csv", index=False)

# SHAP on the full model with city
gbm = lgb.LGBMClassifier(**GBM).fit(X_city, y)
sv = shap.TreeExplainer(gbm).shap_values(X_city)
sv = sv[1] if isinstance(sv, list) else sv
imp = pd.DataFrame({"Factor": [LABELS[c] for c in X_city.columns], "Mean |SHAP|": np.abs(sv).mean(axis=0)}) \
    .sort_values("Mean |SHAP|", ascending=False)
imp.to_csv(BASE / "shap_importance.csv", index=False)

# Average SHAP push for crashes in each city (positive = higher injury risk than average)
city_shap = sv[:, [X_city.columns.get_loc(c) for c in city_cols]].sum(axis=1)
by_city = pd.DataFrame({"City": d["metro"], "City SHAP": city_shap, "Injury": y}) \
    .groupby("City").agg(crashes=("Injury", "size"), injury_share=("Injury", "mean"),
                         mean_city_shap=("City SHAP", "mean")).sort_values("mean_city_shap")
by_city.to_csv(BASE / "shap_city.csv")
np.save(BASE / "shap_values.npy", sv)
X_city.rename(columns=LABELS).to_csv(BASE / "shap_X.csv", index=False)

pd.set_option("display.width", 200)
print(f"Crashes: {len(y):,}  Injury crashes: {y.sum():,} ({y.mean():.1%})\n")
print("Cross-validated AUC (0.5 = coin flip, 1.0 = perfect)")
print(auc.round(3).to_string(index=False))
print("\nWhat matters most (mean |SHAP|, log-odds scale)")
print(imp.head(15).round(3).to_string(index=False))
print("\nCity effect after crash type is known (mean SHAP push)")
print(by_city.round(3).to_string())
print("\nSaved ml_auc.csv, shap_importance.csv, shap_city.csv")
