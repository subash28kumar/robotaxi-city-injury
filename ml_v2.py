"""
Robotaxi paper, version 2, step 3: explainable machine learning on the passenger-service sample.
Gradient boosting (LightGBM) vs logistic regression, 5-fold cross-validated AUC, with and without
the metro area; SHAP values for the gradient boosting model. Same settings as version 1.
Run: python ml_v2.py   (after build_v2.py)
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
OUT = Path(__file__).resolve().parent / "outputs"
SEED = 2026
SMALL = ["Dallas", "Houston", "Miami", "Las Vegas"]

ads = pd.read_csv(OUT / "ads_v2.csv", low_memory=False)


def run(d, tag):
    d = d[d["any_injury"].notna() & (d["market"] != "Other")].reset_index(drop=True)
    y = d["any_injury"].astype(int).values
    metro5 = d["metro"].where(~d["metro"].isin(SMALL), "Other newer markets")
    X_base = pd.DataFrame({
        "Pedestrian, cyclist or scooter": (d["partner"] == "Pedestrian, cyclist or scooter").astype(int),
        "Object, animal or other": d["partner"].isin(["Object or animal", "Other"]).astype(int),
        "Robotaxi stopped or parked": d["sv_stationary"],
        "Robotaxi speed (mph)": d["sv_speed"].fillna(d["sv_speed"].median()),
        "Intersection": (d["roadway"] == "Intersection").astype(int),
        "Parking lot or garage": d["roadway"].isin(["Parking Lot", "Parking Garage"]).astype(int),
        "Night (8 pm-6 am)": d["hour"].isin(list(range(20, 24)) + list(range(0, 6))).astype(int),
        "Waymo": (d["company"] == "Waymo LLC").astype(int),
        "New reporting rules": (d["source_file"] == "current").astype(int),
    })
    cities = pd.get_dummies(metro5, prefix="", prefix_sep="").astype(int).drop(columns="San Francisco Bay Area")
    X_city = pd.concat([X_base, cities], axis=1)
    labels = {f"f{i}": c for i, c in enumerate(X_city.columns)}
    safe = {c: k for k, c in labels.items()}
    Xb, Xc = X_base.rename(columns=safe), X_city.rename(columns=safe)
    city_cols = [safe[c] for c in cities.columns]
    gbm_kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=30, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)

    def cv_auc(X, make):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        pred = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            pred[te] = make().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        return roc_auc_score(y, pred)

    rows = []
    for name, X in [("Without city", Xb), ("With city", Xc)]:
        rows.append({"Sample": tag, "Model": "Gradient boosting (LightGBM)", "Features": name,
                     "AUC": cv_auc(X, lambda: lgb.LGBMClassifier(**gbm_kw))})
        rows.append({"Sample": tag, "Model": "Logistic regression", "Features": name,
                     "AUC": cv_auc(X, lambda: LogisticRegression(max_iter=3000))})
    gbm = lgb.LGBMClassifier(**gbm_kw).fit(Xc, y)
    sv = shap.TreeExplainer(gbm).shap_values(Xc)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"Factor": [labels[c] for c in Xc.columns], "Mean |SHAP|": np.abs(sv).mean(axis=0)}) \
        .sort_values("Mean |SHAP|", ascending=False)
    city_shap = sv[:, [Xc.columns.get_loc(c) for c in city_cols]].sum(axis=1)
    by_city = pd.DataFrame({"City": metro5, "City SHAP": city_shap, "Injury": y}).groupby("City") \
        .agg(crashes=("Injury", "size"), injury_share=("Injury", "mean"), mean_city_shap=("City SHAP", "mean")) \
        .sort_values("mean_city_shap")
    return pd.DataFrame(rows), imp, by_city, len(y), int(y.sum())


auc_ps, imp_ps, city_ps, n, k = run(ads[ads["passenger_service"]], "PS")
auc_all, _, _, _, _ = run(ads, "ALL")
auc = pd.concat([auc_ps, auc_all])
auc.to_csv(OUT / "ml_auc_v2.csv", index=False)
imp_ps.to_csv(OUT / "shap_importance_v2.csv", index=False)
city_ps.to_csv(OUT / "shap_city_v2.csv")
pd.set_option("display.width", 200)
print(f"PS crashes {n}, injury {k} ({k/n:.1%})")
print(auc.round(3).to_string(index=False))
print(imp_ps.round(3).to_string(index=False))
print(city_ps.round(3).to_string())
