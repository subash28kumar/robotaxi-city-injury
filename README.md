# Same robot, different city: robotaxi crash injuries by U.S. city, 2021–2026

Analysis code by Subash Kumar (ORCID [0009-0003-6661-3424](https://orcid.org/0009-0003-6661-3424)), Independent Researcher, Ellicott City, Maryland, USA.

The study uses robotaxi (automated driving system, ADS) crashes reported to NHTSA under its Standing General Order, July 2021 to August 2026, and compares the share of crashes with any injury across nine U.S. metro areas.

## Version 2 (current)

Version 2 analyzes only **passenger-service robotaxis**, defined with three rules applied in order:

1. **Ride-hailing robotaxi fleet:** Waymo, Cruise (including GM reports of Cruise vehicles), Zoox and Tesla. Freight trucks, delivery robots, low-speed shuttles and development or test programs are excluded.
2. **No human operator in the vehicle:** operator type blank or remote only.
3. **The ADS in control:** reports showing driver assistance, "verified not engaged", or manual driving in the narrative are excluded.

| Step | Crashes |
|---|---:|
| All ADS crash reports after de-duplication | 2,955 |
| Ride-hailing robotaxi-fleet reports (rule 1) | 2,649 |
| No in-vehicle operator (rule 2) | 2,258 |
| ADS in control (rule 3) | 2,255 |
| **Analysis sample** (nine metro areas, injury severity known) | **2,202** |

## Main results (version 2)

- 8.9% of passenger-service robotaxi crashes involved an injury.
- Adjusted odds of injury compared with San Francisco: **Los Angeles 2.33 (95% CI 1.57–3.45)** and **Phoenix 2.13 (1.37–3.32)**. Both were stable under every sample definition tested.
- Injury shares did not fall with time in a market.
- Crashes involving pedestrians, cyclists or scooter riders had about 12 times the odds of injury.
- Robotaxi-fleet crashes **with** an in-vehicle safety driver had about twice the injury share (17.8%) of driverless ones, so mixing the two distorts city comparisons.

**Change from version 1:** version 1 used all ADS reports (trucks, shuttles, test programs and vehicles with safety drivers included). Its findings for Dallas (lowest injury odds) and Atlanta (higher injury odds) did not hold in the passenger-service sample: Dallas had only 13 passenger-service crashes, and Atlanta was not statistically significant. See `CHANGELOG.md`.

## Data

Download the two public NHTSA files from https://www.nhtsa.gov/SGOCrashReporting (accessed October 10, 2026) and save them in a folder called `data/` next to the scripts:

| NHTSA file | Save as |
|---|---|
| ADS Incident Report Data, archive (crashes before June 16, 2025) | `data/ADS_archive.csv` |
| ADS Incident Report Data, current (June 16, 2025 onward) | `data/ADS_current.csv` |

A cleaned, de-duplicated copy of the ADS crash file is also archived on Zenodo: https://doi.org/10.5281/zenodo.23461616

## How to run (version 2)

```
pip install -r requirements.txt
python build_v2.py     # de-duplicate, flag the passenger-service sample, sample table
python model_v2.py     # descriptive table, city odds ratios, newer markets, sensitivity analyses
python ml_v2.py        # gradient boosting vs logistic regression, SHAP values
python figures_v2.py   # Figures 1-3 (TIF 600 dpi, PDF, PNG)
```

All results are written to `outputs/` and figures to `outputs/figures/`.

The version 1 scripts (`sgo_build.py`, `sgo_model.py`, `sgo_ml.py`, `sgo_figures.py`) are kept unchanged for the record.

## How to cite

See `CITATION.cff`, or use the "Cite this repository" button on GitHub.

## License

MIT
