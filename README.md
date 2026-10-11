# Same robot, different city: robotaxi crash injuries by U.S. city, 2021–2026

Code for the paper *"Same robot, different city: where robotaxi crashes cause injuries in the United States, 2021–2026"* by Subash Kumar (ORCID [0009-0003-6661-3424](https://orcid.org/0009-0003-6661-3424)).

The study uses every robotaxi (automated driving system, ADS) crash reported to NHTSA under its Standing General Order, July 2021 to August 2026, and compares the share of crashes with any injury across nine U.S. metro areas.

## Data

Download the two public NHTSA files from https://www.nhtsa.gov/SGOCrashReporting and save them in a folder called `data/`:

| NHTSA file | Save as |
|---|---|
| ADS Incident Report Data, archive (data before June 16, 2025) | `data/ADS_archive.csv` |
| ADS Incident Report Data, current (June 16, 2025 onward) | `data/ADS_current.csv` |

## How to run

```
pip install -r requirements.txt
python sgo_build.py     # clean, de-duplicated crash file
python sgo_model.py     # logistic regression: city and newer-market odds of injury
python sgo_ml.py        # gradient boosting + SHAP
python sgo_figures.py   # Figures 1-3
```

## Main results

- 2,955 robotaxi crashes after removing duplicate reports; 9.8% involved an injury.
- Adjusted odds of injury vs San Francisco: Los Angeles 2.39, Atlanta 2.14, Phoenix 2.05, Dallas 0.19.
- Newer markets were not clearly worse as a group, and there was no learning effect over time in a city.

## How to cite

See `CITATION.cff`, or use the "Cite this repository" button on GitHub.

## License

MIT
