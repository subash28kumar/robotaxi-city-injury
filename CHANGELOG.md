# Changelog

## 2.0.0 (2026-10-27)

- New passenger-service robotaxi sample: ride-hailing robotaxi fleets (Waymo, Cruise, Zoox, Tesla), no in-vehicle operator, ADS in control. Analysis sample 2,202 crashes in nine metro areas (version 1: all ADS reports).
- New scripts: `build_v2.py`, `model_v2.py`, `ml_v2.py`, `figures_v2.py`. Outputs go to `outputs/`.
- Sensitivity analyses compare three definitions: all ADS reports, all robotaxi-fleet reports, and the passenger-service sample.
- Los Angeles (OR 2.33) and Phoenix (OR 2.13) results are stable across definitions.
- Corrected: version 1 findings for Dallas (lowest injury odds) and Atlanta (higher injury odds) came from mixing in vehicles with in-vehicle safety drivers and do not hold in the passenger-service sample.
- New: robotaxi-fleet crashes with a safety driver had 17.8% injury share, about twice the driverless share.

## 1.0.0 (2026-10-11)

- First release: all ADS crash reports, July 2021 to August 2026 (2,955 crashes after de-duplication).
