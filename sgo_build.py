"""
Paper 3, step 1: build one clean robotaxi (ADS) crash file from NHTSA's Standing General Order data.

Inputs (same folder as this script, in a subfolder called data/):
    ADS_archive.csv   NHTSA "ADS Incident Report Data", archive (before June 16, 2025)
    ADS_current.csv   NHTSA "ADS Incident Report Data", current (June 16, 2025 onward)

What it does:
    1. Keeps the latest version of each report.
    2. Keeps one report per crash when several companies reported the same crash.
    3. Puts the old and new file formats into the same columns.
    4. Groups cities into metro areas, and marks each metro as an early or newer robotaxi market.

Output: sgo_ads_clean.csv
Run:    python sgo_build.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

# City -> metro area. Anything not listed is "Other".
METRO = {
    "San Francisco": "San Francisco Bay Area", "South San Francisco": "San Francisco Bay Area",
    "Daly City": "San Francisco Bay Area", "San Mateo": "San Francisco Bay Area",
    "Mountain View": "San Francisco Bay Area", "Palo Alto": "San Francisco Bay Area",
    "Sunnyvale": "San Francisco Bay Area", "San Jose": "San Francisco Bay Area",
    "Los Altos": "San Francisco Bay Area", "Foster City": "San Francisco Bay Area",
    "Redwood City": "San Francisco Bay Area", "Menlo Park": "San Francisco Bay Area",
    "Burlingame": "San Francisco Bay Area", "Millbrae": "San Francisco Bay Area",
    "San Bruno": "San Francisco Bay Area", "Brisbane": "San Francisco Bay Area",
    "Oakland": "San Francisco Bay Area", "Berkeley": "San Francisco Bay Area",
    "Emeryville": "San Francisco Bay Area", "Santa Clara": "San Francisco Bay Area",
    "Phoenix": "Phoenix", "Tempe": "Phoenix", "Scottsdale": "Phoenix", "Chandler": "Phoenix",
    "Mesa": "Phoenix", "Gilbert": "Phoenix", "Paradise Valley": "Phoenix", "Glendale": "Phoenix",
    "Los Angeles": "Los Angeles", "Santa Monica": "Los Angeles", "West Hollywood": "Los Angeles",
    "Beverly Hills": "Los Angeles", "Culver City": "Los Angeles", "Inglewood": "Los Angeles",
    "Marina del Rey": "Los Angeles", "Marina Del Rey": "Los Angeles", "Venice": "Los Angeles",
    "Austin": "Austin", "Atlanta": "Atlanta", "Dallas": "Dallas", "Houston": "Houston",
    "Miami": "Miami", "Miami Beach": "Miami", "Las Vegas": "Las Vegas", "Paradise": "Las Vegas",
    "Henderson": "Las Vegas", "Spring Valley": "Las Vegas",
    "Guadalupe": "Phoenix", "Sun City": "Phoenix",
}
# Early markets: robotaxi service at scale before 2023. Newer markets: scaled in 2024 or later.
EARLY = {"San Francisco Bay Area", "Phoenix"}
NEWER = {"Los Angeles", "Austin", "Atlanta", "Dallas", "Houston", "Miami", "Las Vegas"}


def latest_per_crash(df):
    df = df.copy()
    df["Report Version"] = pd.to_numeric(df["Report Version"], errors="coerce")
    df = df.sort_values(["Report ID", "Report Version"]).drop_duplicates("Report ID", keep="last")
    # Several companies can report the same crash (e.g. operator and vehicle maker): keep one.
    has_sid = df["Same Incident ID"].notna() & (df["Same Incident ID"].astype(str).str.strip() != "")
    return pd.concat([df[has_sid].drop_duplicates("Same Incident ID", keep="first"), df[~has_sid]])


def severity(s):
    s = s.astype(str).str.lower()
    out = pd.Series("Unknown", index=s.index)
    out[s.str.contains("no injur")] = "None"
    out[s.str.contains("minor")] = "Minor"
    out[s.str.contains("moderate")] = "Moderate"
    out[s.str.contains("serious")] = "Serious"
    out[s.str.contains("fatal")] = "Fatal"
    return out


def partner(s):
    s = s.astype(str)
    out = pd.Series("Other", index=s.index)
    out[s.isin(["Passenger Car", "SUV", "Pickup Truck", "Van", "Bus", "Heavy Truck",
                "First Responder Vehicle", "Motorcycle"])] = "Motor vehicle"
    out[s.str.contains("Non-Motorist")] = "Pedestrian, cyclist or scooter"
    out[s.isin(["Other Fixed Object", "Pole / Tree", "Animal"])] = "Object or animal"
    return out


def first_present(df, cols):
    out = pd.Series(np.nan, index=df.index, dtype=object)
    for c in cols:
        if c in df:
            out = out.fillna(df[c])
    return out


def load(name, source):
    df = pd.read_csv(DATA / name, low_memory=False, encoding="latin-1")
    df = latest_per_crash(df)
    t = pd.DataFrame(index=df.index)
    t["source_file"] = source
    t["report_id"] = df["Report ID"]
    t["company"] = df["Reporting Entity"].str.strip()
    t["operator"] = df["Operating Entity"].fillna(df["Reporting Entity"]).astype(str).str.strip()
    d = pd.to_datetime(df["Incident Date"], errors="coerce", format="mixed")
    t["year"] = d.dt.year
    t["month"] = d.dt.month
    t["half"] = t["year"].astype("Int64").astype(str) + np.where(t["month"] <= 6, "-H1", "-H2")
    hh = pd.to_numeric(df["Incident Time (24:00)"].astype(str).str.split(":").str[0], errors="coerce")
    t["hour"] = hh.where(hh.between(0, 23))
    t["state"] = df["State"].astype(str).str.strip()
    t["city"] = df["City"].astype(str).str.strip()
    t["metro"] = t["city"].map(METRO).fillna("Other")
    t["market"] = np.select([t["metro"].isin(EARLY), t["metro"].isin(NEWER)], ["Early", "Newer"], "Other")
    t["roadway"] = df["Roadway Type"].fillna("Unknown")
    t["partner_raw"] = df["Crash With"]
    t["partner"] = partner(df["Crash With"])
    t["sv_move"] = df["SV Pre-Crash Movement"].fillna("Unknown")
    t["sv_stationary"] = t["sv_move"].isin(["Stopped", "Parked"]).astype(int)
    sp = pd.to_numeric(df["SV Precrash Speed (MPH)"], errors="coerce")
    t["sv_speed"] = sp.where(sp.between(0, 90))
    t["lighting"] = df["Lighting"] if "Lighting" in df else np.nan
    t["severity"] = severity(df["Highest Injury Severity Alleged"])
    t["hospitalized"] = df["Highest Injury Severity Alleged"].astype(str).str.contains("W/ Hosp").astype(int)
    t["any_injury"] = np.where(t["severity"] == "Unknown", np.nan,
                               t["severity"].isin(["Minor", "Moderate", "Serious", "Fatal"]).astype(float))
    t["moderate_plus"] = np.where(t["severity"] == "Unknown", np.nan,
                                  t["severity"].isin(["Moderate", "Serious", "Fatal"]).astype(float))
    t["airbag"] = first_present(df, ["Any Air Bags Deployed?", "SV Any Air Bags Deployed?"])
    t["engaged"] = first_present(df, ["Engagement Status", "Automation System Engaged?"])
    t["narrative"] = df["Narrative"].astype(str).str.strip()
    return t


ads = pd.concat([load("ADS_archive.csv", "archive"), load("ADS_current.csv", "current")], ignore_index=True)
ads = ads[ads["year"].between(2021, 2026)]
ads.to_csv(BASE / "sgo_ads_clean.csv", index=False)

pd.set_option("display.width", 200)
print(f"Robotaxi (ADS) crashes after removing duplicates: {len(ads):,}")
print("\nBy year:\n", ads["year"].value_counts().sort_index().to_string())
print("\nBy metro:\n", ads["metro"].value_counts().to_string())
print("\nBy company (top 8):\n", ads["company"].value_counts().head(8).to_string())
print("\nInjury severity:\n", ads["severity"].value_counts().to_string())
print("\nCrash partner:\n", ads["partner"].value_counts().to_string())
print(f"\nRobotaxi was stopped or parked: {ads['sv_stationary'].mean():.0%}")
print("\nSaved sgo_ads_clean.csv")
