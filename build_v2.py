"""
Robotaxi paper, version 2, step 1: build the crash file and define the passenger-service robotaxi sample.

Inputs (NHTSA Standing General Order, public files, accessed October 10, 2026), in data/ (see README):
    ADS_archive.csv   ADS Incident Report Data, archive (crashes before June 16, 2025)
    ADS_current.csv   ADS Incident Report Data, current (June 16, 2025 onward)

Steps
    1. Keep the latest version of each report (Report ID, highest Report Version).
    2. Keep one report per crash when several entities reported the same crash (Same Incident ID).
    3. Check that no crash appears in both files (none do: no shared Report ID or Same Incident ID,
       and no match on entity + month + time + city in the overlap months).
    4. Harmonize the old and new file formats.
    5. Flag the PASSENGER-SERVICE ROBOTAXI sample with three rules, applied in order:
         R1 Operator runs a driverless ride-hailing (robotaxi) fleet: Waymo, Cruise (and GM reports
            of Cruise vehicles), Zoox, Tesla. Excluded: freight trucks, delivery robots, low-speed
            shuttles, and development or test programs.
         R2 No human operator inside the vehicle: "Driver / Operator Type" is blank (no operator
            reported) or "Remote (Commercial / Test)" only. Excluded: In-Vehicle, In-Vehicle and
            Remote, Consumer, Other or Unknown.
         R3 The automated driving system was in control: excluded if "Automation System Engaged?"
            is ADAS or unknown, if "Engagement Status" is "Verified Not Engaged", or if the
            narrative says the vehicle was in manual mode or being driven manually.
Output: ads_v2.csv (all deduplicated ADS crashes with flags), sample_waterfall.csv
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

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
    "Fremont": "San Francisco Bay Area", "Portola Valley": "San Francisco Bay Area",
    "San Frabcisco": "San Francisco Bay Area",  # spelling error in one report
    "Phoenix": "Phoenix", "Tempe": "Phoenix", "Scottsdale": "Phoenix", "Chandler": "Phoenix",
    "Mesa": "Phoenix", "Gilbert": "Phoenix", "Paradise Valley": "Phoenix", "Glendale": "Phoenix",
    "Guadalupe": "Phoenix", "Sun City": "Phoenix",
    "Los Angeles": "Los Angeles", "Santa Monica": "Los Angeles", "West Hollywood": "Los Angeles",
    "Beverly Hills": "Los Angeles", "Culver City": "Los Angeles", "Inglewood": "Los Angeles",
    "Marina Del Rey": "Los Angeles", "Venice": "Los Angeles", "Lennox": "Los Angeles",
    "Austin": "Austin", "Atlanta": "Atlanta", "Peachtree Corners": "Atlanta",
    "Dallas": "Dallas", "Houston": "Houston",
    "Miami": "Miami", "Miami Beach": "Miami", "Coral Gables": "Miami",
    "Las Vegas": "Las Vegas", "Paradise": "Las Vegas", "Henderson": "Las Vegas", "Spring Valley": "Las Vegas",
}
TX_DFW = {"Arlington", "Fort Worth"}           # Dallas-Fort Worth cities (Texas only)
EARLY = {"San Francisco Bay Area", "Phoenix"}
NEWER = {"Los Angeles", "Austin", "Atlanta", "Dallas", "Houston", "Miami", "Las Vegas"}

ROBOTAXI_OPERATORS = {"Waymo LLC": "Waymo", "Cruise LLC": "Cruise", "General Motors, LLC": "Cruise",
                      "Zoox, Inc.": "Zoox", "Tesla, Inc.": "Tesla"}
MANUAL = re.compile(r"manual mode|manually (operated|driven|driving)|being driven manually|"
                    r"in manual|assumed manual|took (over )?manual control|disengaged autonomy|"
                    r"conventional mode", re.I)


def latest_per_crash(df):
    df = df.copy()
    df["Report Version"] = pd.to_numeric(df["Report Version"], errors="coerce")
    n0 = len(df)
    df = df.sort_values(["Report ID", "Report Version"]).drop_duplicates("Report ID", keep="last")
    n1 = len(df)
    sid = df["Same Incident ID"].astype(str).str.strip()
    has_sid = df["Same Incident ID"].notna() & (sid != "") & (sid.str.lower() != "nan")
    out = pd.concat([df[has_sid].drop_duplicates("Same Incident ID", keep="first"), df[~has_sid]])
    return out, n0, n1, len(out)


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


def col(df, name):
    return df[name] if name in df else pd.Series(np.nan, index=df.index)


def load(name, source):
    raw = pd.read_csv(DATA / name, low_memory=False, encoding="latin-1")
    df, n0, n1, n2 = latest_per_crash(raw)
    t = pd.DataFrame(index=df.index)
    t["source_file"] = source
    t["report_id"] = df["Report ID"]
    t["company"] = df["Reporting Entity"].astype(str).str.strip()
    t["operator"] = df["Operating Entity"].fillna(df["Reporting Entity"]).astype(str).str.strip()
    t["make"] = df["Make"].astype(str).str.strip().str.title()
    d = pd.to_datetime(df["Incident Date"], errors="coerce", format="mixed")
    t["year"], t["month"] = d.dt.year, d.dt.month
    hh = pd.to_numeric(df["Incident Time (24:00)"].astype(str).str.split(":").str[0], errors="coerce")
    t["hour"] = hh.where(hh.between(0, 23))
    t["state"] = df["State"].astype(str).str.strip().str.upper()
    t["city"] = df["City"].astype(str).str.strip().str.title()
    metro = t["city"].map(METRO)
    metro = metro.where(~(t["city"].isin(TX_DFW) & (t["state"] == "TX")), "Dallas")
    t["metro"] = metro.fillna("Other")
    t["market"] = np.select([t["metro"].isin(EARLY), t["metro"].isin(NEWER)], ["Early", "Newer"], "Other")
    t["roadway"] = df["Roadway Type"].fillna("Unknown")
    t["partner_raw"] = df["Crash With"]
    t["partner"] = partner(df["Crash With"])
    t["sv_move"] = df["SV Pre-Crash Movement"].fillna("Unknown")
    t["sv_stationary"] = t["sv_move"].isin(["Stopped", "Parked"]).astype(int)
    sp = pd.to_numeric(df["SV Precrash Speed (MPH)"], errors="coerce")
    t["sv_speed"] = sp.where(sp.between(0, 90))
    t["lighting"] = col(df, "Lighting")
    t["severity"] = severity(df["Highest Injury Severity Alleged"])
    t["hospitalized"] = df["Highest Injury Severity Alleged"].astype(str).str.contains("W/ Hosp").astype(int)
    t["any_injury"] = np.where(t["severity"] == "Unknown", np.nan,
                               t["severity"].isin(["Minor", "Moderate", "Serious", "Fatal"]).astype(float))
    t["moderate_plus"] = np.where(t["severity"] == "Unknown", np.nan,
                                  t["severity"].isin(["Moderate", "Serious", "Fatal"]).astype(float))
    t["operator_type"] = df["Driver / Operator Type"].fillna("(blank: no operator reported)")
    t["ads_field"] = col(df, "Automation System Engaged?").fillna("(blank)")
    t["engagement_status"] = col(df, "Engagement Status").fillna("(not in file)")
    t["narrative"] = df["Narrative"].astype(str).str.strip()
    return t, (source, n0, n1, n2)


parts = [load("ADS_archive.csv", "archive"), load("ADS_current.csv", "current")]
ads = pd.concat([p[0] for p in parts], ignore_index=True)
dedup = pd.DataFrame([p[1] for p in parts], columns=["file", "rows_all_versions", "latest_version", "one_per_crash"])
n_before_years = len(ads)
ads = ads[ads["year"].between(2021, 2026)].reset_index(drop=True)

# ---- Passenger-service robotaxi rules ----
ads["fleet"] = ads["company"].map(ROBOTAXI_OPERATORS).fillna("Not a robotaxi fleet")
ads["r1_robotaxi_fleet"] = ads["company"].isin(ROBOTAXI_OPERATORS)
ads["r2_no_inside_operator"] = ads["operator_type"].isin(["(blank: no operator reported)", "Remote (Commercial / Test)"])
manual_text = ads["narrative"].str.contains(MANUAL)
ads["r3_ads_in_control"] = ~(ads["ads_field"].isin(["ADAS", "Unknown, see Narrative"])
                             | (ads["engagement_status"] == "Verified Not Engaged") | manual_text)
ads["passenger_service"] = ads["r1_robotaxi_fleet"] & ads["r2_no_inside_operator"] & ads["r3_ads_in_control"]


def group(c):
    if c in ROBOTAXI_OPERATORS:
        return "Robotaxi fleet"
    lc = c.lower()
    if any(k in lc for k in ["aurora", "kodiak", "paccar", "daimler", "navistar", "stack av", "waabi",
                              "plus", "embark", "tusimple", "gatik", "torc", "volvo", "international"]):
        return "Freight truck"
    if "nuro" in lc:
        return "Delivery robot"
    if any(k in lc for k in ["beep", "transdev", "easymile", "navya", "local motors", "ohmio", "karsan"]):
        return "Low-speed shuttle"
    return "Development / test or other program"


ads["program_type"] = ads["company"].map(group)

# Waterfall
w = []
total = len(ads)
w.append(("All ADS crash reports after de-duplication (July 2021 - August 2026)", total, ""))
for g in ["Freight truck", "Delivery robot", "Low-speed shuttle", "Development / test or other program"]:
    w.append((f"  Excluded R1: {g}", -int((ads["program_type"] == g).sum()), ", ".join(
        ads.loc[ads["program_type"] == g, "company"].value_counts().index[:6])))
s1 = ads[ads["r1_robotaxi_fleet"]]
w.append(("Robotaxi-fleet reports (Waymo, Cruise/GM, Zoox, Tesla)", len(s1), ""))
for v, k in s1.loc[~s1["r2_no_inside_operator"], "operator_type"].value_counts().items():
    w.append((f"  Excluded R2: operator type = {v}", -int(k), ""))
s2 = s1[s1["r2_no_inside_operator"]]
w.append(("Driverless robotaxi-fleet reports", len(s2), ""))
r3a = s2["ads_field"].isin(["ADAS", "Unknown, see Narrative"])
r3b = (s2["engagement_status"] == "Verified Not Engaged") & ~r3a
r3c = s2["narrative"].str.contains(MANUAL) & ~r3a & ~r3b
w.append(("  Excluded R3: automation field ADAS or unknown", -int(r3a.sum()), ""))
w.append(("  Excluded R3: engagement verified not engaged", -int(r3b.sum()), ""))
w.append(("  Excluded R3: narrative says manual mode", -int(r3c.sum()), ""))
s3 = s2[s2["r3_ads_in_control"]]
w.append(("PASSENGER-SERVICE ROBOTAXI SAMPLE (all locations)", len(s3), ""))
s4 = s3[s3["market"] != "Other"]
w.append(("  Excluded: outside the nine study metro areas", -(len(s3) - len(s4)), ""))
w.append(("  Excluded: injury severity unknown", -int(s4["any_injury"].isna().sum()), ""))
s5 = s4[s4["any_injury"].notna()]
w.append(("ANALYSIS SAMPLE (nine metro areas, known injury)", len(s5), ""))
wf = pd.DataFrame(w, columns=["Step", "Crashes", "Examples"])
wf.to_csv(OUT / "sample_waterfall.csv", index=False)
dedup.to_csv(OUT / "dedup_counts.csv", index=False)
ads.to_csv(OUT / "ads_v2.csv", index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)
print(dedup.to_string(index=False))
print(f"Before year filter {n_before_years}; after 2021-2026: {len(ads)}")
print(wf.to_string(index=False))
print("\nAnalysis sample by metro and fleet:")
print(pd.crosstab(s5["metro"], s5["fleet"], margins=True).to_string())
print("\nInjury share passenger-service:", round(s5["any_injury"].mean(), 4), int(s5["any_injury"].sum()))
