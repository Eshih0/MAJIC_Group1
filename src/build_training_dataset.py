import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ACLED_CSV_PATH = "acled_export.csv"
OUTPUT_CSV_PATH = "pgmpy_training_dataset.csv"

# Choose time frequency:
# "W" = weekly
# "M" = monthly
TIME_FREQ = "M"

# How many future time steps ahead should the target be?
# Example:
#   if TIME_FREQ = "M" and FUTURE_STEPS = 1,
#   then target for Jan is based on Feb events
FUTURE_STEPS = 1

# Simple zone definitions using bounding boxes
# Replace these with your real border zones
ZONE_DEFS = {

    "Zone_1": {   # top-right (north)
        "lat_min": 14.6,
        "lat_max": 15.5,
        "lon_min": 104.0,
        "lon_max": 105.5
    },

    "Zone_2": {
        "lat_min": 14.0,
        "lat_max": 14.6,
        "lon_min": 104.0,
        "lon_max": 105.5
    },

    "Zone_3": {
        "lat_min": 13.3,
        "lat_max": 14.0,
        "lon_min": 104.0,
        "lon_max": 105.5
    },

    "Zone_4": {   # bottom-right (south)
        "lat_min": 12.4,
        "lat_max": 13.3,
        "lon_min": 104.0,
        "lon_max": 105.5
    }

}


# ============================================================
# 1. LOAD ACLED
# ============================================================

def load_acled(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [
        "event_date",
        "latitude",
        "longitude",
        "event_type",
        "fatalities"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required ACLED columns: {missing}")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)

    df = df.dropna(subset=["event_date", "latitude", "longitude"]).copy()
    return df


# ============================================================
# 2. ASSIGN EVENTS TO ZONES
# ============================================================

def assign_zones(df: pd.DataFrame, zone_defs: dict) -> pd.DataFrame:
    out = df.copy()
    out["zone"] = "OutsideZone"

    for zone_name, z in zone_defs.items():
        mask = (
            (out["latitude"] >= z["lat_min"]) &
            (out["latitude"] <  z["lat_max"]) &
            (out["longitude"] >= z["lon_min"]) &
            (out["longitude"] <  z["lon_max"])
        )
        out.loc[mask, "zone"] = zone_name

    return out


# ============================================================
# 3. ADD TIME SLICE
# ============================================================

def add_time_slice(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    out = df.copy()
    out["time_slice"] = out["event_date"].dt.to_period(freq).astype(str)
    return out


# ============================================================
# 4. BUILD NUMERIC FEATURES PER (ZONE, TIME)
# ============================================================

def build_zone_time_features(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    temp["is_battle"] = (temp["event_type"] == "Battles").astype(int)

    temp["is_violent"] = temp["event_type"].isin([
        "Battles",
        "Violence against civilians",
        "Explosions/Remote violence"
    ]).astype(int)

    temp["is_protest_riot"] = temp["event_type"].isin([
        "Protests",
        "Riots"
    ]).astype(int)

    temp["is_remote_violence"] = (temp["event_type"] == "Explosions/Remote violence").astype(int)

    grouped = (
        temp.groupby(["zone", "time_slice"], dropna=False)
        .agg(
            event_count=("event_type", "size"),
            fatalities_sum=("fatalities", "sum"),
            battle_count=("is_battle", "sum"),
            violent_event_count=("is_violent", "sum"),
            protest_riot_count=("is_protest_riot", "sum"),
            remote_violence_count=("is_remote_violence", "sum"),
        )
        .reset_index()
    )

    grouped["fatalities_per_event"] = np.where(
        grouped["event_count"] > 0,
        grouped["fatalities_sum"] / grouped["event_count"],
        0.0
    )

    grouped["violent_event_ratio"] = np.where(
        grouped["event_count"] > 0,
        grouped["violent_event_count"] / grouped["event_count"],
        0.0
    )

    # Composite tension score
    grouped["acled_tension_score"] = (
        1.0 * grouped["event_count"] +
        2.0 * grouped["violent_event_count"] +
        3.0 * grouped["battle_count"] +
        2.0 * grouped["fatalities_sum"] +
        1.5 * grouped["remote_violence_count"]
    )

    return grouped


# ============================================================
# 5. ENSURE ALL ZONE-TIME COMBINATIONS EXIST
#    This is important so missing months become zeros instead of disappearing.
# ============================================================

def complete_zone_time_grid(features_df: pd.DataFrame, zone_names: list) -> pd.DataFrame:
    all_times = sorted(features_df["time_slice"].unique())

    full_index = pd.MultiIndex.from_product(
        [zone_names, all_times],
        names=["zone", "time_slice"]
    )

    out = features_df.set_index(["zone", "time_slice"]).reindex(full_index).reset_index()

    numeric_cols = [
        "event_count",
        "fatalities_sum",
        "battle_count",
        "violent_event_count",
        "protest_riot_count",
        "remote_violence_count",
        "fatalities_per_event",
        "violent_event_ratio",
        "acled_tension_score",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    return out


# ============================================================
# 6. DISCRETIZATION
# ============================================================

def discretize_with_thresholds(series: pd.Series, bins: list, labels=("Low", "Medium", "High")) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(str)


def discretize_with_quantiles(series: pd.Series, labels=("Low", "Medium", "High")) -> pd.Series:
    ranked = series.rank(method="average")
    try:
        return pd.qcut(ranked, q=len(labels), labels=labels, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series([labels[0]] * len(series), index=series.index, dtype="object")


def discretize_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Threshold-based discretization
    out["event_count_disc"] = discretize_with_thresholds(
        out["event_count"],
        bins=[-np.inf, 1, 5, np.inf]
    )

    out["fatalities_sum_disc"] = discretize_with_thresholds(
        out["fatalities_sum"],
        bins=[-np.inf, 0, 10, np.inf]
    )

    out["battle_count_disc"] = discretize_with_thresholds(
        out["battle_count"],
        bins=[-np.inf, 0, 2, np.inf]
    )

    out["violent_event_count_disc"] = discretize_with_thresholds(
        out["violent_event_count"],
        bins=[-np.inf, 0, 3, np.inf]
    )

    out["protest_riot_count_disc"] = discretize_with_thresholds(
        out["protest_riot_count"],
        bins=[-np.inf, 0, 2, np.inf]
    )

    out["remote_violence_count_disc"] = discretize_with_thresholds(
        out["remote_violence_count"],
        bins=[-np.inf, 0, 1, np.inf]
    )

    out["violent_event_ratio_disc"] = discretize_with_thresholds(
        out["violent_event_ratio"],
        bins=[-np.inf, 0.20, 0.60, np.inf]
    )

    out["fatalities_per_event_disc"] = discretize_with_thresholds(
        out["fatalities_per_event"],
        bins=[-np.inf, 0.0, 1.0, np.inf]
    )

    # Composite score can use quantiles if you don't know good domain cutoffs yet
    out["acled_tension_disc"] = discretize_with_quantiles(out["acled_tension_score"])

    return out


# ============================================================
# 7. CREATE FUTURE TARGET LABEL
#    The target is based on future ACLED tension in the same zone.
# ============================================================

def create_future_target(df: pd.DataFrame, future_steps: int = 1) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["zone", "time_slice"]).reset_index(drop=True)

    # Shift future tension label backward so current row predicts future row
    out["border_conflict_risk_disc"] = (
        out.groupby("zone")["acled_tension_disc"].shift(-future_steps)
    )

    return out


# ============================================================
# 8. FINAL TRAINING DATASET
# ============================================================

def build_training_dataset(acled_csv_path: str,
                           output_csv_path: str,
                           zone_defs: dict,
                           time_freq: str = "M",
                           future_steps: int = 1) -> pd.DataFrame:

    # Load + preprocess
    acled = load_acled(acled_csv_path)
    acled = assign_zones(acled, zone_defs)
    acled = add_time_slice(acled, time_freq)

    # Keep only your actual zones if you don't want OutsideZone
    acled = acled[acled["zone"] != "OutsideZone"].copy()

    # Numeric features
    features = build_zone_time_features(acled)

    # Complete missing zone-time combinations
    zone_names = list(zone_defs.keys())
    features = complete_zone_time_grid(features, zone_names)

    # Discretize
    features = discretize_features(features)

    # Build future target
    features = create_future_target(features, future_steps=future_steps)

    # Drop last row of each zone where future target doesn't exist
    features = features.dropna(subset=["border_conflict_risk_disc"]).copy()

    # Final columns for pgmpy training
    final_cols = [
        "zone",
        "time_slice",
        "event_count_disc",
        "fatalities_sum_disc",
        "battle_count_disc",
        "violent_event_count_disc",
        "protest_riot_count_disc",
        "remote_violence_count_disc",
        "violent_event_ratio_disc",
        "fatalities_per_event_disc",
        "acled_tension_disc",
        "border_conflict_risk_disc",
    ]

    training_df = features[final_cols].copy()

    # Convert all BN columns to strings
    for col in final_cols:
        training_df[col] = training_df[col].astype(str)

    # Save
    training_df.to_csv(output_csv_path, index=False)

    return training_df


# ============================================================
# 9. RUN
# ============================================================

if __name__ == "__main__":
    training_df = build_training_dataset(
        acled_csv_path=ACLED_CSV_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        zone_defs=ZONE_DEFS,
        time_freq=TIME_FREQ,
        future_steps=FUTURE_STEPS
    )

    print("\nTraining dataset created successfully.")
    print(f"Saved to: {OUTPUT_CSV_PATH}\n")
    print(training_df.head(10))
    print("\nColumns:")
    print(list(training_df.columns))
