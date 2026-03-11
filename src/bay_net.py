"""
Thailand-Cambodia Border Conflict Bayesian Network
Zone-Specific Conflict Prediction Model

Merges all training data sources and builds a pgmpy Bayesian Network
that predicts P(Conflict) for each of 4 border zones.

SETUP:
    pip install pgmpy pandas numpy

USAGE:
    python bayesian_network.py

INPUT: All CSV files in ../data/training/ (or update DATA_DIR below)
OUTPUT: Prints model structure, CPTs, and scenario queries per zone
"""

import pandas as pd
import numpy as np
import os
import sys

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.inference import VariableElimination
except ImportError:
    try:
        from pgmpy.models import BayesianNetwork
        from pgmpy.estimators import BayesianEstimator
        from pgmpy.inference import VariableElimination
    except ImportError:
        print("ERROR: pgmpy not installed. Run: pip install pgmpy")
        sys.exit(1)


# =============================================================
# CONFIG
# =============================================================

DATA_DIR = "../data/training"  # Change to your data path

# Known conflict windows (six-month, from model document)
# These are the ground truth labels
CONFLICT_WINDOWS = {
    # (year, half) : [zones that saw conflict]
    (2008, 2): ["Zone_1", "Zone_2"],                          # Oct 2008 - Mar 2009
    (2010, 2): ["Zone_1", "Zone_2"],                          # Oct 2010 - Mar 2011
    (2011, 1): ["Zone_1", "Zone_2"],                          # Apr - Sep 2011
    (2025, 1): ["Zone_1", "Zone_2", "Zone_3", "Zone_4"],     # Feb - Jul 2025
    (2025, 2): ["Zone_1", "Zone_2", "Zone_3", "Zone_4"],     # Oct 2025 - Mar 2026
}

ZONES = ["Zone_1", "Zone_2", "Zone_3", "Zone_4"]

# Dry season months (Nov-Apr) for seasonal multiplier
DRY_MONTHS = [11, 12, 1, 2, 3, 4]


# =============================================================
# LOAD AND MERGE DATA
# =============================================================

def load_all_data():
    """Load all CSV sources and merge into six-month windows per zone."""

    print("Loading data sources...")

    # --- Load each source ---
    s1_thai = pd.read_csv(os.path.join(DATA_DIR, "S1-Vdem-thailand_values.csv"))
    s1_camb = pd.read_csv(os.path.join(DATA_DIR, "S1-Vdem-cambodia_values.csv"))
    s2 = pd.read_csv(os.path.join(DATA_DIR, "S2-nida_poll_pm_approval.csv"))
    s3 = pd.read_csv(os.path.join(DATA_DIR, "S3-bloc_cohesion.csv"))
    s4_thai = pd.read_csv(os.path.join(DATA_DIR, "S4-thailand_economic_stress.csv"))
    s4_camb = pd.read_csv(os.path.join(DATA_DIR, "S4-cambodia_economic_stress.csv"))
    t1 = pd.read_csv(os.path.join(DATA_DIR, "T1-gov_survival.csv"))
    t2 = pd.read_csv(os.path.join(DATA_DIR, "T2-constitutional-court-activity.csv"),
                      comment="#")
    t3 = pd.read_csv(os.path.join(DATA_DIR, "T3-military-civilian-friction-events.csv"))
    t4 = pd.read_csv(os.path.join(DATA_DIR, "T4-nationalist-sentiment-index.csv"),
                      comment="#")
    t5 = pd.read_csv(os.path.join(DATA_DIR, "T5-cambodia-provocation-signal.csv"),
                      comment="#")
    t6 = pd.read_csv(os.path.join(DATA_DIR, "T6-bilateral-channel-health.csv"),
                      comment="#")

    print(f"  S1 Thai: {len(s1_thai)} rows, S1 Camb: {len(s1_camb)} rows")
    print(f"  S2 NIDA: {len(s2)} rows")
    print(f"  S3 Bloc: {len(s3)} rows")
    print(f"  S4 Thai: {len(s4_thai)} rows, S4 Camb: {len(s4_camb)} rows")
    print(f"  T1: {len(t1)} rows, T2: {len(t2)} rows, T3: {len(t3)} rows")
    print(f"  T4: {len(t4)} rows, T5: {len(t5)} rows, T6: {len(t6)} rows")

    return s1_thai, s1_camb, s2, s3, s4_thai, s4_camb, t1, t2, t3, t4, t5, t6


def build_training_windows(s1_thai, s1_camb, s2, s3, s4_thai, s4_camb,
                           t1, t2, t3, t4, t5, t6):
    """
    Build training dataset: one row per (zone, six-month window).
    34 windows (2008-2025) x 4 zones = 136 rows.
    """

    rows = []

    for year in range(2008, 2026):
        for half in [1, 2]:
            # --- Get annual variables for this year ---
            mil_autonomy = _get_val(s1_thai, "Year", year, "Military_Autonomy", "Medium")
            gov_legit_vdem = _get_val(s1_thai, "Year", year, "Gov_Legitimacy", "Medium")
            camb_regime = _get_val(s1_camb, "Year", year, "Regime_Consolidation", "Medium")
            camb_corruption = _get_val(s1_camb, "Year", year, "Corruption_Environment", "Medium")
            bloc_cohesion = _get_val(s3, "Year", year, "Bloc_Cohesion", "Medium")
            thai_econ = _get_val(s4_thai, "Year", year, "Economic_Stress", "Medium")
            camb_econ = _get_val(s4_camb, "Year", year, "Economic_Stress", "Medium")
            court_activity = _get_val(t2, "Year", year, "T2_Activity", "Low")
            mil_friction = _get_val(t3, "Year", year, "T3_Friction", "Low")
            nationalism = _get_val(t4, "Year", year, "T4_Nationalist_Sentiment", "Medium")
            provocation = _get_val(t5, "Year", year, "T5_Provocation_Signal", "Low")
            bilateral = _get_val(t6, "Year", year, "T6_Bilateral_Health", "Medium")

            # --- Get gov survival for this half-year ---
            # Use the worst (lowest) survival in the relevant months
            if half == 1:
                months = [1, 2, 3, 4, 5, 6]
            else:
                months = [7, 8, 9, 10, 11, 12]

            gov_survival = _get_worst_survival(t1, year, months)

            # --- Get NIDA approval (best available for this period) ---
            gov_legit_nida = _get_nida_approval(s2, year, half)

            # Combine V-Dem and NIDA legitimacy: use NIDA when available, else V-Dem
            gov_legitimacy = gov_legit_nida if gov_legit_nida else gov_legit_vdem

            # --- Seasonal multiplier ---
            # H2 (Jul-Dec) includes start of dry season (Nov-Dec)
            # H1 (Jan-Jun) includes bulk of dry season (Jan-Apr)
            season = "Dry" if half == 1 else "Mixed"

            # --- Derive Government Weakness ---
            gov_weakness = _derive_weakness(mil_autonomy, gov_legitimacy,
                                            bloc_cohesion, gov_survival)

            # --- Create row for each zone ---
            for zone in ZONES:
                conflict = _get_conflict_label(year, half, zone)

                # Zone-specific seasonal adjustment for Zone 4
                zone_season = season
                if zone == "Zone_4":
                    # Zone 4 coastal: dry season calmer seas enable naval ops
                    # but ground ops still affected. Mark as "Mixed" always
                    zone_season = "Mixed"

                rows.append({
                    "Year": year,
                    "Half": half,
                    "Zone": zone,
                    "Military_Autonomy": mil_autonomy,
                    "Gov_Legitimacy": gov_legitimacy,
                    "Bloc_Cohesion": bloc_cohesion,
                    "Thai_Econ_Stress": thai_econ,
                    "Gov_Weakness": gov_weakness,
                    "Gov_Survival": gov_survival,
                    "Court_Activity": court_activity,
                    "Mil_Friction": mil_friction,
                    "Nationalism": nationalism,
                    "Cambodia_Provocation": provocation,
                    "Bilateral_Health": bilateral,
                    "Cambodia_Regime": camb_regime,
                    "Cambodia_Corruption": camb_corruption,
                    "Cambodia_Econ_Stress": camb_econ,
                    "Season": zone_season,
                    "Conflict": conflict,
                })

    df = pd.DataFrame(rows)

    # Clean all categorical columns: ensure string type, no NaN
    cat_cols = [
        "Military_Autonomy", "Gov_Legitimacy", "Bloc_Cohesion",
        "Gov_Survival", "Thai_Econ_Stress", "Gov_Weakness",
        "Court_Activity", "Mil_Friction", "Nationalism",
        "Cambodia_Provocation", "Bilateral_Health",
        "Cambodia_Regime", "Cambodia_Corruption", "Cambodia_Econ_Stress",
        "Season", "Conflict"
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Medium").astype(str)
            df[col] = df[col].replace("nan", "Medium")

    print(f"\nTraining windows: {len(df)} rows "
          f"({len(df)//4} windows x {len(ZONES)} zones)")
    print(f"Conflict distribution:")
    print(df.groupby(["Zone", "Conflict"]).size().unstack(fill_value=0))

    return df


# =============================================================
# HELPER FUNCTIONS
# =============================================================

def _get_val(df, key_col, key_val, val_col, default):
    """Get a value from a dataframe by key, with fallback."""
    match = df[df[key_col] == key_val]
    if len(match) > 0 and val_col in match.columns:
        v = match.iloc[0][val_col]
        if pd.notna(v) and v != "N/A":
            return str(v)
    return default


def _get_worst_survival(t1, year, months):
    """Get the worst (lowest) Gov_Survival in a set of months for a year."""
    rank = {"Low": 0, "Medium": 1, "High": 2, "N/A": 2, "nan": 2}
    mask = (t1["Year"] == year) & (t1["Month"].isin(months))
    subset = t1[mask]
    if len(subset) == 0:
        # Check if it's a junta year (2014-2018)
        if 2014 <= year <= 2018:
            return "High"  # Junta = no civilian survival threat
        # Check any data for this year
        any_year = t1[t1["Year"] == year]
        if len(any_year) > 0:
            vals = [str(v) for v in any_year["Gov_Survival"].values]
            if "N/A" in vals or "nan" in vals:
                return "High"  # Junta
        return "Medium"  # Default

    # Filter out NaN/N/A values
    valid = subset[subset["Gov_Survival"].notna()]
    valid = valid[valid["Gov_Survival"] != "N/A"]
    if len(valid) == 0:
        if 2014 <= year <= 2018:
            return "High"  # Junta
        return "Medium"

    worst = min(valid["Gov_Survival"].values, key=lambda x: rank.get(str(x), 1))
    return str(worst)


def _get_nida_approval(s2, year, half):
    """Get NIDA Gov_Legitimacy for a year/half, if available."""
    s2["_year"] = pd.to_datetime(s2["Date"]).dt.year
    s2["_month"] = pd.to_datetime(s2["Date"]).dt.month

    if half == 1:
        mask = (s2["_year"] == year) & (s2["_month"] <= 6)
    else:
        mask = (s2["_year"] == year) & (s2["_month"] > 6)

    subset = s2[mask]
    if len(subset) > 0 and "Gov_Legitimacy" in subset.columns:
        return str(subset.iloc[-1]["Gov_Legitimacy"])
    return None


def _derive_weakness(mil_autonomy, gov_legitimacy, bloc_cohesion, gov_survival):
    """
    Derive the central Government Weakness variable.
    Three states: Stable / Fragile / Collapsed
    """
    score = 0

    # Military autonomy
    if mil_autonomy == "High":
        score += 2
    elif mil_autonomy == "Medium":
        score += 1

    # Government legitimacy (inverted: low legitimacy = more weakness)
    if gov_legitimacy == "Low":
        score += 2
    elif gov_legitimacy == "Medium":
        score += 1

    # Bloc cohesion (high cohesion against gov = more weakness for elected gov)
    if bloc_cohesion == "High":
        score += 2
    elif bloc_cohesion == "Medium":
        score += 1

    # Government survival
    if gov_survival == "Low":
        score += 3  # Heavy weight: immediate survival threat
    elif gov_survival == "Medium":
        score += 1

    if score >= 7:
        return "Collapsed"
    elif score >= 4:
        return "Fragile"
    else:
        return "Stable"


def _get_conflict_label(year, half, zone):
    """Look up whether this zone had conflict in this window."""
    key = (year, half)
    if key in CONFLICT_WINDOWS:
        if zone in CONFLICT_WINDOWS[key]:
            return "Yes"
    return "No"


# =============================================================
# BUILD BAYESIAN NETWORK
# =============================================================

def build_bn(training_data, zone):
    """
    Build and fit a Bayesian Network for a specific zone.

    Simplified structure to avoid sparse CPTs:
        Military_Autonomy -----> Gov_Weakness
        Gov_Legitimacy --------> Gov_Weakness
        Bloc_Cohesion ---------> Gov_Weakness
        Gov_Survival ----------> Gov_Weakness
        Thai_Econ_Stress ------> Gov_Weakness

        Court_Activity --------> Trigger_Level
        Mil_Friction ----------> Trigger_Level
        Nationalism -----------> Trigger_Level
        Cambodia_Provocation --> Trigger_Level

        Gov_Weakness ----------> Conflict
        Trigger_Level ----------> Conflict
        Season ----------------> Conflict

        Cambodia_Regime -------> Cambodia_Provocation

    This gives Conflict only 3 parents (3x3x2 = 18 CPT cells)
    instead of 7 parents (1,458 CPT cells).
    """

    zone_data = training_data[training_data["Zone"] == zone].copy()

    # --- Derive Trigger_Level from sub-triggers ---
    def compute_trigger_level(row):
        score = 0
        for col in ["Court_Activity", "Mil_Friction", "Nationalism",
                     "Cambodia_Provocation"]:
            val = str(row[col])
            if val == "High":
                score += 2
            elif val == "Medium":
                score += 1
        # Also factor in bilateral health (inverted: low health = high trigger)
        bh = str(row["Bilateral_Health"])
        if bh == "Low":
            score += 2
        elif bh == "Medium":
            score += 1

        if score >= 6:
            return "High"
        elif score >= 3:
            return "Medium"
        else:
            return "Low"

    zone_data["Trigger_Level"] = zone_data.apply(compute_trigger_level, axis=1)

    # Define network structure
    # Zone 4 drops Season since it's always "Mixed" (coastal/maritime)
    edges = [
        # Structural -> Government Weakness
        ("Military_Autonomy", "Gov_Weakness"),
        ("Gov_Legitimacy", "Gov_Weakness"),
        ("Bloc_Cohesion", "Gov_Weakness"),
        ("Gov_Survival", "Gov_Weakness"),
        ("Thai_Econ_Stress", "Gov_Weakness"),

        # Gov Weakness + Triggers -> Conflict
        ("Gov_Weakness", "Conflict"),
        ("Trigger_Level", "Conflict"),
    ]

    if zone != "Zone_4":
        edges.append(("Season", "Conflict"))

    model = BayesianNetwork(edges)

    # Select columns for fitting
    fit_cols = [
        "Military_Autonomy", "Gov_Legitimacy", "Bloc_Cohesion",
        "Gov_Survival", "Thai_Econ_Stress", "Gov_Weakness",
        "Trigger_Level", "Conflict"
    ]

    if zone != "Zone_4":
        fit_cols.insert(-1, "Season")  # Add Season before Conflict

    fit_data = zone_data[fit_cols].copy()

    # Define valid states for each node
    state_names = {
        "Military_Autonomy": ["Low", "Medium", "High"],
        "Gov_Legitimacy": ["Low", "Medium", "High"],
        "Bloc_Cohesion": ["Low", "Medium", "High"],
        "Gov_Survival": ["Low", "Medium", "High"],
        "Thai_Econ_Stress": ["Low", "Medium", "High"],
        "Gov_Weakness": ["Stable", "Fragile", "Collapsed"],
        "Trigger_Level": ["Low", "Medium", "High"],
        "Conflict": ["No", "Yes"],
    }

    if zone != "Zone_4":
        state_names["Season"] = ["Dry", "Mixed"]

    # Convert all columns to pandas Categorical dtype
    for col in fit_cols:
        fit_data[col] = fit_data[col].fillna("Medium").astype(str)
        fit_data[col] = fit_data[col].replace("nan", "Medium")
        fit_data[col] = pd.Categorical(fit_data[col], categories=state_names[col])

    # Fit with Bayesian Estimator (Dirichlet prior for small N)
    model.fit(
        fit_data,
        estimator=BayesianEstimator,
        prior_type="dirichlet",
        pseudo_counts=0.15,  # Low prior: trust the data
        state_names=state_names
    )

    return model


# =============================================================
# INFERENCE
# =============================================================

def run_scenarios(model, zone):
    """Run key scenarios for a zone."""

    inference = VariableElimination(model)

    print(f"\n{'='*60}")
    print(f"ZONE: {zone}")
    print(f"{'='*60}")

    # Scenario 1: 2025-like high risk
    print("\n--- Scenario: 2025-like conditions (HIGH RISK) ---")
    try:
        evidence = {
            "Gov_Weakness": "Collapsed",
            "Trigger_Level": "High",
        }
        if zone != "Zone_4":
            evidence["Season"] = "Dry"
        result = inference.query(variables=["Conflict"], evidence=evidence)
        _print_result(result)
    except Exception as e:
        print(f"  Error: {e}")

    # Scenario 2: Stable period (2015-2018 like)
    print("\n--- Scenario: Stable period (LOW RISK) ---")
    try:
        evidence = {
            "Gov_Weakness": "Stable",
            "Trigger_Level": "Low",
        }
        if zone != "Zone_4":
            evidence["Season"] = "Mixed"
        result = inference.query(variables=["Conflict"], evidence=evidence)
        _print_result(result)
    except Exception as e:
        print(f"  Error: {e}")

    # Scenario 3: Fragile gov, moderate triggers
    print("\n--- Scenario: Fragile government, moderate triggers ---")
    try:
        evidence = {
            "Gov_Weakness": "Fragile",
            "Trigger_Level": "Medium",
        }
        if zone != "Zone_4":
            evidence["Season"] = "Dry"
        result = inference.query(variables=["Conflict"], evidence=evidence)
        _print_result(result)
    except Exception as e:
        print(f"  Error: {e}")

    # Scenario 4: Only know Gov_Weakness is Collapsed
    print("\n--- Scenario: Government collapsed (partial evidence) ---")
    try:
        result = inference.query(
            variables=["Conflict"],
            evidence={"Gov_Weakness": "Collapsed"}
        )
        _print_result(result)
    except Exception as e:
        print(f"  Error: {e}")

    return inference


def _print_result(result):
    """Pretty print a pgmpy query result."""
    states = result.state_names["Conflict"]
    probs = result.values
    for state, prob in zip(states, probs):
        marker = " <---" if state == "Yes" else ""
        print(f"  P(Conflict={state}) = {prob:.4f}{marker}")


# =============================================================
# VALIDATION
# =============================================================

def validate(model, training_data, zone):
    """Basic in-sample validation for a zone."""

    zone_data = training_data[training_data["Zone"] == zone].copy()
    inference = VariableElimination(model)

    evidence_cols = [
        "Military_Autonomy", "Gov_Legitimacy", "Bloc_Cohesion",
        "Gov_Survival", "Thai_Econ_Stress",
        "Trigger_Level",
    ]

    if zone != "Zone_4":
        evidence_cols.append("Season")

    # Compute Trigger_Level for validation rows
    def compute_trigger_level(row):
        score = 0
        for col in ["Court_Activity", "Mil_Friction", "Nationalism",
                     "Cambodia_Provocation"]:
            val = str(row[col])
            if val == "High": score += 2
            elif val == "Medium": score += 1
        bh = str(row["Bilateral_Health"])
        if bh == "Low": score += 2
        elif bh == "Medium": score += 1
        if score >= 6: return "High"
        elif score >= 3: return "Medium"
        else: return "Low"

    zone_data["Trigger_Level"] = zone_data.apply(compute_trigger_level, axis=1)

    correct = 0
    total = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    for _, row in zone_data.iterrows():
        evidence = {col: str(row[col]) for col in evidence_cols}
        # Clean any nan strings
        evidence = {k: ("Medium" if v == "nan" else v) for k, v in evidence.items()}
        try:
            result = inference.query(variables=["Conflict"], evidence=evidence)
            states = result.state_names["Conflict"]
            probs = result.values
            predicted = states[np.argmax(probs)]
            actual = row["Conflict"]

            if predicted == actual:
                correct += 1
            if actual == "Yes" and predicted == "Yes":
                tp += 1
            elif actual == "No" and predicted == "Yes":
                fp += 1
            elif actual == "No" and predicted == "No":
                tn += 1
            elif actual == "Yes" and predicted == "No":
                fn += 1
            total += 1
        except:
            pass

    if total > 0:
        accuracy = correct / total * 100
        conflict_count = (zone_data["Conflict"] == "Yes").sum()
        base_accuracy = max(conflict_count, len(zone_data) - conflict_count) / len(zone_data) * 100

        print(f"\n  Validation ({zone}):")
        print(f"    Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"    Naive baseline: {base_accuracy:.1f}%")
        print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
        if tp + fp > 0:
            print(f"    Precision: {tp/(tp+fp):.2f}")
        if tp + fn > 0:
            print(f"    Recall: {tp/(tp+fn):.2f}")


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 60)
    print("THAILAND-CAMBODIA BORDER CONFLICT")
    print("BAYESIAN NETWORK - ZONE PREDICTION MODEL")
    print("=" * 60)

    # Load data
    sources = load_all_data()

    # Build training windows
    training = build_training_windows(*sources)

    # Print training data summary
    print("\nTraining data sample:")
    print(training[["Year", "Half", "Zone", "Gov_Weakness",
                    "Gov_Survival", "Nationalism", "Conflict"]].head(20).to_string(index=False))

    # Build and query per zone
    models = {}
    inferences = {}

    for zone in ZONES:
        print(f"\n{'#'*60}")
        print(f"BUILDING MODEL: {zone}")
        print(f"{'#'*60}")

        zone_data = training[training["Zone"] == zone]
        conflict_yes = (zone_data["Conflict"] == "Yes").sum()
        conflict_no = (zone_data["Conflict"] == "No").sum()
        base_rate = conflict_yes / len(zone_data) * 100
        print(f"  Conflict windows: {conflict_yes} Yes, {conflict_no} No")
        print(f"  Base rate: {base_rate:.1f}%")

        model = build_bn(training, zone)
        models[zone] = model

        inf = run_scenarios(model, zone)
        inferences[zone] = inf

        validate(model, training, zone)

    # Summary comparison across zones
    print(f"\n{'='*60}")
    print("CROSS-ZONE COMPARISON: P(Conflict=Yes | Gov_Weakness=Collapsed)")
    print(f"{'='*60}")

    for zone in ZONES:
        try:
            result = inferences[zone].query(
                variables=["Conflict"],
                evidence={"Gov_Weakness": "Collapsed"}
            )
            states = result.state_names["Conflict"]
            probs = result.values
            yes_idx = states.index("Yes")
            print(f"  {zone}: P(Conflict=Yes) = {probs[yes_idx]:.4f}")
        except Exception as e:
            print(f"  {zone}: Error - {e}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")

    return models, training, inferences


if __name__ == "__main__":
    models, training, inferences = main()