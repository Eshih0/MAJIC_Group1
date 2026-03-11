import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


# ============================================================
# CONFIG
# ============================================================

ACLED_TRAINING_CSV = "pgmpy_training_dataset.csv"

# Optional extra datasets. Leave as None if you don't have them yet.
SOCIAL_MEDIA_CSV = None   # example: "social_media_dataset.csv"
GOV_CSV = None            # example: "gov_dataset.csv"
ECON_CSV = None           # example: "econ_dataset.csv"

TARGET_COL = "border_conflict_risk_disc"
RISK_STATES = ["Low", "Medium", "High"]

# Usually keep this False at first
INCLUDE_ZONE_IN_BN = False


# ============================================================
# 1. LOAD + MERGE CSV FILES
# ============================================================

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def merge_datasets(
    acled_csv: str,
    social_csv: Optional[str] = None,
    gov_csv: Optional[str] = None,
    econ_csv: Optional[str] = None,
) -> pd.DataFrame:
    df = load_csv(acled_csv)

    for extra_path in [social_csv, gov_csv, econ_csv]:
        if extra_path is not None:
            extra_df = load_csv(extra_path)
            df = df.merge(extra_df, on=["zone", "time_slice"], how="left")

    return df


# ============================================================
# 2. PREPARE DATA FOR PGMPY
# ============================================================

def convert_all_bn_columns_to_string(df: pd.DataFrame, bn_columns: list) -> pd.DataFrame:
    out = df.copy()
    for col in bn_columns:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def choose_bn_columns(df: pd.DataFrame) -> list:
    candidate_cols = [
        # ACLED-derived columns
        "event_count_disc",
        "fatalities_sum_disc",
        "battle_count_disc",
        "violent_event_count_disc",
        "protest_riot_count_disc",
        "remote_violence_count_disc",
        "violent_event_ratio_disc",
        "fatalities_per_event_disc",
        "acled_tension_disc",

        # Optional external columns
        "sm_rumor_disc",
        "sm_anger_disc",
        "sm_post_spike_disc",
        "gov_instability_disc",
        "economic_stress_disc",

        # Optional zone node
        "zone" if INCLUDE_ZONE_IN_BN else None,

        # Target
        TARGET_COL,
    ]

    candidate_cols = [c for c in candidate_cols if c is not None]
    existing_cols = [c for c in candidate_cols if c in df.columns]
    return existing_cols


def drop_rows_missing_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    return df.dropna(subset=[target_col]).copy()


# ============================================================
# 3. DEFINE THE BAYESIAN NETWORK STRUCTURE
# ============================================================

def build_model_structure(df: pd.DataFrame) -> DiscreteBayesianNetwork:
    edges = []

    # ACLED tension -> observed conflict indicators
    if "acled_tension_disc" in df.columns and "battle_count_disc" in df.columns:
        edges.append(("acled_tension_disc", "battle_count_disc"))

    if "acled_tension_disc" in df.columns and "fatalities_sum_disc" in df.columns:
        edges.append(("acled_tension_disc", "fatalities_sum_disc"))

    if "acled_tension_disc" in df.columns and "protest_riot_count_disc" in df.columns:
        edges.append(("acled_tension_disc", "protest_riot_count_disc"))

    if "acled_tension_disc" in df.columns and "violent_event_count_disc" in df.columns:
        edges.append(("acled_tension_disc", "violent_event_count_disc"))

    if "acled_tension_disc" in df.columns and "remote_violence_count_disc" in df.columns:
        edges.append(("acled_tension_disc", "remote_violence_count_disc"))

    # External drivers -> intermediate social signals
    if "gov_instability_disc" in df.columns and "sm_rumor_disc" in df.columns:
        edges.append(("gov_instability_disc", "sm_rumor_disc"))

    if "gov_instability_disc" in df.columns and "sm_anger_disc" in df.columns:
        edges.append(("gov_instability_disc", "sm_anger_disc"))

    if "economic_stress_disc" in df.columns and "sm_anger_disc" in df.columns:
        edges.append(("economic_stress_disc", "sm_anger_disc"))

    if "economic_stress_disc" in df.columns and "protest_riot_count_disc" in df.columns:
        edges.append(("economic_stress_disc", "protest_riot_count_disc"))

    # Parent nodes -> target
    target_parents = [
        "acled_tension_disc",
        "gov_instability_disc",
        "economic_stress_disc",
        "sm_rumor_disc",
        "sm_anger_disc",
    ]

    for parent in target_parents:
        if parent in df.columns and TARGET_COL in df.columns:
            edges.append((parent, TARGET_COL))

    # Optional zone-specific priors
    if INCLUDE_ZONE_IN_BN and "zone" in df.columns:
        for child in ["acled_tension_disc", "sm_rumor_disc", "sm_anger_disc", TARGET_COL]:
            if child in df.columns:
                edges.append(("zone", child))

    # Fallback minimal structure if only ACLED columns exist
    if len(edges) == 0 and "acled_tension_disc" in df.columns and TARGET_COL in df.columns:
        edges = [
            ("acled_tension_disc", "battle_count_disc"),
            ("acled_tension_disc", "fatalities_sum_disc"),
            ("acled_tension_disc", "protest_riot_count_disc"),
            ("acled_tension_disc", TARGET_COL),
        ]

    return DiscreteBayesianNetwork(edges)


# ============================================================
# 4. TRAIN MODEL
# ============================================================

def train_model(df: pd.DataFrame) -> DiscreteBayesianNetwork:
    model = build_model_structure(df)

    model.fit(
        df,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=5,
    )

    return model


# ============================================================
# 5. SCORE EACH ROW USING INFERENCE
# ============================================================

def extract_risk_probability(query_result, target_state: str) -> float:
    variable_name = query_result.variables[0]

    if hasattr(query_result, "state_names") and variable_name in query_result.state_names:
        states = query_result.state_names[variable_name]
    else:
        states = RISK_STATES

    probs = query_result.values
    state_to_prob = dict(zip(states, probs))
    return float(state_to_prob.get(target_state, np.nan))


def score_dataset(
    model: DiscreteBayesianNetwork,
    df_for_scoring: pd.DataFrame,
    id_df: pd.DataFrame
) -> pd.DataFrame:
    infer = VariableElimination(model)

    evidence_cols = [c for c in df_for_scoring.columns if c != TARGET_COL]

    scored_rows = []

    for i, row in df_for_scoring.iterrows():
        evidence = {}

        for col in evidence_cols:
            val = row[col]
            if pd.notna(val):
                evidence[col] = val

        result = infer.query(
            variables=[TARGET_COL],
            evidence=evidence,
            show_progress=False
        )

        p_low = extract_risk_probability(result, "Low")
        p_medium = extract_risk_probability(result, "Medium")
        p_high = extract_risk_probability(result, "High")

        risk_score = 0.0 * p_low + 0.5 * p_medium + 1.0 * p_high

        output_row = {
            "zone": id_df.loc[i, "zone"],
            "time_slice": id_df.loc[i, "time_slice"],
            TARGET_COL: row[TARGET_COL] if TARGET_COL in row else np.nan,
            "pred_p_low": p_low,
            "pred_p_medium": p_medium,
            "pred_p_high": p_high,
            "pred_risk_score": risk_score,
        }

        scored_rows.append(output_row)

    return pd.DataFrame(scored_rows)


# ============================================================
# 6. HEATMAP FUNCTIONS
# ============================================================

def sort_time_slices(values) -> list:
    return sorted(values)


def plot_risk_heatmap(scored_df: pd.DataFrame, value_col: str = "pred_p_high") -> None:
    pivot = scored_df.pivot_table(
        index="zone",
        columns="time_slice",
        values=value_col,
        aggfunc="mean"
    )

    pivot = pivot.reindex(columns=sort_time_slices(list(pivot.columns)))

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title(f"Heatmap of {value_col} by Zone and Time Slice")
    ax.set_xlabel("Time Slice")
    ax.set_ylabel("Zone")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    plt.tight_layout()
    plt.show()


def plot_latest_zone_heatmap(scored_df: pd.DataFrame, value_col: str = "pred_p_high") -> None:
    latest_time = sort_time_slices(scored_df["time_slice"].unique())[-1]
    latest_df = scored_df[scored_df["time_slice"] == latest_time].copy()
    latest_df = latest_df.sort_values("zone")

    values = latest_df[value_col].to_numpy().reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(3, 4))
    im = ax.imshow(values, aspect="auto")

    ax.set_xticks([0])
    ax.set_xticklabels([latest_time])
    ax.set_yticks(range(len(latest_df["zone"])))
    ax.set_yticklabels(latest_df["zone"])

    ax.set_title(f"Latest Zone Risk Heatmap\n({value_col})")
    ax.set_ylabel("Zone")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    plt.tight_layout()
    plt.show()


# ============================================================
# 7. SIMPLE EVALUATION
# ============================================================

def evaluate_simple_accuracy(scored_df: pd.DataFrame) -> float:
    def choose_label(row):
        probs = {
            "Low": row["pred_p_low"],
            "Medium": row["pred_p_medium"],
            "High": row["pred_p_high"],
        }
        return max(probs, key=probs.get)

    eval_df = scored_df.dropna(subset=[TARGET_COL]).copy()
    eval_df["pred_label"] = eval_df.apply(choose_label, axis=1)

    accuracy = (eval_df["pred_label"] == eval_df[TARGET_COL]).mean()
    return float(accuracy)


# ============================================================
# 8. MAIN
# ============================================================

def main():
    merged = merge_datasets(
        acled_csv=ACLED_TRAINING_CSV,
        social_csv=SOCIAL_MEDIA_CSV,
        gov_csv=GOV_CSV,
        econ_csv=ECON_CSV,
    )

    print("Merged rows:", len(merged))
    print("Merged columns:", list(merged.columns))

    bn_cols = choose_bn_columns(merged)

    id_df = merged[["zone", "time_slice"]].copy()
    train_df = merged[bn_cols].copy()
    train_df = drop_rows_missing_target(train_df, TARGET_COL)

    # Need matching ids after row drops
    id_df = id_df.loc[train_df.index].copy()

    train_df = convert_all_bn_columns_to_string(train_df, list(train_df.columns))

    print("\nBN columns used:")
    print(list(train_df.columns))

    model = train_model(train_df)

    print("\nModel trained successfully.")
    print("Learned CPDs for:")
    for cpd in model.get_cpds():
        print("-", cpd.variable)

    scored = score_dataset(model, train_df, id_df)

    scored.to_csv("scored_conflict_risk.csv", index=False)
    print("\nSaved scored results to: scored_conflict_risk.csv")

    accuracy = evaluate_simple_accuracy(scored)
    print("Simple label accuracy:", round(accuracy, 3))

    plot_risk_heatmap(scored, value_col="pred_p_high")
    plot_latest_zone_heatmap(scored, value_col="pred_p_high")


if __name__ == "__main__":
    main()