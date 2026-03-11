"""
V-Dem Data Extraction & Discretization
Thailand-Cambodia Border Conflict Model

SETUP:
1. pip install pandas
2. Download V-Dem Country-Year dataset from https://v-dem.net
   - Go to "Data" > "Download" > "Country-Year: V-Dem Full+Others"
   - Save CSV into your data/ folder
3. Run: python3 vdem_extract.py

OUTPUT:
- thailand_values.csv  (raw scores + discretized states)
- cambodia_values.csv  (raw scores + discretized states)
"""

import pandas as pd
import numpy as np


# =============================================================
# LOAD V-DEM
# =============================================================

def load_vdem(filepath="../data/raw/V-Dem-CY-Full+Others-v15.csv"):

    cols = [
        "country_name", "year",

        # Thailand
        "v2x_libdem",      # Liberal democracy index (0-1)
        "v2csreprss",      # CSO repression (ordinal)
        "v2clacjst",       # Compliance with judiciary (ordinal)
        "v2x_cspart",      # Civil society participation (0-1)
        "v2x_polyarchy",   # Electoral democracy index (0-1)
        "v2xlg_legcon",    # Legislative constraints on exec (0-1)
        "v2x_execorr",     # Executive corruption (0-1)
        "v2jupoatck",      # Gov attacks on judiciary (ordinal)
        "v2juhccomp",      # High court independence (ordinal)
        "v2x_corr",        # Political corruption index (0-1)

        # Cambodia
        "v2x_clphy",       # Physical violence index (0-1)
        "v2x_clpol",       # Political civil liberties (0-1)
    ]

    try:
        df = pd.read_csv(filepath, usecols=cols, encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: '{filepath}' not found.")
        print("Download from: https://v-dem.net/data/the-v-dem-dataset/")
        return None, None
    except ValueError:
        df = pd.read_csv(filepath, encoding="utf-8")
        available = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"Warning: missing columns: {missing}")
        df = df[available]

    thailand = df[df["country_name"] == "Thailand"].copy()
    cambodia = df[df["country_name"] == "Cambodia"].copy()

    print(f"Thailand: {len(thailand)} rows ({thailand['year'].min()}-{thailand['year'].max()})")
    print(f"Cambodia: {len(cambodia)} rows ({cambodia['year'].min()}-{cambodia['year'].max()})")

    return thailand, cambodia


# =============================================================
# DISCRETIZE THAILAND
# =============================================================

def discretize_thailand(df, start_year=2008):
    df = df[df["year"] >= start_year].copy()

    # Military Autonomy (from v2x_libdem, inverted)
    # Lower liberal democracy = higher military autonomy
    #   < 0.25  -> High   (post-coup, junta)
    #   0.25-0.40 -> Medium (weak elected gov, military has leverage)
    #   > 0.40  -> Low    (stable civilian control)
    def mil_autonomy(val):
        if pd.isna(val): return np.nan
        if val < 0.25: return "High"
        if val < 0.40: return "Medium"
        return "Low"

    df["Military_Autonomy"] = df["v2x_libdem"].apply(mil_autonomy)

    # Civilian Government Legitimacy (from v2x_polyarchy)
    #   < 0.30  -> Low    (post-coup or caretaker)
    #   0.30-0.50 -> Medium (elected but fragile)
    #   > 0.50  -> High   (stable elected government)
    def gov_legitimacy(val):
        if pd.isna(val): return np.nan
        if val < 0.30: return "Low"
        if val < 0.50: return "Medium"
        return "High"

    df["Gov_Legitimacy"] = df["v2x_polyarchy"].apply(gov_legitimacy)

    # Royalist-Military Bloc Cohesion
    # (from v2juhccomp - v2jupoatck)
    # High court independence + low gov attacks = cohesive establishment bloc
    #   score > 1.5  -> High
    #   score 0-1.5  -> Medium
    #   score < 0    -> Low (gov is fighting the bloc)
    def bloc_cohesion(row):
        court = row.get("v2juhccomp", np.nan)
        attacks = row.get("v2jupoatck", np.nan)
        if pd.isna(court) or pd.isna(attacks): return np.nan
        score = court - attacks
        if score > 1.5: return "High"
        if score > 0: return "Medium"
        return "Low"

    df["Bloc_Cohesion"] = df.apply(bloc_cohesion, axis=1)

    # Economic Stress (V-Dem corruption as supplementary proxy)
    # Primary data comes from Bank of Thailand / FRED
    #   > 0.6  -> High
    #   0.3-0.6 -> Medium
    #   < 0.3  -> Low
    def econ_stress(val):
        if pd.isna(val): return np.nan
        if val > 0.6: return "High"
        if val > 0.3: return "Medium"
        return "Low"

    df["Economic_Stress"] = df["v2x_corr"].apply(econ_stress)

    return df


# =============================================================
# DISCRETIZE CAMBODIA
# =============================================================

def discretize_cambodia(df, start_year=2008):
    df = df[df["year"] >= start_year].copy()

    # Regime Consolidation (from v2x_clpol, inverted)
    # Less political civil liberty = more consolidated regime
    # More consolidated = more willing to provoke externally
    #   < 0.25  -> High   (tightly controlled, one-party state)
    #   0.25-0.45 -> Medium
    #   > 0.45  -> Low    (relatively open)
    def regime_consolidation(val):
        if pd.isna(val): return np.nan
        if val < 0.25: return "High"
        if val < 0.45: return "Medium"
        return "Low"

    df["Regime_Consolidation"] = df["v2x_clpol"].apply(regime_consolidation)

    # State Violence Capacity (from v2x_clphy, inverted)
    # Lower physical integrity = state uses more force
    #   < 0.40  -> High
    #   0.40-0.65 -> Medium
    #   > 0.65  -> Low
    def violence_capacity(val):
        if pd.isna(val): return np.nan
        if val < 0.40: return "High"
        if val < 0.65: return "Medium"
        return "Low"

    df["Violence_Capacity"] = df["v2x_clphy"].apply(violence_capacity)

    # Corruption Environment (from v2x_corr)
    # Higher corruption = scam economy thrives more easily
    #   > 0.7  -> High
    #   0.4-0.7 -> Medium
    #   < 0.4  -> Low
    def corruption_env(val):
        if pd.isna(val): return np.nan
        if val > 0.7: return "High"
        if val > 0.4: return "Medium"
        return "Low"

    df["Corruption_Environment"] = df["v2x_corr"].apply(corruption_env)

    return df


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 60)
    print("V-DEM EXTRACTION")
    print("=" * 60)

    thailand, cambodia = load_vdem()

    if thailand is None:
        return

    # Discretize
    thai = discretize_thailand(thailand)
    camb = discretize_cambodia(cambodia)

    # ----- THAILAND OUTPUT -----
    thai_out = thai[[
        "year",
        "v2x_libdem", "Military_Autonomy",
        "v2x_polyarchy", "Gov_Legitimacy",
        "v2juhccomp", "v2jupoatck", "Bloc_Cohesion",
        "v2x_corr", "Economic_Stress"
    ]].copy()

    thai_out.columns = [
        "Year",
        "Liberal_Democracy_Raw", "Military_Autonomy",
        "Electoral_Democracy_Raw", "Gov_Legitimacy",
        "Court_Independence_Raw", "Gov_Attacks_Judiciary_Raw", "Bloc_Cohesion",
        "Corruption_Raw", "Economic_Stress"
    ]

    # ----- CAMBODIA OUTPUT -----
    camb_out = camb[[
        "year",
        "v2x_clpol", "Regime_Consolidation",
        "v2x_clphy", "Violence_Capacity",
        "v2x_corr", "Corruption_Environment"
    ]].copy()

    camb_out.columns = [
        "Year",
        "Political_Civil_Liberties_Raw", "Regime_Consolidation",
        "Physical_Violence_Index_Raw", "Violence_Capacity",
        "Corruption_Raw", "Corruption_Environment"
    ]

    # Print
    print("\n" + "=" * 60)
    print("THAILAND")
    print("=" * 60)
    print(thai_out.to_string(index=False))

    print("\n" + "=" * 60)
    print("CAMBODIA")
    print("=" * 60)
    print(camb_out.to_string(index=False))

    # Save
    thai_out.to_csv("../data/training/thailand_values.csv", index=False)
    camb_out.to_csv("../data/training/cambodia_values.csv", index=False)

    print(f"\nSaved thailand_values.csv")
    print(f"Saved cambodia_values.csv")


if __name__ == "__main__":
    main()