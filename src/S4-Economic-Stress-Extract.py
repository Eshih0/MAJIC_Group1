"""
Economic Stress Data Extraction
Thailand + Cambodia

Sources:
- Thailand GDP growth: World Bank, Macrotrends, Trading Economics
- Thailand household debt/GDP: Bank of Thailand, CEIC, Bangkok Post
- Cambodia GDP growth: World Bank
- Cambodia inflation: World Bank

Saves to: ../data/training/thailand_economic_stress.csv
         ../data/training/cambodia_economic_stress.csv

Discretization:
- High stress: negative GDP growth OR household debt > 90% OR GDP growth < 1%
- Medium stress: GDP growth 1-3% OR household debt 80-90%
- Low stress: GDP growth > 3% AND household debt < 80%
"""

import pandas as pd
import os


# =============================================================
# THAILAND ECONOMIC DATA
# =============================================================

thailand_data = [
    # Sources: World Bank, Macrotrends, CEIC, Bank of Thailand, Bangkok Post
    # GDP growth: annual % (World Bank / Macrotrends)
    # Household debt/GDP: annual average % (Bank of Thailand via CEIC / Bangkok Post)
    #   Pre-2012 figures exclude some smaller lenders per BOT methodology change

    {"Year": 2008, "GDP_Growth_Pct": 1.73, "Household_Debt_GDP_Pct": 55.0,
     "Notes": "Global financial crisis. Samak/Somchai govts dissolved by court. Abhisit became PM Dec."},

    {"Year": 2009, "GDP_Growth_Pct": -0.69, "Household_Debt_GDP_Pct": 58.0,
     "Notes": "Recession. First negative growth since 1997 crisis. Red Shirt protests."},

    {"Year": 2010, "GDP_Growth_Pct": 7.51, "Household_Debt_GDP_Pct": 61.0,
     "Notes": "Strong rebound. But Red Shirt crackdown killed 90+. Political instability."},

    {"Year": 2011, "GDP_Growth_Pct": 0.84, "Household_Debt_GDP_Pct": 64.0,
     "Notes": "Massive floods. GDP nearly flat. Yingluck became PM Aug."},

    {"Year": 2012, "GDP_Growth_Pct": 7.24, "Household_Debt_GDP_Pct": 72.0,
     "Notes": "Post-flood recovery. BOT methodology change added more lenders to debt figure."},

    {"Year": 2013, "GDP_Growth_Pct": 2.69, "Household_Debt_GDP_Pct": 77.0,
     "Notes": "Growth slowed. Political crisis began Nov. Household debt rising rapidly."},

    {"Year": 2014, "GDP_Growth_Pct": 0.98, "Household_Debt_GDP_Pct": 79.5,
     "Notes": "May coup. Near-zero growth. Political vacuum. Household debt approaching 80% threshold."},

    {"Year": 2015, "GDP_Growth_Pct": 3.13, "Household_Debt_GDP_Pct": 81.2,
     "Notes": "Junta stabilization. Household debt crossed 80% 'watchful level' (BOT)."},

    {"Year": 2016, "GDP_Growth_Pct": 3.28, "Household_Debt_GDP_Pct": 79.6,
     "Notes": "Moderate growth. King Bhumibol died Oct. Brief debt dip."},

    {"Year": 2017, "GDP_Growth_Pct": 3.90, "Household_Debt_GDP_Pct": 78.3,
     "Notes": "Best growth since 2012. New constitution in effect."},

    {"Year": 2018, "GDP_Growth_Pct": 4.19, "Household_Debt_GDP_Pct": 77.8,
     "Notes": "Strongest growth in years. Pre-election spending."},

    {"Year": 2019, "GDP_Growth_Pct": 2.11, "Household_Debt_GDP_Pct": 79.8,
     "Notes": "Growth slowed sharply. Election held Mar. Trade war impact."},

    {"Year": 2020, "GDP_Growth_Pct": -6.05, "Household_Debt_GDP_Pct": 89.3,
     "Notes": "COVID crash. Worst contraction since 1997. Debt surged as GDP denominator shrank."},

    {"Year": 2021, "GDP_Growth_Pct": 1.57, "Household_Debt_GDP_Pct": 94.7,
     "Notes": "Weak recovery. Household debt hit all-time high ~95.5% in Q1. Protests continued."},

    {"Year": 2022, "GDP_Growth_Pct": 2.46, "Household_Debt_GDP_Pct": 91.4,
     "Notes": "Tourism slowly returning. Debt still above 90%."},

    {"Year": 2023, "GDP_Growth_Pct": 1.92, "Household_Debt_GDP_Pct": 91.3,
     "Notes": "Disappointing growth. Election May. Srettha became PM Aug. Debt peaked Q4 at 91.3%."},

    {"Year": 2024, "GDP_Growth_Pct": 2.80, "Household_Debt_GDP_Pct": 88.4,
     "Notes": "Moderate growth. Srettha removed Aug. Paetongtarn became PM. Debt declining slowly."},

    {"Year": 2025, "GDP_Growth_Pct": 2.30, "Household_Debt_GDP_Pct": 86.4,
     "Notes": "Border conflict. Paetongtarn removed. Anutin became PM. Debt at 87.4% Q1, 86.4% Q3."},
]


# =============================================================
# CAMBODIA ECONOMIC DATA
# =============================================================

cambodia_data = [
    # Sources: World Bank, IMF
    # GDP growth: annual % (World Bank)
    # Inflation: annual % (World Bank / IMF)

    {"Year": 2008, "GDP_Growth_Pct": 6.69, "Inflation_Pct": 25.0,
     "Notes": "High growth but extreme inflation from food/fuel crisis."},

    {"Year": 2009, "GDP_Growth_Pct": 0.09, "Inflation_Pct": -0.7,
     "Notes": "Global crisis hit garment exports. Near-zero growth. Deflation."},

    {"Year": 2010, "GDP_Growth_Pct": 6.04, "Inflation_Pct": 4.0,
     "Notes": "Strong rebound. Garment sector recovered."},

    {"Year": 2011, "GDP_Growth_Pct": 7.07, "Inflation_Pct": 5.5,
     "Notes": "Continued strong growth. Construction boom."},

    {"Year": 2012, "GDP_Growth_Pct": 7.26, "Inflation_Pct": 2.9,
     "Notes": "Stable high growth period."},

    {"Year": 2013, "GDP_Growth_Pct": 7.36, "Inflation_Pct": 2.9,
     "Notes": "Contested election. Opposition boycotted. Growth continued."},

    {"Year": 2014, "GDP_Growth_Pct": 7.07, "Inflation_Pct": 3.9,
     "Notes": "Stable. Hun Sen consolidated power."},

    {"Year": 2015, "GDP_Growth_Pct": 7.04, "Inflation_Pct": 1.2,
     "Notes": "Continued high growth. Low inflation."},

    {"Year": 2016, "GDP_Growth_Pct": 6.95, "Inflation_Pct": 3.0,
     "Notes": "Stable growth. Political crackdown beginning."},

    {"Year": 2017, "GDP_Growth_Pct": 7.00, "Inflation_Pct": 2.9,
     "Notes": "CNRP dissolved Nov. Cambodia became de facto one-party state."},

    {"Year": 2018, "GDP_Growth_Pct": 7.47, "Inflation_Pct": 2.5,
     "Notes": "CPP won all seats. Casino/scam economy growing."},

    {"Year": 2019, "GDP_Growth_Pct": 7.05, "Inflation_Pct": 1.9,
     "Notes": "Peak pre-COVID growth. Scam compounds expanding."},

    {"Year": 2020, "GDP_Growth_Pct": -3.10, "Inflation_Pct": 2.9,
     "Notes": "COVID crash. Tourism and garments collapsed."},

    {"Year": 2021, "GDP_Growth_Pct": 3.03, "Inflation_Pct": 2.9,
     "Notes": "Partial recovery. Scam industry exploded post-COVID."},

    {"Year": 2022, "GDP_Growth_Pct": 5.16, "Inflation_Pct": 5.3,
     "Notes": "Recovery continued. Inflation spike from global factors."},

    {"Year": 2023, "GDP_Growth_Pct": 5.58, "Inflation_Pct": 2.1,
     "Notes": "Hun Manet became PM. Scam crackdown pressure growing."},

    {"Year": 2024, "GDP_Growth_Pct": 5.30, "Inflation_Pct": 1.5,
     "Notes": "Moderate growth. OFAC sanctions on Ly Yong Phat. Scam economy under pressure."},

    {"Year": 2025, "GDP_Growth_Pct": 3.50, "Inflation_Pct": 2.0,
     "Notes": "Border conflict. Tourism hit. Scam crackdown intensified. Growth slowed significantly."},
]


# =============================================================
# DISCRETIZE THAILAND
# =============================================================

def discretize_thailand(data):
    df = pd.DataFrame(data)

    # Economic Stress: High / Medium / Low
    # Combines GDP growth and household debt burden
    #
    # High: GDP negative OR GDP < 1% OR household debt > 90%
    #   (Any of these alone creates serious political pressure)
    # Medium: GDP 1-3% OR household debt 80-90%
    # Low: GDP > 3% AND household debt < 80%

    def classify(row):
        gdp = row["GDP_Growth_Pct"]
        debt = row["Household_Debt_GDP_Pct"]

        # High stress triggers
        if gdp < 0:
            return "High"
        if gdp < 1.0:
            return "High"
        if debt > 90:
            return "High"

        # Low stress: need both good growth AND manageable debt
        if gdp > 3.0 and debt < 80:
            return "Low"

        # Everything else
        return "Medium"

    df["Economic_Stress"] = df.apply(classify, axis=1)

    return df


# =============================================================
# DISCRETIZE CAMBODIA
# =============================================================

def discretize_cambodia(data):
    df = pd.DataFrame(data)

    # Cambodia Economic Stress: High / Medium / Low
    # Different thresholds because Cambodia is a developing economy
    # with normally higher growth rates (6-7% baseline)
    #
    # High: GDP negative OR GDP < 2% OR inflation > 10%
    # Medium: GDP 2-5% OR inflation 5-10%
    # Low: GDP > 5% AND inflation < 5%

    def classify(row):
        gdp = row["GDP_Growth_Pct"]
        inf = row["Inflation_Pct"]

        # High stress triggers
        if gdp < 0:
            return "High"
        if gdp < 2.0:
            return "High"
        if inf > 10:
            return "High"

        # Low stress: need both good growth AND stable prices
        if gdp > 5.0 and inf < 5.0:
            return "Low"

        # Everything else
        return "Medium"

    df["Economic_Stress"] = df.apply(classify, axis=1)

    return df


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 60)
    print("ECONOMIC STRESS EXTRACTION")
    print("=" * 60)

    # Thailand
    thai_df = discretize_thailand(thailand_data)

    print("\nTHAILAND")
    print("-" * 60)
    print(thai_df[["Year", "GDP_Growth_Pct", "Household_Debt_GDP_Pct",
                   "Economic_Stress"]].to_string(index=False))

    # Cambodia
    camb_df = discretize_cambodia(cambodia_data)

    print("\nCAMBODIA")
    print("-" * 60)
    print(camb_df[["Year", "GDP_Growth_Pct", "Inflation_Pct",
                   "Economic_Stress"]].to_string(index=False))

    # Save
    out_dir = "../data/training"
    os.makedirs(out_dir, exist_ok=True)

    thai_df.to_csv(os.path.join(out_dir, "S4-thailand_economic_stress.csv"), index=False)
    camb_df.to_csv(os.path.join(out_dir, "S4-cambodia_economic_stress.csv"), index=False)

    print(f"\nSaved to {out_dir}/thailand_economic_stress.csv")
    print(f"Saved to {out_dir}/cambodia_economic_stress.csv")


if __name__ == "__main__":
    main()