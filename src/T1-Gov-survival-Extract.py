"""
Government Survival Probability: Data Extraction
Sources: Bangkok Post, Khaosod English, IPU Parline, Wikipedia

Excludes: Constitutional Court petitions/rulings (already in bloc_cohesion.csv)

Tracks: No-confidence motions, coalition defections, coalition seat margin
Monthly resolution for key event periods, quarterly for stable periods.

Saves to: ../data/training/T1-gov_survival.csv

Discretization:
- Low survival (High threat):  Active no-confidence + coalition defection OR seat margin < 20
- Medium survival: No-confidence filed but expected to survive OR margin 20-50
- High survival (Low threat): No active threats, comfortable majority
"""

import pandas as pd
import os


# =============================================================
# DATA: Monthly government survival events
# =============================================================
# Only includes months where something changed or key stable-period
# snapshots. Not every single month 2008-2025.
#
# Coalition seat margins are for the lower house (500 seats,
# majority = 251). Margin = coalition seats - 251.
#
# Sources: Wikipedia (Thai political crisis pages, election pages),
# Bangkok Post, Khaosod English, The Diplomat, Al Jazeera, IPU Parline

data = [
    # === SAMAK SUNDARAVEJ (Jan-Sep 2008) ===
    {"Year": 2008, "Month": 1, "PM": "Samak Sundaravej",
     "Coalition_Seats": 315, "Margin": 64,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "PPP-led coalition formed after Dec 2007 election."},
    {"Year": 2008, "Month": 6, "PM": "Samak Sundaravej",
     "Coalition_Seats": 315, "Margin": 64,
     "No_Confidence_Filed": 1, "Coalition_Defection": 0,
     "Notes": "Opposition filed no-confidence motion in late June. Government survived."},
    {"Year": 2008, "Month": 9, "PM": "Samak Sundaravej",
     "Coalition_Seats": 315, "Margin": 64,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Samak removed by court Sep 9 (cooking show). NOT a no-confidence event."},

    # === SOMCHAI WONGSAWAT (Sep-Dec 2008) ===
    {"Year": 2008, "Month": 10, "PM": "Somchai Wongsawat",
     "Coalition_Seats": 310, "Margin": 59,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Somchai took over from Samak. PAD airport seizures Nov."},
    {"Year": 2008, "Month": 12, "PM": "Somchai Wongsawat",
     "Coalition_Seats": 0, "Margin": -251,
     "No_Confidence_Filed": 0, "Coalition_Defection": 1,
     "Notes": "PPP dissolved by court Dec 2. Coalition collapsed. Mass defection to form Abhisit govt."},

    # === ABHISIT VEJJAJIVA (Dec 2008 - Aug 2011) ===
    {"Year": 2009, "Month": 3, "PM": "Abhisit Vejjajiva",
     "Coalition_Seats": 262, "Margin": 11,
     "No_Confidence_Filed": 1, "Coalition_Defection": 0,
     "Notes": "First no-confidence motion filed by Puea Thai. Abhisit survived by ~53 votes."},
    {"Year": 2009, "Month": 6, "PM": "Abhisit Vejjajiva",
     "Coalition_Seats": 262, "Margin": 11,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Stable period. Red Shirt protests in Apr dispersed."},
    {"Year": 2010, "Month": 3, "PM": "Abhisit Vejjajiva",
     "Coalition_Seats": 260, "Margin": 9,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Red Shirt protests escalating but no parliamentary threat."},
    {"Year": 2010, "Month": 6, "PM": "Abhisit Vejjajiva",
     "Coalition_Seats": 260, "Margin": 9,
     "No_Confidence_Filed": 1, "Coalition_Defection": 0,
     "Notes": "No-confidence debate. Survived. Post-crackdown period."},
    {"Year": 2011, "Month": 3, "PM": "Abhisit Vejjajiva",
     "Coalition_Seats": 258, "Margin": 7,
     "No_Confidence_Filed": 1, "Coalition_Defection": 0,
     "Notes": "Third no-confidence motion. Survived. Margin thin."},
    {"Year": 2011, "Month": 5, "PM": "Abhisit Vejjajiva",
     "Coalition_Seats": 258, "Margin": 7,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "House dissolved May 10. Election called for July."},

    # === YINGLUCK SHINAWATRA (Aug 2011 - May 2014) ===
    {"Year": 2011, "Month": 8, "PM": "Yingluck Shinawatra",
     "Coalition_Seats": 300, "Margin": 49,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Pheu Thai won landslide. 6-party coalition. Strong majority."},
    {"Year": 2012, "Month": 6, "PM": "Yingluck Shinawatra",
     "Coalition_Seats": 300, "Margin": 49,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Stable. No significant threats."},
    {"Year": 2012, "Month": 11, "PM": "Yingluck Shinawatra",
     "Coalition_Seats": 300, "Margin": 49,
     "No_Confidence_Filed": 1, "Coalition_Defection": 0,
     "Notes": "No-confidence debate Nov 26-28. Yingluck survived easily with 3/5 majority."},
    {"Year": 2013, "Month": 6, "PM": "Yingluck Shinawatra",
     "Coalition_Seats": 295, "Margin": 44,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Stable. Rice pledging scheme controversy building."},
    {"Year": 2013, "Month": 11, "PM": "Yingluck Shinawatra",
     "Coalition_Seats": 295, "Margin": 44,
     "No_Confidence_Filed": 1, "Coalition_Defection": 0,
     "Notes": "No-confidence filed over amnesty bill. Anti-govt protests began. Yingluck survived vote."},
    {"Year": 2013, "Month": 12, "PM": "Yingluck Shinawatra",
     "Coalition_Seats": 295, "Margin": 44,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "House dissolved Dec 9. Caretaker government."},
    {"Year": 2014, "Month": 3, "PM": "Yingluck Shinawatra (caretaker)",
     "Coalition_Seats": 0, "Margin": 0,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Caretaker. Feb election boycotted, annulled by court."},

    # === PRAYUTH CHAN-O-CHA (May 2014 - Jul 2019, elected PM 2019-Aug 2022) ===
    # Junta period: no parliament, no no-confidence possible
    {"Year": 2014, "Month": 6, "PM": "Prayuth Chan-o-cha (junta)",
     "Coalition_Seats": 0, "Margin": 0,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Military junta. No parliament. N/A for survival probability."},
    {"Year": 2016, "Month": 6, "PM": "Prayuth Chan-o-cha (junta)",
     "Coalition_Seats": 0, "Margin": 0,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Military junta. No parliament."},
    {"Year": 2018, "Month": 6, "PM": "Prayuth Chan-o-cha (junta)",
     "Coalition_Seats": 0, "Margin": 0,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Military junta. Pre-election."},

    # Post-2019 election Prayuth
    {"Year": 2019, "Month": 7, "PM": "Prayuth Chan-o-cha",
     "Coalition_Seats": 254, "Margin": 3,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Elected PM via Senate vote. Razor-thin margin in House."},
    {"Year": 2020, "Month": 2, "PM": "Prayuth Chan-o-cha",
     "Coalition_Seats": 254, "Margin": 3,
     "No_Confidence_Filed": 1, "Coalition_Defection": 0,
     "Notes": "No-confidence debate Feb. Prayuth survived. FFP dissolved same month."},
    {"Year": 2020, "Month": 9, "PM": "Prayuth Chan-o-cha",
     "Coalition_Seats": 270, "Margin": 19,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Gained seats from FFP dissolution fallout. Protests building."},
    {"Year": 2021, "Month": 2, "PM": "Prayuth Chan-o-cha",
     "Coalition_Seats": 268, "Margin": 17,
     "No_Confidence_Filed": 1, "Coalition_Defection": 0,
     "Notes": "No-confidence motion Feb. Survived."},
    {"Year": 2021, "Month": 9, "PM": "Prayuth Chan-o-cha",
     "Coalition_Seats": 265, "Margin": 14,
     "No_Confidence_Filed": 1, "Coalition_Defection": 0,
     "Notes": "Another no-confidence. Survived but margin eroding."},
    {"Year": 2022, "Month": 7, "PM": "Prayuth Chan-o-cha",
     "Coalition_Seats": 260, "Margin": 9,
     "No_Confidence_Filed": 1, "Coalition_Defection": 0,
     "Notes": "No-confidence debate Jul 18-22. Survived. Last censure before term expiry."},
    {"Year": 2022, "Month": 8, "PM": "Prayuth Chan-o-cha",
     "Coalition_Seats": 260, "Margin": 9,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Suspended by court Aug (term limit). Returned Sep after favorable ruling."},

    # === SRETTHA THAVISIN (Aug 2023 - Aug 2024) ===
    {"Year": 2023, "Month": 8, "PM": "Srettha Thavisin",
     "Coalition_Seats": 314, "Margin": 63,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "11-party coalition formed. Large majority. MFP in opposition."},
    {"Year": 2024, "Month": 3, "PM": "Srettha Thavisin",
     "Coalition_Seats": 310, "Margin": 59,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Stable. Some minor coalition friction but no defections."},
    {"Year": 2024, "Month": 6, "PM": "Srettha Thavisin",
     "Coalition_Seats": 310, "Margin": 59,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Approval declining (12.85%) but coalition holding."},

    # === PAETONGTARN SHINAWATRA (Aug 2024 - Aug 2025) ===
    {"Year": 2024, "Month": 9, "PM": "Paetongtarn Shinawatra",
     "Coalition_Seats": 305, "Margin": 54,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Took over from Srettha. MFP dissolved Aug 7. Coalition held."},
    {"Year": 2024, "Month": 12, "PM": "Paetongtarn Shinawatra",
     "Coalition_Seats": 300, "Margin": 49,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Thamanat faction (19 MPs) expelled from PPRP Dec. Coalition gradually shifting."},
    {"Year": 2025, "Month": 3, "PM": "Paetongtarn Shinawatra",
     "Coalition_Seats": 292, "Margin": 41,
     "No_Confidence_Filed": 1, "Coalition_Defection": 0,
     "Notes": "People's Party filed no-confidence motion Mar. Paetongtarn survived vote Mar 26."},
    {"Year": 2025, "Month": 5, "PM": "Paetongtarn Shinawatra",
     "Coalition_Seats": 292, "Margin": 41,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Border skirmish May 28. Tensions rising with Bhumjaithai over Interior Ministry."},
    {"Year": 2025, "Month": 6, "PM": "Paetongtarn Shinawatra",
     "Coalition_Seats": 223, "Margin": -28,
     "No_Confidence_Filed": 1, "Coalition_Defection": 1,
     "Notes": "Hun Sen phone call leaked Jun 18. Bhumjaithai (69 seats) withdrew Jun 19. No-confidence announced Jun 24. Coalition lost majority."},
    {"Year": 2025, "Month": 7, "PM": "Paetongtarn Shinawatra (suspended)",
     "Coalition_Seats": 254, "Margin": 3,
     "No_Confidence_Filed": 1, "Coalition_Defection": 1,
     "Notes": "Paetongtarn suspended by court Jul 1. Remaining 6 parties held at ~254 seats. Caretaker Phumtham."},
    {"Year": 2025, "Month": 8, "PM": "Paetongtarn Shinawatra (removed)",
     "Coalition_Seats": 0, "Margin": 0,
     "No_Confidence_Filed": 0, "Coalition_Defection": 1,
     "Notes": "Paetongtarn removed Aug 29. Government collapsed. People's Party backed Anutin."},

    # === ANUTIN CHARNVIRAKUL (Sep 2025 - Feb 2026) ===
    {"Year": 2025, "Month": 9, "PM": "Anutin Charnvirakul",
     "Coalition_Seats": 195, "Margin": -56,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "Minority govt. People's Party confidence and supply. Required to dissolve House within 4 months."},
    {"Year": 2025, "Month": 12, "PM": "Anutin Charnvirakul",
     "Coalition_Seats": 195, "Margin": -56,
     "No_Confidence_Filed": 0, "Coalition_Defection": 0,
     "Notes": "House dissolved Dec 12. Caretaker PM. Election Feb 8 2026."},
]


# =============================================================
# DISCRETIZE
# =============================================================

def discretize(df):
    """
    Gov_Survival:
    - Low:  Active no-confidence + defection, OR margin < 0 (lost majority),
            OR junta (no parliament = no civilian survival concept)
    - Medium: No-confidence filed but coalition holding, OR thin margin (0-20)
    - High: No active threats, comfortable margin (> 20)
    """
    def classify(row):
        ncf = row["No_Confidence_Filed"]
        defection = row["Coalition_Defection"]
        margin = row["Margin"]
        pm = row["PM"]

        # Junta periods: no civilian government to survive
        if "junta" in pm.lower():
            return "N/A"

        # Caretaker / collapsed
        if margin < 0:
            return "Low"

        # Active no-confidence AND defection = extreme threat
        if ncf == 1 and defection == 1:
            return "Low"

        # Defection alone (major party leaving)
        if defection == 1:
            return "Low"

        # No-confidence filed but coalition holding
        if ncf == 1 and margin > 20:
            return "Medium"

        # No-confidence with thin margin
        if ncf == 1 and margin <= 20:
            return "Low"

        # No threats but thin margin
        if margin <= 20:
            return "Medium"

        # Comfortable majority, no active threats
        return "High"

    df["Gov_Survival"] = df.apply(classify, axis=1)
    return df


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 60)
    print("GOVERNMENT SURVIVAL PROBABILITY")
    print("=" * 60)

    df = pd.DataFrame(data)
    df = discretize(df)

    print(df[["Year", "Month", "PM", "Coalition_Seats", "Margin",
              "No_Confidence_Filed", "Coalition_Defection",
              "Gov_Survival"]].to_string(index=False))

    # Save
    out_dir = "../data/training"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "T1-gov_survival.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()