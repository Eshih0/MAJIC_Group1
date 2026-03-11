"""
Royalist-Military Bloc Cohesion: Data Extraction
Sources: iLaw / TLHR (lese majeste prosecutions), Bangkok Post / Wikipedia (Constitutional Court rulings)

Saves to: ../data/training/S3-bloc_cohesion.csv

Logic:
- High lese majeste prosecution rate = establishment actively enforcing political control = High cohesion
- Constitutional Court rulings against elected government/opposition = establishment aligned and acting = High cohesion
- Low prosecutions + court not targeting government = Low cohesion (bloc is passive or fractured)

Discretization:
- High: Establishment bloc is unified, actively targeting elected government or opposition
- Medium: Some enforcement activity but not at peak levels
- Low: Bloc is passive, fractured, or the government IS the military (no need to enforce)
"""

import pandas as pd
import os

# =============================================================
# RAW DATA: Lese Majeste Prosecutions per Year
# =============================================================
# Sources: iLaw, TLHR, FIDH, Attorney General's Office,
# Wikipedia (List of prosecuted lese majeste cases),
# Cambridge Law Journal, Political Prisoners in Thailand blog

lese_majeste_data = [
    # Pre-2006: ~4 prosecutions/year average (Attorney General 1984-2000)
    {"Year": 2008, "LM_Prosecutions_Est": 77,  "LM_Notes": "Post-2006 coup surge. 4800+ webpages blocked. iLaw/TLHR data."},
    {"Year": 2009, "LM_Prosecutions_Est": 80,  "LM_Notes": "Continued high enforcement under Abhisit government. Multiple cases tied to Red Shirt movement."},
    {"Year": 2010, "LM_Prosecutions_Est": 478, "LM_Notes": "Wikipedia/iLaw: 478 cases in 2010. Peak year. Red Shirt crackdown. 60,000 websites banned."},
    {"Year": 2011, "LM_Prosecutions_Est": 100, "LM_Notes": "Remained high early year. Declined after Yingluck took office Aug 2011."},
    {"Year": 2012, "LM_Prosecutions_Est": 30,  "LM_Notes": "Declined significantly under Yingluck. She said she would not seek to reform the law but enforcement dropped."},
    {"Year": 2013, "LM_Prosecutions_Est": 25,  "LM_Notes": "Continued low under Yingluck. Conviction rate ~76% (UN OHCHR)."},
    {"Year": 2014, "LM_Prosecutions_Est": 50,  "LM_Notes": "May 2014 coup. NCPO granted military tribunal authority. 127 arrested May 2014-Jul 2017. Spike post-coup."},
    {"Year": 2015, "LM_Prosecutions_Est": 45,  "LM_Notes": "Junta enforcement. 60-year sentence for Facebook posts (Pongsak). Conviction rate jumped to ~96%."},
    {"Year": 2016, "LM_Prosecutions_Est": 55,  "LM_Notes": "King Bhumibol died Oct 2016. Spike in prosecutions late year. Jatupat arrested for sharing BBC article."},
    {"Year": 2017, "LM_Prosecutions_Est": 40,  "LM_Notes": "Multiple major cases (Pai Dao Din, Somsak Facebook share cases). TLHR documented several."},
    {"Year": 2018, "LM_Prosecutions_Est": 5,   "LM_Notes": "TLHR: No new Section 112 cases prosecuted in 2018. Courts dismissed several existing cases. Dramatic drop."},
    {"Year": 2019, "LM_Prosecutions_Est": 5,   "LM_Notes": "Enforcement remained low. Post-election period. Prayuth continued as PM via Senate."},
    {"Year": 2020, "LM_Prosecutions_Est": 10,  "LM_Notes": "Low until Nov 2020. Prayuth pledged to enforce 'all laws' against pro-democracy protesters in late Nov."},
    {"Year": 2021, "LM_Prosecutions_Est": 130, "LM_Notes": "Massive surge. 259 individuals charged Nov 2020-Oct 2023 per FIDH. 43-year sentence handed down."},
    {"Year": 2022, "LM_Prosecutions_Est": 90,  "LM_Notes": "Continued high enforcement against pro-democracy movement. FIDH/iLaw tracking."},
    {"Year": 2023, "LM_Prosecutions_Est": 80,  "LM_Notes": "100 verdicts reached by Oct 2023 (FIDH). 79% conviction rate. Arnon Nampa convicted Sep 2023."},
    {"Year": 2024, "LM_Prosecutions_Est": 50,  "LM_Notes": "Move Forward dissolved Aug 2024 for advocating Section 112 reform. Srettha removed Aug 2024."},
    {"Year": 2025, "LM_Prosecutions_Est": 30,  "LM_Notes": "Paetongtarn removed Aug 2025. Focus shifted to border conflict. Enforcement continued but attention diverted."},
]

# =============================================================
# RAW DATA: Constitutional Court Rulings Against Government/Opposition
# =============================================================
# Sources: Wikipedia (Constitutional Court of Thailand),
# East Asia Forum, HRW, House of Commons Library, Al Jazeera

court_rulings_data = [
    {"Year": 2007, "Court_Actions": 1, "Court_Direction": "Anti-Elected",   "Court_Notes": "Thai Rak Thai dissolved. Executives banned. Junta-appointed tribunal."},
    {"Year": 2008, "Court_Actions": 2, "Court_Direction": "Anti-Elected",   "Court_Notes": "PM Samak banned (cooking show). Coalition dissolved Dec 2008. 109 executives banned. Led to Abhisit PM."},
    {"Year": 2009, "Court_Actions": 0, "Court_Direction": "Neutral",        "Court_Notes": "No major anti-government rulings. Abhisit (Democrat) in power with establishment support."},
    {"Year": 2010, "Court_Actions": 0, "Court_Direction": "Neutral",        "Court_Notes": "No major rulings. Red Shirt crackdown handled by military, not courts."},
    {"Year": 2011, "Court_Actions": 0, "Court_Direction": "Neutral",        "Court_Notes": "Yingluck won landslide July. No immediate court action."},
    {"Year": 2012, "Court_Actions": 0, "Court_Direction": "Neutral",        "Court_Notes": "No major rulings against government."},
    {"Year": 2013, "Court_Actions": 1, "Court_Direction": "Anti-Elected",   "Court_Notes": "Court ruled against constitutional amendment attempt by Yingluck government."},
    {"Year": 2014, "Court_Actions": 1, "Court_Direction": "Anti-Elected",   "Court_Notes": "PM Yingluck removed May 2014 (national security chief transfer). Coup followed."},
    {"Year": 2015, "Court_Actions": 0, "Court_Direction": "Pro-Establishment", "Court_Notes": "Junta in power. No need for court to act against government."},
    {"Year": 2016, "Court_Actions": 0, "Court_Direction": "Pro-Establishment", "Court_Notes": "Junta in power. Court approved new constitution."},
    {"Year": 2017, "Court_Actions": 0, "Court_Direction": "Pro-Establishment", "Court_Notes": "Junta in power. New constitution took effect."},
    {"Year": 2018, "Court_Actions": 0, "Court_Direction": "Pro-Establishment", "Court_Notes": "Junta in power. No oppositional rulings needed."},
    {"Year": 2019, "Court_Actions": 2, "Court_Direction": "Anti-Opposition", "Court_Notes": "Thai Raksa Chart dissolved (Feb). Prayuth ruled eligible to continue as PM (Sep)."},
    {"Year": 2020, "Court_Actions": 1, "Court_Direction": "Anti-Opposition", "Court_Notes": "Future Forward Party dissolved (Feb). Thanathorn banned 10 years."},
    {"Year": 2021, "Court_Actions": 0, "Court_Direction": "Neutral",        "Court_Notes": "No major dissolutions. Protest prosecutions via criminal courts, not Constitutional Court."},
    {"Year": 2022, "Court_Actions": 1, "Court_Direction": "Mixed",          "Court_Notes": "Prayuth suspended Aug (term limit case). Returned Sep after favorable ruling."},
    {"Year": 2023, "Court_Actions": 1, "Court_Direction": "Anti-Opposition", "Court_Notes": "MFP blocked from forming government. Pita suspended for media shares case."},
    {"Year": 2024, "Court_Actions": 2, "Court_Direction": "Anti-Elected",   "Court_Notes": "Move Forward dissolved Aug 7. PM Srettha removed Aug 14. Two major rulings in one week."},
    {"Year": 2025, "Court_Actions": 1, "Court_Direction": "Anti-Elected",   "Court_Notes": "PM Paetongtarn removed Aug 29 after Hun Sen phone call crisis."},
]


# =============================================================
# DISCRETIZE: Combine into Bloc Cohesion score
# =============================================================

def compute_bloc_cohesion(lm_row, court_row):
    """
    Combine lese majeste prosecution intensity and court ruling direction
    into a single Bloc_Cohesion state: High / Medium / Low.

    High cohesion = establishment is unified and actively using institutions
                    against elected government or popular opposition
    Medium = some activity but not at peak, or bloc is in maintenance mode
    Low = bloc is passive, fractured, or enforcement has dropped off
    """
    lm_count = lm_row["LM_Prosecutions_Est"]
    court_dir = court_row["Court_Direction"]
    court_actions = court_row["Court_Actions"]

    # Score from lese majeste prosecutions
    if lm_count >= 80:
        lm_score = 2    # Heavy enforcement
    elif lm_count >= 25:
        lm_score = 1    # Moderate enforcement
    else:
        lm_score = 0    # Low enforcement

    # Score from court rulings
    if court_dir in ["Anti-Elected", "Anti-Opposition"] and court_actions >= 1:
        court_score = 2  # Court actively targeting elected government or opposition
    elif court_dir == "Pro-Establishment":
        court_score = 1  # Bloc in power, no need to act (still cohesive)
    elif court_dir == "Mixed":
        court_score = 1
    else:
        court_score = 0  # Neutral or no significant rulings

    total = lm_score + court_score

    if total >= 3:
        return "High"
    elif total >= 1:
        return "Medium"
    else:
        return "Low"


def main():
    # Build dataframes
    lm_df = pd.DataFrame(lese_majeste_data)
    court_df = pd.DataFrame(court_rulings_data)

    # Merge on year
    merged = lm_df.merge(court_df, on="Year", how="inner")

    # Compute discretized state
    merged["Bloc_Cohesion"] = merged.apply(
        lambda row: compute_bloc_cohesion(row, row), axis=1
    )

    # Build output
    output = merged[[
        "Year",
        "LM_Prosecutions_Est",
        "Court_Actions",
        "Court_Direction",
        "Bloc_Cohesion",
        "LM_Notes",
        "Court_Notes"
    ]].copy()

    # Print
    print("=" * 60)
    print("BLOC COHESION DATA")
    print("=" * 60)
    print(output[["Year", "LM_Prosecutions_Est", "Court_Actions",
                  "Court_Direction", "Bloc_Cohesion"]].to_string(index=False))

    # Save
    out_dir = "../data/training"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "S3-bloc_cohesion.csv")
    output.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()