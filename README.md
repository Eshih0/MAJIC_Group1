# Thailand-Cambodia Border Conflict Prediction Model

A Bayesian Network model that predicts the probability of border military action between Thailand and Cambodia across four geographic zones, using political, institutional, economic, and geospatial data.

## Overview

The model is built around a central empirical finding: **Thai civilian government weakness is the dominant leading indicator of conflict.** Every major escalation since 2008 has occurred during periods of political fragility or transition. All other variables function as inputs into that weakness assessment or as conditional multipliers.

The border is divided into four prediction zones:

- **Zone 1** (Northeast Dangrek): Preah Vihear Temple, Emerald Triangle, Phu Makhua Hill, Nam Yuen District
- **Zone 2** (Western Dangrek): Prasat Ta Muen Thom, Ta Krabey, Ta Khwai, O'Smach
- **Zone 3** (Central Border): Poipet/Sa Kaeo, Serei Saophoan, Prey Chan Village
- **Zone 4** (Southeast Coast/Maritime): Ban Chamrak/Trat, Thma Da/Pursat, Khlong Yai, Ko Kut

Each zone has its own model with zone-specific conflict outcome coding, seasonal multipliers, and scam economy weights, while sharing national-level political inputs.

## Model Architecture

The network is structured in three layers:

**Structural Layer** (annual updates): Military Autonomy, Civilian Government Legitimacy, Royalist-Military Bloc Cohesion, Economic Stress

**Trigger Layer** (weekly/event-driven updates): Government Survival Probability, Constitutional Court Activity, Military-Civilian Friction Events, Nationalist Sentiment, Cambodia Provocation Signal, Bilateral Channel Health

**Output**: P(Border Military Action Within 6 Months) per zone

A central **Government Weakness** variable (Stable / Fragile / Collapsed-Caretaker) sits between the structural and trigger layers and the output. Calibrated against 34 historical six-month windows since 2008 with an 18% base rate.

## Data Sources

### Thailand (Structural)
| Variable | Source | URL |
|----------|--------|-----|
| Military Autonomy | V-Dem Dataset | v-dem.net |
| Government Legitimacy | NIDA Poll, IPU Parline | nidapoll.nida.ac.th/en, data.ipu.org |
| Bloc Cohesion | iLaw (lese majeste data), Bangkok Post (court rulings) | ilaw.or.th/en, bangkokpost.com |
| Economic Stress | Bank of Thailand, FRED | bot.or.th/en/statistics.html, fred.stlouisfed.org |

### Cambodia (Provocation Baseline)
| Variable | Source | URL |
|----------|--------|-----|
| Regime Consolidation | V-Dem Dataset | v-dem.net |
| Violence Capacity | V-Dem Dataset | v-dem.net |
| Corruption Environment | V-Dem Dataset | v-dem.net |

### Trigger Layer
| Variable | Source | URL |
|----------|--------|-----|
| Conflict Events | ACLED | acleddata.com |
| Court Activity | Constitutional Court of Thailand, Prachatai | constitutionalcourt.or.th, prachatai.com/english |
| Nationalist Sentiment | X/Twitter API, Bangkok Post | bangkokpost.com/opinion |
| Cambodia Provocation | Sentinel-2 imagery, Hun Sen Facebook, ICJ tracker | browser.dataspace.copernicus.eu, icj-cij.org/cases |
| Bilateral Channels | Thai MFA, Cambodian MFA, Khmer Times | mfa.go.th/en, mfaic.gov.kh, khmertimeskh.com |
| Scam Economy | OFAC Sanctions, US TIP Report, UNODC | sanctionssearch.ofac.treas.gov, unodc.org |

## Installation

### Requirements
- Python 3.9+
- pgmpy
- pandas
- numpy

### Setup

```bash
git clone https://github.com/Eshih0/MAJIC_Group1.git
cd MAJIC_Group1

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download V-Dem Country-Year dataset from [v-dem.net](https://v-dem.net/data/the-v-dem-dataset/) (select "Country-Year: V-Dem Full+Others")
2. Save the CSV to `data/`
3. Note: The V-Dem CSV is too large for GitHub. It is listed in `.gitignore`.

## Usage

### 1. Extract and Discretize V-Dem Data
```bash
cd src
python vdem_extract.py
```
Outputs `thailand_values.csv` and `cambodia_values.csv` with raw scores and discretized High/Medium/Low states.

### 2. Extract Bloc Cohesion Data
```bash
python bloc_cohesion_extract.py
```
Outputs `data/training/bloc_cohesion.csv` with lese majeste prosecution counts, Constitutional Court ruling direction, and discretized Bloc_Cohesion states.

### 3. Build and Query the Bayesian Network
```python
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# Load training data, define network structure, fit CPTs
# See src/ scripts for full implementation
```

## Directory Structure

```
MAJIC_Group1/
  data/
    training/              # Discretized training CSVs
      bloc_cohesion.csv
    V-Dem-CY-Full+Others-v15.csv  # (gitignored, download manually)
  src/
    vdem_extract.py        # V-Dem data extraction for Thailand + Cambodia
    bloc_cohesion_extract.py  # Bloc cohesion from iLaw + court rulings
  docs/
    Thailand_Cambodia_BN_Node_Reference.docx  # Full node definitions and causal logic
    Source_Zone_Assignment_Matrix.docx         # Source-to-zone mapping
  notebooks/               # Jupyter notebooks for exploration
  .gitignore
  README.md
  requirements.txt
```

## Zone-Specific Considerations

| Factor | Zone 1 | Zone 2 | Zone 3 | Zone 4 |
|--------|--------|--------|--------|--------|
| Terrain | Highland/mountain | Highland/mountain | Lowland/plains | Coastal/maritime |
| Dry season effect | Strong | Strong | Moderate | Complex (ground vs naval) |
| Scam compounds | None | High (O'Smach) | Highest (Poipet) | High (Koh Kong) |
| ICJ relevance | Yes | Yes | No | No |
| Military command | 2nd Army Region | 2nd Army Region | 2nd Army Region | Chanthaburi-Trat Command + RTN |
| Historical base rate | High (conflict in 2008-2011, 2025) | High (conflict in 2008-2011, 2025) | Low (conflict only in 2025) | Low (conflict only in 2025) |

## Calibration

- **Training window**: January 2008 to June 2026 (34 six-month windows)
- **Positive windows**: 5 (Oct 2008-Mar 2009, Oct 2010-Mar 2011, Apr-Sep 2011, Feb-Jul 2025, Oct 2025-Mar 2026)
- **Base rate**: ~18%
- **Estimator**: BayesianEstimator with Dirichlet priors (appropriate for small N)
- **Validation**: Leave-one-out cross-validation, Brier score target < 0.12


## Team

MAJIC Group 1

## License

MIT License

---

**Disclaimer**: This model is for academic and research purposes. Border security assessments should incorporate multiple data sources and expert validation.
