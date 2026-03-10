# Border Risk Escalation Model: Thailand-Cambodia

A data-driven AI model that uses Bayesian Networks to analyze geospatial data and determine risk escalation factors at the Thailand-Cambodia border.

## Features

- **Geospatial Data Integration**: Combines conflict events, economic indicators, population density, and satellite imagery
- **Bayesian Network Analysis**: Models causal relationships between risk factors
- **Risk Escalation Scoring**: Computes probabilistic escalation risk per region
- **Interactive Heat Map**: Visualizes risk levels across the border area
- **Real-time Updates**: Integrates with ACLED and other live data sources

## Data Sources

- **ACLED**: Armed Conflict Location & Event Data
- **UCDP**: Uppsala Conflict Data Program
- **OpenStreetMap**: Geospatial boundaries and infrastructure
- **Humanitarian Data Exchange**: Crisis datasets
- **Satellite Imagery**: Sentinel-2, Landsat 8 via Google Earth Engine
- **National Statistics**: Thailand NSO, Cambodia NIS

## Installation

### Requirements
- Python 3.9+
- PostgreSQL/PostGIS (for spatial queries)
- GDAL (for geospatial processing)

### Setup

```bash
git clone https://github.com/Eshih0/border-risk-escalation-model.git
cd border-risk-escalation-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download initial data
python src/data_ingestion.py
```

## Usage

### 1. Data Ingestion & Preprocessing
```bash
python src/data_ingestion.py --sources acled,ucdp,osm
python src/preprocessing.py
```

### 2. Train Bayesian Network
```python
from src.bayesian_network import BorderRiskNetwork

bn = BorderRiskNetwork()
bn.load_data('data/processed/training_data.csv')
bn.build_network()
bn.train()
bn.save_model('models/border_risk_bn.pkl')
```

### 3. Generate Risk Scores & Heat Map
```python
from src.risk_scoring import RiskScorer
from src.visualization import HeatMapGenerator

scorer = RiskScorer(model_path='models/border_risk_bn.pkl')
escalation_scores = scorer.compute_region_scores('data/processed/current_data.csv')

visualizer = HeatMapGenerator(border_geojson='data/borders.geojson')
visualizer.generate_heatmap(escalation_scores, output_path='outputs/heatmap.html')
```

### 4. Run Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## Architecture

### Bayesian Network Structure

```
    Incident Frequency
          |
          v
    Regional Stability <-- Economic Stress
          |                     ^
          v                     |
   Risk Escalation <-- Population Density
          ^                     |
          |                     v
    Troop Concentration   Border Activity
```

### Risk Score Calculation
```
Escalation_Score = P(High Risk | Evidence)
  = Sum of weighted probabilistic inferences from Bayesian Network
  = Normalized to 0-1 scale
```

## Directory Structure

- **data/**: Raw and processed datasets
- **src/**: Core Python modules
- **notebooks/**: Jupyter notebooks for exploration and training
- **tests/**: Unit tests
- **config/**: Configuration files
- **docker/**: Containerization for deployment

## Project Status

- [ ] Data ingestion from ACLED, UCDP
- [ ] Data preprocessing and feature engineering
- [ ] Bayesian Network design & parameterization
- [ ] Risk scoring implementation
- [ ] Heat map visualization
- [ ] API deployment
- [ ] Real-time updates integration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

MIT License

## Contact

Developed by: Eshih0
For questions or collaboration, reach out via GitHub Issues.

---

**Disclaimer**: This model is for research and analytical purposes. Border security assessments should incorporate multiple data sources and expert validation.