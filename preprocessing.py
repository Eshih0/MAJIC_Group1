"""
Data Preprocessing Module
Cleans, normalizes, and prepares data for Bayesian Network training.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data cleaning, normalization, and feature engineering."""
    
    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed'):
        """Initialize preprocessor."""
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(processed_data_dir, exist_ok=True)
    
    def load_acled_data(self):
        """Load and clean ACLED conflict event data."""
        logger.info("Loading ACLED data...")
        
        acled_path = os.path.join(self.raw_data_dir, 'acled_events.csv')
        
        try:
            df = pd.read_csv(acled_path)
            
            # Convert date column
            df['event_date'] = pd.to_datetime(df['event_date'])
            
            # Handle missing values
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)
            
            # Remove rows with missing coordinates
            df = df.dropna(subset=['latitude', 'longitude'])
            
            logger.info(f"Loaded {len(df)} ACLED records")
            
            return df
            
        except FileNotFoundError:
            logger.warning("ACLED data file not found. Using sample data.")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration."""
        logger.info("Creating sample data for demonstration...")
        
        # Sample locations near Thailand-Cambodia border
        border_locations = [
            {'lat': 14.2, 'lon': 102.8},  # Surin-Oddar Meanchey
            {'lat': 14.5, 'lon': 102.5},  # Sisaket-Oddar Meanchey
            {'lat': 15.2, 'lon': 102.3},  # Yasothon-Oddar Meanchey
            {'lat': 13.8, 'lon': 104.8},  # Trat-Koh Kong
            {'lat': 13.5, 'lon': 104.9},  # Trat coast
        ]
        
        records = []
        for _ in range(100):
            loc = np.random.choice(border_locations)
            records.append({
                'event_id_cnty': f'evt_{_}',
                'event_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365)),
                'event_type': np.random.choice(['Battles', 'Violence against civilians', 'Strategic developments']),
                'country': np.random.choice(['Thailand', 'Cambodia']),
                'latitude': loc['lat'] + np.random.normal(0, 0.2),
                'longitude': loc['lon'] + np.random.normal(0, 0.2),
                'fatalities': np.random.poisson(2),
                'actor1': 'Military',
                'actor2': 'Military'
            })
        
        return pd.DataFrame(records)
    
    def aggregate_by_region(self, df, grid_size_km=5):
        """
        Aggregate conflict events into geographic grid cells.
        
        Args:
            df: DataFrame with latitude/longitude
            grid_size_km: Size of grid cells in kilometers
            
        Returns:
            GeoDataFrame with aggregated features
        """
        logger.info(f"Aggregating data into {grid_size_km}km grid cells...")
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
            crs='EPSG:4326'
        )
        
        # Create grid (simplified approach using lat/lon rounding)
        # For production, use actual grid cells
        grid_resolution = 1.0  # ~111km at equator, adjust for border region
        
        gdf['grid_lat'] = (gdf['latitude'] // grid_resolution * grid_resolution)
        gdf['grid_lon'] = (gdf['longitude'] // grid_resolution * grid_resolution)
        
        # Aggregate by grid cell
        aggregated = gdf.groupby(['grid_lat', 'grid_lon']).agg({
            'event_id_cnty': 'count',
            'fatalities': 'sum',
            'event_type': lambda x: len(set(x)),
            'event_date': ['min', 'max']
        }).reset_index()
        
        aggregated.columns = ['grid_lat', 'grid_lon', 'incident_count', 
                             'total_fatalities', 'event_type_diversity', 
                             'first_event', 'last_event']
        
        # Recreate geometry
        aggregated['geometry'] = gpd.points_from_xy(
            aggregated['grid_lon'], 
            aggregated['grid_lat']
        )
        aggregated = gpd.GeoDataFrame(aggregated, crs='EPSG:4326')
        
        logger.info(f"Created {len(aggregated)} grid cells")
        
        return aggregated
    
    def compute_features(self, df_regional):
        """
        Compute features for Bayesian Network from regional data.
        
        Args:
            df_regional: Regional aggregated data
            
        Returns:
            DataFrame with computed features
        """
        logger.info("Computing Bayesian Network features...")
        
        features = df_regional.copy()
        
        # Feature 1: Incident Frequency (discretized)
        features['incident_frequency'] = pd.cut(
            features['incident_count'],
            bins=3,
            labels=['low', 'medium', 'high']
        )
        
        # Feature 2: Fatality severity
        features['fatality_severity'] = pd.cut(
            features['total_fatalities'],
            bins=[0, 5, 15, np.inf],
            labels=['low', 'medium', 'high']
        )
        
        # Feature 3: Temporal trend (recent vs. historical)
        features['days_since_last_event'] = (
            datetime.now() - pd.to_datetime(features['last_event'])
        ).dt.days
        
        features['temporal_trend'] = pd.cut(
            features['days_since_last_event'],
            bins=[0, 30, 90, np.inf],
            labels=['active', 'moderate', 'dormant']
        )
        
        # Feature 4: Event diversity
        features['event_diversity'] = pd.cut(
            features['event_type_diversity'],
            bins=2,
            labels=['uniform', 'diverse']
        )
        
        # Mock features (in production, integrate actual data sources)
        features['economic_stress'] = np.random.choice(['low', 'medium', 'high'], len(features))
        features['population_density'] = np.random.choice(['sparse', 'moderate', 'dense'], len(features))
        features['troop_concentration'] = np.random.choice(['low', 'medium', 'high'], len(features))
        features['border_activity'] = np.random.choice(['low', 'medium', 'high'], len(features))
        
        # Create target variable for training (based on expert judgment or known events)
        features['risk_escalation'] = np.random.choice(['low', 'medium', 'high'], len(features))
        
        logger.info(f"Computed features for {len(features)} regions")
        
        return features
    
    def save_processed_data(self, df, filename):
        """Save processed data to CSV."""
        output_path = os.path.join(self.processed_data_dir, filename)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        return output_path

def main():
    """Main entry point for preprocessing."""
    preprocessor = DataPreprocessor()
    
    print("\n=== Starting Data Preprocessing ===\n")
    
    # Load ACLED data
    df_acled = preprocessor.load_acled_data()
    
    # Aggregate by region
    df_regional = preprocessor.aggregate_by_region(df_acled, grid_size_km=5)
    
    # Compute features
    df_features = preprocessor.compute_features(df_regional)
    
    # Save
    preprocessor.save_processed_data(df_features, 'border_risk_features.csv')
    
    print("\n=== Data Preprocessing Complete ===\n")
    print(df_features.head())

if __name__ == '__main__':
    main()