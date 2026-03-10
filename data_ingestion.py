"""
Data Ingestion Module
Downloads and consolidates data from multiple sources:
- ACLED (Armed Conflict Location & Event Data)
- UCDP (Uppsala Conflict Data Program)
- OpenStreetMap
- Satellite imagery metadata
"""

import requests
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """Handles downloading and ingesting geospatial data from multiple sources."""
    
    def __init__(self, config_path='config/config.yaml'):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.raw_data_dir = 'data/raw'
        os.makedirs(self.raw_data_dir, exist_ok=True)
    
    def fetch_acled_data(self, days_back=365):
        """
        Fetch conflict event data from ACLED.
        
        Args:
            days_back: Number of days of historical data to retrieve
            
        Returns:
            DataFrame with conflict events
        """
        logger.info("Fetching ACLED data...")
        
        acled_config = self.config['data_sources']['acled']
        url = acled_config['api_url']
        
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'country': 'Thailand|Cambodia',
            'event_date__gte': start_date,
            'event_date__lte': end_date,
            'event_type': '|'.join(acled_config['event_types']),
            'limit': 10000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data['data'])
            
            # Keep relevant columns
            keep_cols = ['event_id_cnty', 'iso', 'event_date', 'year', 'event_type', 
                        'sub_event_type', 'actor1', 'actor2', 'country', 'admin1', 
                        'latitude', 'longitude', 'fatalities', 'source_scale']
            
            df = df[[col for col in keep_cols if col in df.columns]]
            
            # Save to CSV
            output_path = os.path.join(self.raw_data_dir, 'acled_events.csv')
            df.to_csv(output_path, index=False)
            logger.info(f"Downloaded {len(df)} ACLED records to {output_path}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching ACLED data: {e}")
            return None
    
    def fetch_ucdp_data(self):
        """
        Fetch organized violence data from UCDP.
        
        Returns:
            DataFrame with conflict data
        """
        logger.info("Fetching UCDP data...")
        
        ucdp_config = self.config['data_sources']['ucdp']
        base_url = ucdp_config['base_url']
        
        try:
            # UCDP provides organized violence by country
            url = f"{base_url}/organized_violence"
            params = {'country': ['Thailand', 'Cambodia']}
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            output_path = os.path.join(self.raw_data_dir, 'ucdp_violence.csv')
            df.to_csv(output_path, index=False)
            logger.info(f"Downloaded UCDP data to {output_path}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching UCDP data: {e}")
            return None
    
    def fetch_openstreetmap_boundaries(self):
        """
        Download OpenStreetMap administrative boundaries.
        Uses geofabrik extracts for Thailand and Cambodia.
        
        Returns:
            List of downloaded file paths
        """
        logger.info("Fetching OpenStreetMap boundary data...")
        
        urls = {
            'thailand': 'https://download.geofabrik.de/asia/thailand-latest.osm.pbf',
            'cambodia': 'https://download.geofabrik.de/asia/cambodia-latest.osm.pbf'
        }
        
        downloaded_files = []
        
        for country, url in urls.items():
            try:
                logger.info(f"Downloading OSM data for {country}...")
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                file_path = os.path.join(self.raw_data_dir, f'{country}_osm.pbf')
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloaded_files.append(file_path)
                logger.info(f"Downloaded {country} OSM data to {file_path}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error downloading {country} OSM data: {e}")
        
        return downloaded_files
    
    def fetch_population_data(self):
        """
        Fetch population density data from WorldPop.
        For this example, we'll create mock data with references to actual sources.
        
        Returns:
            DataFrame with population estimates
        """
        logger.info("Fetching population density data...")
        
        # Note: Actual WorldPop data requires registration
        # https://www.worldpop.org/
        
        logger.info("WorldPop data requires manual download and registration")
        logger.info("Visit: https://www.worldpop.org/ for Thailand/Cambodia datasets")
        
        return None
    
    def consolidate_raw_data(self):
        """
        Consolidate all downloaded raw data into a single index file.
        
        Returns:
            DataFrame with metadata about all sources
        """
        logger.info("Consolidating raw data sources...")
        
        metadata = {
            'source': [],
            'file_path': [],
            'record_count': [],
            'date_range': [],
            'download_date': []
        }
        
        # Check ACLED data
        acled_path = os.path.join(self.raw_data_dir, 'acled_events.csv')
        if os.path.exists(acled_path):
            df_acled = pd.read_csv(acled_path)
            metadata['source'].append('ACLED')
            metadata['file_path'].append(acled_path)
            metadata['record_count'].append(len(df_acled))
            metadata['date_range'].append(f"{df_acled['event_date'].min()} to {df_acled['event_date'].max()}")
            metadata['download_date'].append(datetime.now().isoformat())
        
        metadata_df = pd.DataFrame(metadata)
        metadata_path = os.path.join(self.raw_data_dir, 'data_manifest.csv')
        metadata_df.to_csv(metadata_path, index=False)
        
        logger.info(f"Data manifest saved to {metadata_path}")
        
        return metadata_df

def main():
    """Main entry point for data ingestion."""
    ingestion = DataIngestion()
    
    # Fetch data from all sources
    print("\n=== Starting Data Ingestion ===\n")
    
    acled_df = ingestion.fetch_acled_data(days_back=365)
    ucdp_df = ingestion.fetch_ucdp_data()
    osm_files = ingestion.fetch_openstreetmap_boundaries()
    pop_df = ingestion.fetch_population_data()
    
    # Consolidate metadata
    metadata = ingestion.consolidate_raw_data()
    
    print("\n=== Data Ingestion Complete ===\n")
    print(metadata)

if __name__ == '__main__':
    main()