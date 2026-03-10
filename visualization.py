"""
Visualization Module
Generates interactive heat maps and other visualizations of risk escalation.
"""

import logging
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeatMapGenerator:
    """Generates interactive heat maps of risk escalation."""
    
    def __init__(self, center_lat=13.5, center_lon=103.5, zoom_start=8):
        """
        Initialize heat map generator.
        
        Args:
            center_lat: Center latitude of map
            center_lon: Center longitude of map
            zoom_start: Initial zoom level
        """
        self.center = [center_lat, center_lon]
        self.zoom_start = zoom_start
        self.map = None
    
    def generate_heatmap(self, scores_df, output_path='outputs/heatmap.html',
                        color_scheme='RdYlGn_r'):
        """
        Generate interactive heat map from risk scores.
        
        Args:
            scores_df: DataFrame with columns 'grid_lat', 'grid_lon', 'escalation_probability'
            output_path: Output HTML file path
            color_scheme: Matplotlib color scheme
        """
        logger.info(f"Generating heat map ({len(scores_df)} regions)...")
        
        # Create base map
        self.map = folium.Map(
            location=self.center,
            zoom_start=self.zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Prepare heat map data
        heat_data = []
        for idx, row in scores_df.iterrows():
            heat_data.append([
                float(row['grid_lat']),
                float(row['grid_lon']),
                float(row['escalation_probability'])
            ])
        
        # Add heat layer
        HeatMap(
            heat_data,
            min_opacity=0.2,
            radius=20,
            blur=15,
            max_zoom=1
        ).add_to(self.map)
        
        # Add markers for high-risk regions
        for idx, row in scores_df.iterrows():
            if row['escalation_probability'] >= 0.67:
                popup_text = f"""
                <b>Risk Region</b><br>
                Lat: {row['grid_lat']:.2f}<br>
                Lon: {row['grid_lon']:.2f}<br>
                Risk Score: {row['escalation_probability']:.2%}<br>
                Incidents: {row.get('incident_count', 0):.0f}<br>
                Fatalities: {row.get('fatalities', 0):.0f}
                """
                
                color = self._get_color(row['escalation_probability'])
                
                folium.CircleMarker(
                    location=[row['grid_lat'], row['grid_lon']],
                    radius=5,
                    popup=folium.Popup(popup_text, max_width=250),
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(self.map)
        
        # Add legend
        self._add_legend(self.map)
        
        # Save map
        self.map.save(output_path)
        logger.info(f"Heat map saved to {output_path}")
        
        return self.map
    
    def _get_color(self, probability):
        """Get color based on risk probability."""
        if probability >= 0.67:
            return '#d73027'  # Red
        elif probability >= 0.33:
            return '#fee08b'  # Yellow
        else:
            return '#91bfdb'  # Blue
    
    def _add_legend(self, map_obj):
        """Add legend to map."""
        legend_html = '''
        <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
        <b>Risk Escalation Scale</b><br>
        <p><i style="background:#d73027"></i> High Risk (≥67%)</p>
        <p><i style="background:#fee08b"></i> Medium Risk (33-67%)</p>
        <p><i style="background:#91bfdb"></i> Low Risk (<33%)</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def add_border_lines(self, border_geojson_path=None):
        """
        Add Thailand-Cambodia border lines to map.
        
        Args:
            border_geojson_path: Path to GeoJSON file with border
        """
        if border_geojson_path is None:
            logger.info("Using hardcoded border approximation")
            
            # Approximate Thailand-Cambodia border coordinates
            border_coords = [
                [13.4, 104.8],
                [13.5, 104.9],
                [13.8, 104.8],
                [14.2, 102.8],
                [14.5, 102.5],
                [15.2, 102.3]
            ]
            
            folium.PolyLine(
                border_coords,
                color='black',
                weight=2,
                opacity=0.8,
                popup='Thailand-Cambodia Border'
            ).add_to(self.map)
        else:
            try:
                with open(border_geojson_path, 'r') as f:
                    border_geojson = json.load(f)
                
                folium.GeoJson(border_geojson).add_to(self.map)
            except Exception as e:
                logger.error(f"Could not load border GeoJSON: {e}")

class RiskDashboard:
    """Creates a summary dashboard of risk metrics."""
    
    @staticmethod
    def create_summary_report(scores_df, output_path='outputs/risk_summary.txt'):
        """
        Create text summary report of risk assessment.
        
        Args:
            scores_df: DataFrame with risk scores
            output_path: Output file path
        """
        logger.info("Generating risk summary report...")
        
        high_risk = scores_df[scores_df['escalation_probability'] >= 0.67]
        medium_risk = scores_df[(scores_df['escalation_probability'] >= 0.33) & 
                              (scores_df['escalation_probability'] < 0.67)]
        low_risk = scores_df[scores_df['escalation_probability'] < 0.33]
        
        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║         THAILAND-CAMBODIA BORDER RISK ESCALATION REPORT            ║
║                    Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}                    ║
╚══════════════���═════════════════════════════════════════════════════╝

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────────────
Total Regions Analyzed:        {len(scores_df)}
Analysis Grid Cell Size:       5km x 5km

RISK DISTRIBUTION
─────────────────────────────────────────────────────────────────────
High Risk Regions (≥67%):      {len(high_risk)} ({len(high_risk)/len(scores_df)*100:.1f}%)
Medium Risk Regions (33-67%):  {len(medium_risk)} ({len(medium_risk)/len(scores_df)*100:.1f}%)
Low Risk Regions (<33%):       {len(low_risk)} ({len(low_risk)/len(scores_df)*100:.1f}%)

AGGREGATE METRICS
─────────────────────────────────────────────────────────────────────
Mean Escalation Probability:   {scores_df['escalation_probability'].mean():.1%}
Median Escalation Probability: {scores_df['escalation_probability'].median():.1%}
Max Escalation Probability:    {scores_df['escalation_probability'].max():.1%}
Min Escalation Probability:    {scores_df['escalation_probability'].min():.1%}

TOP 5 HIGHEST RISK REGIONS
─────────────────────────────────────────────────────────────────────
"""
        
        top_5 = scores_df.nlargest(5, 'escalation_probability')
        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            report += f"\n{i}. Location ({row['grid_lat']:.2f}, {row['grid_lon']:.2f})\n"
            report += f"   Risk Score: {row['escalation_probability']:.1%}\n"
            report += f"   Incidents: {row.get('incident_count', 0):.0f}\n"
            report += f"   Fatalities: {row.get('fatalities', 0):.0f}\n"
        
        report += f"""

RECOMMENDATIONS
─────────────────────────────────────────────────────────────────────
1. Increase monitoring in high-risk regions (marked in red on heat map)
2. Enhance border patrol presence in top 5 highest-risk locations
3. Establish early warning system for medium-risk regions
4. Coordinate with regional stakeholders for conflict prevention
5. Update risk assessment weekly with fresh data

METHODOLOGY NOTES
─────────────────────────────────────────────────────────────────────
- Risk scores computed using Bayesian Network inference
- Evidence variables: incident frequency, economic stress, population 
  density, troop concentration, border activity
- Network trained on historical conflict data from ACLED and UCDP
- Scores represent probabilistic risk of escalation in each region
- Scores should be validated against expert judgment

═════════════════════════════════════════════════════════════════════
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")
        print(report)

def main():
    """Main entry point for visualization."""
    print("\n=== Generating Visualizations ===\n")
    
    # Load scores
    scores = pd.read_csv('data/processed/escalation_scores.csv')
    
    # Generate heat map
    viz = HeatMapGenerator()
    viz.generate_heatmap(scores, output_path='outputs/risk_heatmap.html')
    viz.add_border_lines()
    
    # Generate summary report
    RiskDashboard.create_summary_report(scores)

if __name__ == '__main__':
    main()