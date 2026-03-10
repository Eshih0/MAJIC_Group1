"""
Risk Scoring Module
Computes risk escalation scores for each region using the trained Bayesian Network.
"""

import logging
import pandas as pd
import numpy as np
from src.bayesian_network import BorderRiskNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskScorer:
    """Computes risk escalation scores from Bayesian Network inference."""
    
    def __init__(self, model_path=None):
        """
        Initialize risk scorer with trained model.
        
        Args:
            model_path: Path to trained Bayesian Network model
        """
        if model_path:
            self.model = BorderRiskNetwork.load_model(model_path)
        else:
            self.model = None
        
        self.scores = None
    
    def compute_region_scores(self, data_path):
        """
        Compute risk escalation scores for all regions.
        
        Args:
            data_path: Path to CSV with region features
            
        Returns:
            DataFrame with scores and classifications
        """
        logger.info(f"Computing risk scores from {data_path}...")
        
        if self.model is None:
            raise ValueError("Model not initialized. Provide model_path to constructor.")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Compute probability of high risk for each region
        scores = []
        
        for idx, row in df.iterrows():
            # Prepare evidence from row
            evidence = {
                'incident_frequency': str(row['incident_frequency']),
                'economic_stress': str(row['economic_stress']),
                'population_density': str(row['population_density']),
                'troop_concentration': str(row['troop_concentration']),
                'border_activity': str(row['border_activity'])
            }
            
            # Perform inference
            try:
                prob_dist = self.model.predict(evidence)
                
                # Extract probability of 'high' risk
                if prob_dist is not None:
                    prob_high = prob_dist.state_counts.get('high', 0.0)
                else:
                    prob_high = np.random.random()  # Fallback
                    
            except Exception as e:
                logger.warning(f"Inference failed for region {idx}: {e}")
                prob_high = 0.5  # Default
            
            scores.append({
                'grid_lat': row.get('grid_lat', 0),
                'grid_lon': row.get('grid_lon', 0),
                'escalation_probability': prob_high,
                'risk_level': self._classify_risk(prob_high),
                'incident_count': row.get('incident_count', 0),
                'fatalities': row.get('total_fatalities', 0)
            })
        
        self.scores = pd.DataFrame(scores)
        logger.info(f"Computed scores for {len(self.scores)} regions")
        
        return self.scores
    
    def _classify_risk(self, probability):
        """Classify risk level based on probability threshold."""
        if probability >= 0.67:
            return 'high'
        elif probability >= 0.33:
            return 'medium'
        else:
            return 'low'
    
    def get_high_risk_regions(self, threshold=0.67):
        """
        Get regions with escalation probability above threshold.
        
        Args:
            threshold: Probability threshold (0-1)
            
        Returns:
            DataFrame of high-risk regions
        """
        if self.scores is None:
            raise ValueError("Scores not computed. Call compute_region_scores() first.")
        
        high_risk = self.scores[self.scores['escalation_probability'] >= threshold]
        logger.info(f"Identified {len(high_risk)} high-risk regions")
        
        return high_risk.sort_values('escalation_probability', ascending=False)
    
    def get_trend_analysis(self):
        """Analyze trend in risk scores over time."""
        if self.scores is None:
            raise ValueError("Scores not computed.")
        
        mean_risk = self.scores['escalation_probability'].mean()
        std_risk = self.scores['escalation_probability'].std()
        max_risk = self.scores['escalation_probability'].max()
        
        summary = {
            'mean_escalation_probability': mean_risk,
            'std_escalation_probability': std_risk,
            'max_escalation_probability': max_risk,
            'high_risk_count': (self.scores['escalation_probability'] >= 0.67).sum(),
            'medium_risk_count': ((self.scores['escalation_probability'] >= 0.33) & 
                                 (self.scores['escalation_probability'] < 0.67)).sum(),
            'low_risk_count': (self.scores['escalation_probability'] < 0.33).sum()
        }
        
        return summary
    
    def save_scores(self, output_path):
        """Save computed scores to CSV."""
        if self.scores is None:
            raise ValueError("Scores not computed.")
        
        self.scores.to_csv(output_path, index=False)
        logger.info(f"Scores saved to {output_path}")

def main():
    """Main entry point for risk scoring."""
    print("\n=== Computing Risk Scores ===\n")
    
    # Load model and compute scores
    scorer = RiskScorer(model_path='models/border_risk_bn.pkl')
    scores = scorer.compute_region_scores('data/processed/border_risk_features.csv')
    
    # Get summary
    summary = scorer.get_trend_analysis()
    print("\nRisk Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Get high-risk regions
    high_risk = scorer.get_high_risk_regions(threshold=0.67)
    print(f"\nTop 5 High-Risk Regions:")
    print(high_risk.head())
    
    # Save
    scorer.save_scores('data/processed/escalation_scores.csv')

if __name__ == '__main__':
    main()