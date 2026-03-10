"""
Bayesian Network Module
Defines, trains, and performs inference on the border risk escalation network.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BorderRiskNetwork:
    """
    Bayesian Network model for border risk escalation.
    
    Network Structure:
        Incident Frequency ----\
                                 \
        Economic Stress -----> Regional Stability ----\
             |                   ^                       \
             v                   |                        \
        Population Density ------/                    Risk Escalation
             |
             v
        Border Activity
        
        Troop Concentration ----\
                                  \
                                   Risk Escalation
    """
    
    def __init__(self):
        """Initialize the Bayesian Network structure."""
        self.model = None
        self.inference = None
        self.data = None
        self.cpds = {}
        
    def build_network(self):
        """Build the Bayesian Network structure."""
        logger.info("Building Bayesian Network structure...")
        
        # Define edges (parent -> child relationships)
        edges = [
            ('incident_frequency', 'risk_escalation'),
            ('economic_stress', 'regional_stability'),
            ('population_density', 'regional_stability'),
            ('troop_concentration', 'risk_escalation'),
            ('border_activity', 'regional_stability'),
            ('regional_stability', 'risk_escalation'),
        ]
        
        self.model = BayesianNetwork(edges)
        logger.info(f"Network created with {len(edges)} edges")
        
        return self.model
    
    def load_data(self, data_path):
        """
        Load training data from CSV.
        
        Args:
            data_path: Path to CSV file with training data
        """
        logger.info(f"Loading training data from {data_path}...")
        
        self.data = pd.read_csv(data_path)
        
        # Select relevant columns for the network
        required_cols = [
            'incident_frequency', 'economic_stress', 'population_density',
            'troop_concentration', 'border_activity', 'regional_stability',
            'risk_escalation'
        ]
        
        self.data = self.data[[col for col in required_cols if col in self.data.columns]]
        
        logger.info(f"Loaded {len(self.data)} training samples")
        
        return self.data
    
    def fit_cpds(self):
        """
        Fit Conditional Probability Distributions (CPDs) from data.
        Uses Maximum Likelihood Estimation.
        """
        logger.info("Fitting Conditional Probability Distributions (CPDs)...")
        
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Estimate CPDs using Maximum Likelihood Estimation
        estimator = MaximumLikelihoodEstimator
        
        for node in self.model.nodes():
            parents = list(self.model.predecessors(node))
            
            if not parents:
                # Root node (no parents)
                cpd = self._compute_cpd_no_parents(node)
            else:
                # Non-root node
                cpd = self._compute_cpd_with_parents(node, parents)
            
            self.model.add_cpds(cpd)
            self.cpds[node] = cpd
        
        # Validate CPDs sum to 1
        if self.model.check_model():
            logger.info("CPDs validated successfully")
        else:
            logger.error("CPD validation failed")
    
    def _compute_cpd_no_parents(self, node):
        """Compute CPD for a node with no parents."""
        try:
            value_counts = self.data[node].value_counts(normalize=True).sort_index()
            states = list(value_counts.index)
            cpd_values = value_counts.values
        except:
            # Fallback: uniform distribution
            states = ['low', 'medium', 'high']
            cpd_values = np.array([1/3, 1/3, 1/3])
        
        cpd = TabularCPD(
            variable=node,
            variable_card=len(states),
            values=cpd_values.reshape(len(states), 1)
        )
        
        return cpd
    
    def _compute_cpd_with_parents(self, node, parents):
        """Compute CPD for a node with parents."""
        try:
            # Count occurrences
            grouped = self.data.groupby(parents + [node]).size()
            
            # Normalize to get conditional probabilities
            grouped = grouped.div(grouped.groupby(level=list(range(len(parents)))).sum(), level=list(range(len(parents))))
            
            # Reshape for TabularCPD
            node_states = list(self.data[node].unique())
            cpd_values = np.array([grouped.get((p, n), 1/len(node_states)) 
                                 for p in grouped.index for n in node_states])
            cpd_values = cpd_values.reshape(len(node_states), -1)
            
        except:
            # Fallback: uniform distribution
            node_states = list(self.data[node].unique())
            parent_combos = len(self.data[parents].drop_duplicates())
            cpd_values = np.ones((len(node_states), parent_combos)) / len(node_states)
        
        cpd = TabularCPD(
            variable=node,
            variable_card=len(node_states),
            values=cpd_values,
            evidence=parents,
            evidence_card=[len(self.data[p].unique()) for p in parents]
        )
        
        return cpd
    
    def train(self):
        """Train the Bayesian Network (fit CPDs)."""
        logger.info("Training Bayesian Network...")
        self.fit_cpds()
        
        # Initialize inference engine
        self.inference = VariableElimination(self.model)
        logger.info("Training complete. Inference engine initialized.")
    
    def predict(self, evidence):
        """
        Perform inference given evidence.
        
        Args:
            evidence: Dictionary of observed variables and their values
                     e.g., {'incident_frequency': 'high', 'economic_stress': 'medium'}
        
        Returns:
            Probabilities of risk_escalation states
        """
        if self.inference is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Query for risk_escalation probability given evidence
            result = self.inference.query(
                variables=['risk_escalation'],
                evidence=evidence,
                show_progress=False
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None
    
    def get_most_probable_explanation(self, observation):
        """
        Find the most probable explanation for observed evidence.
        
        Args:
            observation: Dictionary of observed values
            
        Returns:
            Dictionary with most probable state for each variable
        """
        if self.inference is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            mpe = self.inference.map_query(
                variables=list(self.model.nodes()),
                evidence=observation
            )
            return mpe
        except Exception as e:
            logger.error(f"MPE computation failed: {e}")
            return None
    
    def save_model(self, filepath):
        """Save trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model

def main():
    """Main entry point for Bayesian Network training."""
    print("\n=== Training Bayesian Network ===\n")
    
    # Create network
    bn = BorderRiskNetwork()
    bn.build_network()
    
    # Load data
    bn.load_data('data/processed/border_risk_features.csv')
    
    # Train
    bn.train()
    
    # Example inference
    evidence = {
        'incident_frequency': 'high',
        'economic_stress': 'medium',
        'troop_concentration': 'high'
    }
    
    print(f"\nInference with evidence: {evidence}")
    result = bn.predict(evidence)
    print(result)
    
    # Save model
    bn.save_model('models/border_risk_bn.pkl')

if __name__ == '__main__':
    main()