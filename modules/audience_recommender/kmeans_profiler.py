"""
AI-Powered Audience Profiling using K-Means Clustering
Data-driven customer segmentation and persona generation
"""
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


class KMeansAudienceProfiler:
    """K-Means clustering for audience segmentation"""
    
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_profiles = {}
        self.is_trained = False
    
    def segment_audience(self, customer_data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Segment audience using K-Means clustering
        
        Args:
            customer_data: DataFrame with customer features
            
        Returns:
            Dictionary mapping cluster ID to segment profile
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            logger.info(f"Segmenting audience into {self.n_clusters} clusters...")
            
            # Select numeric features for clustering
            feature_cols = [col for col in customer_data.columns if customer_data[col].dtype in ['int64', 'float64']]
            
            if len( feature_cols) < 2:
                logger.warning("Not enough numeric features for clustering")
                return self._create_default_segments()
            
            X = customer_data[feature_cols]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = self.kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to data
            customer_data['cluster'] = cluster_labels
            
            # Generate cluster profiles
            segments = {}
            for cluster_id in range(self.n_clusters):
                cluster_data = customer_data[customer_data['cluster'] == cluster_id]
                
                segment_profile = {
                    'cluster_id': cluster_id,
                    'size': len(cluster_data),
                    'percentage': (len(cluster_data) / len(customer_data)) * 100,
                    'characteristics': self._describe_cluster(cluster_data, feature_cols),
                    'persona': self._generate_persona(cluster_data, cluster_id)
                }
                
                segments[cluster_id] = segment_profile
            
            self.cluster_profiles = segments
            self.is_trained = True
            
            logger.info(f"Successfully created {len(segments)} audience segments")
            return segments
            
        except Exception as e:
            logger.error(f"Error in K-Means clustering: {e}")
            return self._create_default_segments()
    
    def _describe_cluster(self, cluster_data: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Describe cluster characteristics"""
        characteristics = {}
        
        for col in feature_cols:
            if cluster_data[col].dtype in ['int64', 'float64']:
                characteristics[col] = {
                    'mean': float(cluster_data[col].mean()),
                    'median': float(cluster_data[col].median()),
                    'std': float(cluster_data[col].std())
                }
        
        return characteristics
    
    def _generate_persona(self, cluster_data: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
        """Generate buyer persona from cluster"""
        persona_names = [
            "Budget-Conscious Buyer",
            "Premium Quality Seeker",
            "Convenience Shopper",
            "Deal Hunter",
            "Brand Loyalist"
        ]
        
        # Calculate persona attributes
        if 'age' in cluster_data.columns:
            avg_age = cluster_data['age'].mean()
            age_range = f"{int(avg_age - 5)}-{int(avg_age + 5)}"
        else:
            age_range = "25-45"
        
        if 'income' in cluster_data.columns:
            avg_income = cluster_data['income'].mean()
            if avg_income < 30000:
                income_level = "Low"
            elif avg_income < 75000:
                income_level = "Medium"
            else:
                income_level = "High"
        else:
            income_level = "Medium"
        
        persona = {
            'name': persona_names[cluster_id % len(persona_names)],
            'age_range': age_range,
            'income_level': income_level,
            'segment_size': len(cluster_data),
            'key_traits': self._identify_key_traits(cluster_data)
        }
        
        return persona
    
    def _identify_key_traits(self, cluster_data: pd.DataFrame) -> List[str]:
        """Identify key traits of the segment"""
        traits = []
        
        if 'purchase_frequency' in cluster_data.columns:
            avg_freq = cluster_data['purchase_frequency'].mean()
            if avg_freq > 10:
                traits.append("Frequent buyer")
            elif avg_freq < 3:
                traits.append("Occasional buyer")
        
        if 'avg_order_value' in cluster_data.columns:
            avg_value = cluster_data['avg_order_value'].mean()
            if avg_value > 100:
                traits.append("High-value customer")
            elif avg_value < 30:
                traits.append("Budget-conscious")
        
        if not traits:
            traits.append("General consumer")
        
        return traits
    
    def _create_default_segments(self) -> Dict[int, Dict[str, Any]]:
        """Create default segments when clustering fails"""
        return {
            0: {
                'cluster_id': 0,
                'size': 100,
                'percentage': 40.0,
                'persona': {
                    'name': 'Budget-Conscious Buyer',
                    'age_range': '25-35',
                    'income_level': 'Low-Medium',
                    'segment_size': 40,
                    'key_traits': ['Price-sensitive', 'Value-seeking']
                }
            },
            1: {
                'cluster_id': 1,
                'size': 100,
                'percentage': 35.0,
                'persona': {
                    'name': 'Premium Quality Seeker',
                    'age_range': '35-50',
                    'income_level': 'High',
                    'segment_size': 35,
                    'key_traits': ['Quality-focused', 'Brand-conscious']
                }
            },
            2: {
                'cluster_id': 2,
                'size': 75,
                'percentage': 25.0,
                'persona': {
                    'name': 'Convenience Shopper',
                    'age_range': '25-40',
                    'income_level': 'Medium',
                    'segment_size': 25,
                    'key_traits': ['Time-conscious', 'Tech-savvy']
                }
            }
        }
    
    def generate_synthetic_customer_data(self, n_samples: int = 300) -> pd.DataFrame:
        """Generate synthetic customer data for demo"""
        np.random.seed(42)
        
        data = {
            'age': np.random.normal(35, 10, n_samples).astype(int),
            'income': np.random.lognormal(10.5, 0.5, n_samples),
            'purchase_frequency': np.random.poisson(5, n_samples),
            'avg_order_value': np.random.gamma(2, 30, n_samples),
            'time_on_site_minutes': np.random.exponential(5, n_samples),
            'cart_abandonment_rate': np.random.beta(2, 5, n_samples)
        }
        
        df = pd.DataFrame(data)
        df['age'] = df['age'].clip(18, 75)
        df['income'] = df['income'].clip(15000, 200000)
        
        return df
