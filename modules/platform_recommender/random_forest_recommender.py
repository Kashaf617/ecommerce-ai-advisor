"""
AI-Powered Platform Recommendation using Random Forest
ML-driven platform selection based on product characteristics
"""
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger
from config import MODELS_DIR

logger = get_logger(__name__)


class RandomForestPlatformRecommender:
    """Random Forest classifier for platform recommendations"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.model_path = MODELS_DIR / 'random_forest_platform_model.pkl'
        self.feature_names = []
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            if self.model_path.exists():
                import joblib
                logger.info(f"Loading Random Forest model from {self.model_path}")
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                logger.info("Random Forest model loaded")
            else:
                logger.info("Creating new Random Forest model")
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.is_trained = False
                logger.info("Random Forest model created (untrained)")
        except Exception as e:
            logger.error(f"Error loading/creating Random Forest: {e}")
            self.model = None
    
    def create_features(self, product_info: Dict[str, Any]) -> pd.DataFrame:
        """Create feature vector for prediction"""
        features = {}
        
        # Category encoding
        category = product_info.get('category', 'general').lower()
        category_map = {
            'electronics': 1, 'fashion': 2, 'beauty': 3, 'home': 4,
            'sports': 5, 'books': 6, 'toys': 7
        }
        features['category_code'] = category_map.get(category, 0)
        
        # Price tier
        price = product_info.get('price', 50)
        if price < 20:
            features['price_tier'] = 1  # Low
        elif price < 100:
            features['price_tier'] = 2  # Medium
        else:
            features['price_tier'] = 3  # High
        
        # Target market
        market = product_info.get('target_market', 'international').lower()
        features['market_pakistan'] = 1 if market == 'pakistan' else 0
        features['market_international'] = 1 if market == 'international' else 0
        
        # Business model
        features['quantity'] = product_info.get('quantity', 100)
        features['has_high_volume'] = 1 if features['quantity'] > 500 else 0
        
        # Budget
        budget = product_info.get('budget', 1000)
        features['budget_tier'] = 1 if budget < 1000 else 2 if budget < 10000 else 3
        
        self.feature_names = list(features.keys())
        return pd.DataFrame([features])
    
    def train_model(self, training_data: List[Dict[str, Any]]):
        """Train the Random Forest model"""
        if self.model is None:  # Use 'is None' to avoid triggering __len__
            logger.error("Model not initialized")
            return False
        
        try:
            import joblib
            logger.info(f"Training Random Forest with {len(training_data)} examples")
            
            # Create features and labels
            X_list = []
            y_list = []
            
            for example in training_data:
                features = self.create_features(example)
                X_list.append(features)
                y_list.append(example['best_platform'])
            
            X = pd.concat(X_list, ignore_index=True)
            y = np.array(y_list)
            
            # Train
            self.model.fit(X, y)
            self.is_trained = True
            logger.info("Model trained successfully")
            
            # Save model (with error handling)
            try:
                # Ensure directory exists
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.model, str(self.model_path))
                logger.info(f"Model saved to {self.model_path}")
            except Exception as save_error:
                logger.warning(f"Could not save model: {save_error}. Model still usable in memory.")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_platforms(self, product_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict best platforms with probabilities"""
        if self.model is None:  # Use 'is None' to avoid triggering __len__
            return self._fallback_recommendation(product_info)
        
        # Check if model is trained using classes_ attribute (set after fit)
        if not self.is_trained or not hasattr(self.model, 'classes_'):
            logger.warning("Model not trained, using fallback recommendations")
            return self._fallback_recommendation(product_info)
        
        try:
            # Create features
            features = self.create_features(product_info)
            
            # Predict probabilities
            probabilities = self.model.predict_proba(features)[0]
            platforms = self.model.classes_
            
            # Create recommendations
            recommendations = []
            for platform, prob in zip(platforms, probabilities):
                recommendations.append({
                    'platform': platform,
                    'probability': float(prob),
                    'confidence_level': 'High' if prob > 0.6 else 'Medium' if prob > 0.3 else 'Low',
                    'is_ml_prediction': True
                })
            
            # Sort by probability
            recommendations.sort(key=lambda x: x['probability'], reverse=True)
            
            logger.info(f"ML prediction: {recommendations[0]['platform']} ({recommendations[0]['probability']:.2%})")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error predicting platforms: {e}")
            return self._fallback_recommendation(product_info)
    
    def _fallback_recommendation(self, product_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rule-based fallback"""
        market = product_info.get('target_market', 'international').lower()
        
        if market == 'pakistan':
            platforms = [
                {'platform': 'daraz', 'probability': 0.7, 'confidence_level': 'Medium', 'is_ml_prediction': False},
                {'platform': 'amazon', 'probability': 0.2, 'confidence_level': 'Low', 'is_ml_prediction': False},
                {'platform': 'shopify', 'probability': 0.1, 'confidence_level': 'Low', 'is_ml_prediction': False}
            ]
        else:
            platforms = [
                {'platform': 'amazon', 'probability': 0.6, 'confidence_level': 'Medium', 'is_ml_prediction': False},
                {'platform': 'shopify', 'probability': 0.3, 'confidence_level': 'Low', 'is_ml_prediction': False},
                {'platform': 'daraz', 'probability': 0.1, 'confidence_level': 'Low', 'is_ml_prediction': False}
            ]
        
        return platforms
    
    def generate_synthetic_training_data(self, n_samples: int = 200) -> List[Dict[str, Any]]:
        """Generate synthetic training data"""
        import random
        
        training_data = []
        categories = ['electronics', 'fashion', 'beauty', 'home', 'sports']
        
        for _ in range(n_samples):
            market = random.choice(['pakistan', 'international', 'asia'])
            category = random.choice(categories)
            price = random.uniform(10, 500)
            quantity = random.randint(10, 1000)
            
            # Rule-based best platform
            if market == 'pakistan':
                best_platform = 'daraz' if random.random() < 0.7 else 'shopify'
            else:
                if price > 100:
                    best_platform = 'amazon'
                else:
                    best_platform = random.choice(['amazon', 'shopify'])
            
            example = {
                'category': category,
                'price': price,
                'target_market': market,
                'quantity': quantity,
                'budget': quantity * price * 0.7,
                'best_platform': best_platform
            }
            
            training_data.append(example)
        
        return training_data
