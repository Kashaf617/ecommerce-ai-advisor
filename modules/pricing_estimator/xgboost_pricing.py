"""
AI-Powered Pricing using XGBoost
Machine learning-driven price optimization
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger
from config import MODELS_DIR

logger = get_logger(__name__)


class XGBoostPricingModel:
    """XGBoost model for intelligent pricing predictions"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.model_path = MODELS_DIR / 'xgboost_pricing_model.json'
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            import xgboost as xgb
            
            if self.model_path.exists():
                logger.info(f"Loading XGBoost model from {self.model_path}")
                self.model = xgb.XGBRegressor()
                self.model.load_model(str(self.model_path))
                self.is_trained = True
                logger.info("XGBoost model loaded successfully")
            else:
                logger.info("Creating new XGBoost model")
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    objective='reg:squarederror'
                )
                self.is_trained = False
                logger.info("XGBoost model created (untrained)")
        except Exception as e:
            logger.error(f"Error loading/creating XGBoost model: {e}")
            self.model = None
    
    def create_features(self, product_data: Dict[str, Any], 
                       market_data: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Create feature matrix for price prediction
        
        Args:
            product_data: Product information
            market_data: Market statistics
            
        Returns:
            DataFrame with features
        """
        features = {}
        
        # Product features
        features['cost'] = product_data.get('cost', 0)
        features['quantity'] = product_data.get('quantity', 100)
        features['target_margin'] = product_data.get('target_margin', 0.30)
        
        # Category encoding (simplified)
        category = product_data.get('category', 'general').lower()
        category_map = {
            'electronics': 1, 'fashion': 2, 'beauty': 3, 'home': 4,
            'sports': 5, 'books': 6, 'toys': 7, 'automotive': 8
        }
        features['category_code'] = category_map.get(category, 0)
        
        # Market features
        if market_data:
            features['competitor_avg_price'] = market_data.get('average_price', 0)
            features['competitor_count'] = market_data.get('competitor_count', 0)
            features['market_avg_rating'] = market_data.get('average_rating', 4.0)
            features['market_demand_level'] = self._encode_demand(market_data.get('demand_level', 'Medium'))
        else:
            # Default values
            features['competitor_avg_price'] = product_data.get('cost', 0) * 2
            features['competitor_count'] = 10
            features['market_avg_rating'] = 4.0
            features['market_demand_level'] = 2  # Medium
        
        # Platform features
        platform = product_data.get('platform', 'amazon')
        features['platform_amazon'] = 1 if platform == 'amazon' else 0
        features['platform_daraz'] = 1 if platform == 'daraz' else 0
        features['platform_shopify'] = 1 if platform == 'shopify' else 0
        
        # Additional features
        features['shipping_cost'] = product_data.get('shipping_cost', product_data.get('cost', 0) * 0.1)
        features['has_competitors'] = 1 if features['competitor_count'] > 0 else 0
        
        return pd.DataFrame([features])
    
    def _encode_demand(self, demand_level: str) -> int:
        """Encode demand level to integer"""
        demand_map = {'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
        return demand_map.get(demand_level.lower(), 2)
    
    def train_model(self, training_data: List[Dict[str, Any]]):
        """
        Train the XGBoost model
        
        Args:
            training_data: List of training examples with features and optimal_price
        """
        if not self.model:
            logger.error("Model not initialized")
            return False
        
        try:
            logger.info(f"Training XGBoost model with {len(training_data)} examples")
            
            # Create features and labels
            X_list = []
            y_list = []
            
            for example in training_data:
                features = self.create_features(example, example.get('market_data'))
                X_list.append(features)
                y_list.append(example['optimal_price'])
            
            X = pd.concat(X_list, ignore_index=True)
            y = np.array(y_list)
            
            # Train model
            self.model.fit(X, y)
            self.is_trained = True
            
            # Save model
            self.model.save_model(str(self.model_path))
            logger.info(f"Model trained and saved to {self.model_path}")
            
            # Show feature importance
            self._log_feature_importance(X.columns)
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def _log_feature_importance(self, feature_names):
        """Log feature importance"""
        try:
            importance_dict = dict(zip(feature_names, self.model.feature_importances_))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            logger.info("Feature Importance (Top 5):")
            for feature, importance in sorted_importance[:5]:
                logger.info(f"  {feature}: {importance:.4f}")
        except Exception as e:
            logger.warning(f"Could not log feature importance: {e}")
    
    def predict_price(self, product_data: Dict[str, Any], 
                     market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict optimal price using ML
        
        Args:
            product_data: Product information
            market_data: Market statistics
            
        Returns:
            Dictionary with prediction and metadata
        """
        if not self.model:
            return self._fallback_prediction(product_data)
        
        try:
            # Create features
            features = self.create_features(product_data, market_data)
            
            # Predict price
            predicted_price = self.model.predict(features)[0]
            
            # Calculate confidence based on training status
            if self.is_trained:
                confidence = 0.85
                confidence_level = 'High'
            else:
                # Untrained model - use rule-based fallback
                return self._fallback_prediction(product_data)
            
            # Calculate metrics
            cost = product_data.get('cost', 0)
            profit = predicted_price - cost
            margin = (profit / predicted_price * 100) if predicted_price > 0 else 0
            roi = (profit / cost * 100) if cost > 0 else 0
            
            return {
                'predicted_price': float(predicted_price),
                'cost': cost,
                'profit': float(profit),
                'margin_percent': float(margin),
                'roi_percent': float(roi),
                'confidence': confidence,
                'confidence_level': confidence_level,
                'model_type': 'XGBoost ML',
                'is_ml_prediction': True,
                'feature_importance': self._get_top_features(features) if self.is_trained else {}
            }
            
        except Exception as e:
            logger.error(f"Error predicting price: {e}")
            return self._fallback_prediction(product_data)
    
    def _fallback_prediction(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based fallback when ML not available"""
        cost = product_data.get('cost', 0)
        target_margin = product_data.get('target_margin', 0.30)
        
        # Simple markup formula
    def _get_top_features(self, features: pd.DataFrame) -> Dict[str, float]:
        """Get top contributing features"""
        if not self.is_trained:
            return {}
        
        try:
            feature_importance = dict(zip(
                features.columns,
                self.model.feature_importances_
            ))
            # Return top 3
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_features[:3])
        except:
            return {}
    
    def generate_synthetic_training_data(self, n_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Generate synthetic training data for demo purposes
        In production, replace with real historical pricing data
        """
        import random
        
        training_data = []
        categories = ['electronics', 'fashion', 'beauty', 'home', 'sports']
        platforms = ['amazon', 'daraz', 'shopify']
        
        for i in range(n_samples):
            cost = random.uniform(10, 500)
            competitor_avg = cost * random.uniform(1.5, 3.0)
            competitor_count = random.randint(1, 50)
            demand_level = random.choice(['Low', 'Medium', 'High'])
            
            # Calculate "optimal" price based on market conditions
            base_price = cost * 2.0
            competition_factor = 1.0 - (min(competitor_count, 30) / 100)  # More competition = lower price
            demand_factor = {'low': 0.9, 'medium': 1.0, 'high': 1.2}[demand_level.lower()]
            
            optimal_price = base_price * competition_factor * demand_factor
            
            example = {
                'cost': cost,
                'quantity': random.randint(10, 1000),
                'target_margin': 0.30,
                'category': random.choice(categories),
                'platform': random.choice(platforms),
                'shipping_cost': cost * 0.1,
                'market_data': {
                    'average_price': competitor_avg,
                    'competitor_count': competitor_count,
                    'average_rating': random.uniform(3.5, 5.0),
                    'demand_level': demand_level
                },
                'optimal_price': optimal_price
            }
            training_data.append(example)
        
        return training_data
