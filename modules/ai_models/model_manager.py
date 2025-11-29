"""
AI Model Manager - Centralized model loading and caching
"""
from pathlib import Path
from typing import Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import PROJECT_ROOT
from utils.logger import get_logger

logger = get_logger(__name__)

class AIModelManager:
    """Manages AI/ML model loading and caching"""
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_dir = PROJECT_ROOT / 'models'
            cls._instance.model_dir.mkdir(parents=True, exist_ok=True)
        return cls._instance
    
    def get_sbert_model(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Get or load SBERT model for semantic similarity
        
        Args:
            model_name: Name of the sentence-transformers model
            
        Returns:
            SentenceTransformer model
        """
        cache_key = f'sbert_{model_name}'
        
        if cache_key not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SBERT model: {model_name}")
                
                # Load model (will cache to disk automatically)
                model = SentenceTransformer(model_name)
                self._models[cache_key] = model
                
                logger.info(f"SBERT model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading SBERT model: {e}")
                return None
        
        return self._models.get(cache_key)
    
    def get_spacy_model(self, model_name: str = 'en_core_web_sm'):
        """
        Get or load spaCy NER model
        
        Args:
            model_name: Name of the spaCy model
            
        Returns:
            spaCy model
        """
        cache_key = f'spacy_{model_name}'
        
        if cache_key not in self._models:
            try:
                import spacy
                logger.info(f"Loading spaCy model: {model_name}")
                
                # Try to load model
                try:
                    nlp = spacy.load(model_name)
                except OSError:
                    logger.warning(f"spaCy model not found, downloading...")
                    import subprocess
                    subprocess.run(['python', '-m', 'spacy', 'download', model_name])
                    nlp = spacy.load(model_name)
                
                self._models[cache_key] = nlp
                logger.info(f"spaCy model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading spaCy model: {e}")
                return None
        
        return self._models.get(cache_key)
    
    def get_xgboost_model(self, model_path: Optional[Path] = None):
        """
        Get or load XGBoost model
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            XGBoost model or None if not trained
        """
        cache_key = 'xgboost_pricing'
        
        if cache_key not in self._models:
            if model_path and model_path.exists():
                try:
                    import xgboost as xgb
                    logger.info(f"Loading XGBoost model from {model_path}")
                    model = xgb.XGBRegressor()
                    model.load_model(str(model_path))
                    self._models[cache_key] = model
                    logger.info("XGBoost model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading XGBoost model: {e}")
                    return None
            else:
                logger.warning("XGBoost model not found. Will use rule-based fallback.")
                return None
        
        return self._models.get(cache_key)
    
    def clear_cache(self):
        """Clear all cached models"""
        self._models.clear()
        logger.info("Model cache cleared")
    
    def get_model_info(self):
        """Get information about loaded models"""
        return {
            'loaded_models': list(self._models.keys()),
            'cache_size': len(self._models),
            'model_dir': str(self.model_dir)
        }
