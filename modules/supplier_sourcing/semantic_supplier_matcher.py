"""
AI-Powered Supplier Matching using SBERT
Semantic similarity for intelligent supplier recommendations
"""
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger
from modules.ai_models import AIModelManager

logger = get_logger(__name__)


class SemanticSupplierMatcher:
    """Uses SBERT for semantic supplier matching"""
    
    def __init__(self):
        self.model_manager = AIModelManager()
        self.sbert_model = None
        self.supplier_embeddings = None
        self.suppliers = []
        self._load_model()
    
    def _load_model(self):
        """Load SBERT model"""
        try:
            logger.info("Loading SBERT model for supplier matching...")
            self.sbert_model = self.model_manager.get_sbert_model()
            if self.sbert_model:
                logger.info("SBERT model loaded successfully")
            else:
                logger.warning("Failed to load SBERT model, using fallback")
        except Exception as e:
            logger.error(f"Error loading SBERT: {e}")
            self.sbert_model = None
    
    def index_suppliers(self, suppliers: List[Dict[str, Any]]):
        """
        Create embeddings for all suppliers
        
        Args:
            suppliers: List of supplier dictionaries
        """
        if not self.sbert_model or not suppliers:
            self.suppliers = suppliers
            return
        
        try:
            logger.info(f"Creating embeddings for {len(suppliers)} suppliers...")
            
            # Create descriptions for embedding
            supplier_texts = [
                f"{s['name']} {s['type']} {s.get('specialties', [])} {s['region']}"
                for s in suppliers
            ]
            
            # Generate embeddings
            self.supplier_embeddings = self.sbert_model.encode(supplier_texts)
            self.suppliers = suppliers
            
            logger.info(f"Indexed {len(suppliers)} suppliers")
            
        except Exception as e:
            logger.error(f"Error indexing suppliers: {e}")
            self.suppliers = suppliers
    
    def find_best_suppliers(self, requirements: str, top_k: int = 5,
                           budget: float = None, quantity: int = None) -> List[Dict[str, Any]]:
        """
        Find best suppliers using semantic search
        
        Args:
            requirements: Product requirements description
            top_k: Number of suppliers to return
            budget: Budget constraint
            quantity: Quantity needed
            
        Returns:
            List of suppliers ranked by relevance
        """
        if not self.sbert_model or not self.suppliers:
            # Fallback to traditional matching
            return self._fallback_matching(requirements, top_k, budget, quantity)
        
        try:
            # Generate query embedding
            query_embedding = self.sbert_model.encode([requirements])[0]
            
            # Calculate similarities
            similarities = self._cosine_similarity(query_embedding, self.supplier_embeddings)
            
            # Add similarity scores to suppliers
            suppliers_with_scores = []
            for i, supplier in enumerate(self.suppliers):
                supplier_copy = supplier.copy()
                supplier_copy['semantic_match_score'] = float(similarities[i])
                
                # Filter by budget and quantity if provided
                if budget and quantity:
                    estimated_cost = supplier_copy.get('estimated_unit_cost', 0) * quantity
                    if estimated_cost > budget:
                        supplier_copy['fits_budget'] = False
                        supplier_copy['semantic_match_score'] *= 0.5  # Penalize
                    else:
                        supplier_copy['fits_budget'] = True
                
                # Check MOQ
                if quantity:
                    moq_range = supplier_copy.get('moq_range', [0, 999999])
                    if quantity < moq_range[0] or quantity > moq_range[1]:
                        supplier_copy['semantic_match_score'] *= 0.7  # Penalize
                
                suppliers_with_scores.append(supplier_copy)
            
            # Sort by semantic match score
            suppliers_with_scores.sort(
                key=lambda x: x['semantic_match_score'],
                reverse=True
            )
            
            logger.info(f"Found {len(suppliers_with_scores)} matching suppliers")
            return suppliers_with_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic supplier matching: {e}")
            return self._fallback_matching(requirements, top_k, budget, quantity)
    
    def _fallback_matching(self, requirements: str, top_k: int,
                          budget: float = None, quantity: int = None) -> List[Dict[str, Any]]:
        """Fallback to keyword matching"""
        if not self.suppliers:
            return []
        
        # Simple keyword matching
        keywords = requirements.lower().split()
        scored_suppliers = []
        
        for supplier in self.suppliers:
            score = 0
            supplier_text = f"{supplier['name']} {supplier['type']} {supplier.get('specialties', [])}".lower()
            
            for keyword in keywords:
                if keyword in supplier_text:
                    score += 1
            
            supplier_copy = supplier.copy()
            supplier_copy['semantic_match_score'] = score / max(len(keywords), 1)
            
            # Budget check
            if budget and quantity:
                estimated_cost = supplier_copy.get('estimated_unit_cost', 0) * quantity
                supplier_copy['fits_budget'] = estimated_cost <= budget
            
            scored_suppliers.append(supplier_copy)
        
        scored_suppliers.sort(key=lambda x: x['semantic_match_score'], reverse=True)
        return scored_suppliers[:top_k]
    
    def cluster_suppliers_by_type(self, n_clusters: int = 5) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster suppliers by similarity
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Dictionary mapping cluster ID to suppliers
        """
        if not self.sbert_model or not self.supplier_embeddings or len(self.suppliers) < n_clusters:
            return {0: self.suppliers}
        
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.supplier_embeddings)
            
            clusters = {}
            for i, label in enumerate(cluster_labels):
                label = int(label)
                if label not in clusters:
                    clusters[label] = []
                
                supplier = self.suppliers[i].copy()
                supplier['cluster_id'] = label
                clusters[label].append(supplier)
            
            logger.info(f"Clustered {len(self.suppliers)} suppliers into {len(clusters)} groups")
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering suppliers: {e}")
            return {0: self.suppliers}
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity"""
        if len(vec2.shape) == 1:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        else:
            similarities = []
            for v in vec2:
                sim = np.dot(vec1, v) / (np.linalg.norm(vec1) * np.linalg.norm(v))
                similarities.append(sim)
            return np.array(similarities)
