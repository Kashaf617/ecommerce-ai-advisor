"""
AI-Powered Product Matching using SBERT
Semantic similarity and deduplication for marketplace data
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger
from modules.ai_models import AIModelManager

logger = get_logger(__name__)


class SemanticProductMatcher:
    """Uses SBERT for semantic product matching and deduplication"""
    
    def __init__(self):
        self.model_manager = AIModelManager()
        self.sbert_model = None
        self._load_model()
    
    def _load_model(self):
        """Load SBERT model"""
        try:
            logger.info("Loading SBERT model for semantic matching...")
            self.sbert_model = self.model_manager.get_sbert_model()
            if self.sbert_model:
                logger.info("SBERT model loaded successfully")
            else:
                logger.warning("Failed to load SBERT model, using fallback")
        except Exception as e:
            logger.error(f"Error loading SBERT: {e}")
            self.sbert_model = None
    
    def find_similar_products(self, query: str, products: List[Dict[str, Any]], 
                             top_k: int = 10, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find products similar to query using semantic search
        
        Args:
            query: Search query or product description
            products: List of product dictionaries
            top_k: Number of top matches to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of products sorted by similarity
        """
        if not self.sbert_model or not products:
            return products[:top_k]  # Fallback to first k products
        
        try:
            # Create product descriptions for embedding
            product_texts = [
                f"{p.get('title', '')} {p.get('description', '')} {p.get('category', '')}"
                for p in products
            ]
            
            # Generate embeddings
            query_embedding = self.sbert_model.encode([query])[0]
            product_embeddings = self.sbert_model.encode(product_texts)
            
            # Calculate cosine similarity
            similarities = self._cosine_similarity(query_embedding, product_embeddings)
            
            # Add similarity scores to products
            for i, product in enumerate(products):
                product['similarity_score'] = float(similarities[i])
            
            # Filter by threshold and sort
            relevant_products = [
                p for p in products 
                if p['similarity_score'] >= threshold
            ]
            relevant_products.sort(
                key=lambda x: x['similarity_score'], 
                reverse=True
            )
            
            logger.info(f"Found {len(relevant_products)} products above similarity threshold {threshold}")
            return relevant_products[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
            return products[:top_k]  # Fallback
    
    def deduplicate_products(self, products: List[Dict[str, Any]], 
                            similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """
        Remove duplicate products using semantic similarity
        
        Args:
            products: List of product dictionaries
            similarity_threshold: Products above this similarity are considered duplicates
            
        Returns:
            Deduplicated list of products
        """
        if not self.sbert_model or not products:
            return products
        
        try:
            # Create product descriptions
            product_texts = [
                f"{p.get('title', '')} {p.get('description', '')}"
                for p in products
            ]
            
            # Generate embeddings
            embeddings = self.sbert_model.encode(product_texts)
            
            # Find duplicates
            unique_indices = []
            seen_embeddings = []
            
            for i, embedding in enumerate(embeddings):
                is_duplicate = False
                
                for seen_embedding in seen_embeddings:
                    similarity = self._cosine_similarity(embedding, seen_embedding)
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_indices.append(i)
                    seen_embeddings.append(embedding)
            
            unique_products = [products[i] for i in unique_indices]
            
            logger.info(f"Deduplicated {len(products)} products to {len(unique_products)} unique items")
            return unique_products
            
        except Exception as e:
            logger.error(f"Error in deduplication: {e}")
            return products  #Fallback
    
    def cluster_products_by_similarity(self, products: List[Dict[str, Any]], 
                                      n_clusters: int = 5) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group products into clusters based on semantic similarity
        
        Args:
            products: List of product dictionaries
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary mapping cluster ID to list of products
        """
        if not self.sbert_model or not products or len(products) < n_clusters:
            return {0: products}  # Return all in one cluster
        
        try:
            from sklearn.cluster import KMeans
            
            # Create embeddings
            product_texts = [
                f"{p.get('title', '')} {p.get('description', '')}"
                for p in products
            ]
            embeddings = self.sbert_model.encode(product_texts)
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Group products by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                label = int(label)
                if label not in clusters:
                    clusters[label] = []
                products[i]['cluster_id'] = label
                clusters[label].append(products[i])
            
            logger.info(f"Clustered {len(products)} products into {len(clusters)} clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return {0: products}
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between vectors"""
        if isinstance(vec2, list):
            vec2 = np.array(vec2)
        
        if len(vec2.shape) == 1:
            # Single vector
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        else:
            # Multiple vectors
            similarities = []
            for v in vec2:
                sim = np.dot(vec1, v) / (np.linalg.norm(vec1) * np.linalg.norm(v))
                similarities.append(sim)
            return np.array(similarities)
