"""
AI-Powered Duplicate Product Remover
Uses SBERT for semantic similarity matching
"""
from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


class DuplicateRemover:
    """Remove duplicate products using AI-powered similarity matching"""
    
    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai
        self.sbert_model = None
        
        if use_ai:
            try:
                from modules.ai_models.model_manager import ModelManager
                model_manager = ModelManager()
                self.sbert_model = model_manager.get_sbert_model()
                logger.info("DuplicateRemover initialized with SBERT")
            except Exception as e:
                logger.warning(f"Could not load SBERT: {e}. Using fuzzy matching fallback.")
                self.use_ai = False
        else:
            logger.info("DuplicateRemover initialized (fuzzy matching mode)")
    
    def find_duplicates(self, products: List[Dict[str, Any]], 
                       similarity_threshold: float = 0.85) -> List[List[int]]:
        """
        Find duplicate products
        
        Args:
            products: List of product dictionaries
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of duplicate groups (each group is list of indices)
        """
        if not products:
            return []
        
        logger.info(f"Finding duplicates in {len(products)} products (AI: {self.use_ai})")
        
        if self.use_ai and self.sbert_model:
            return self._find_duplicates_ai(products, similarity_threshold)
        else:
            return self._find_duplicates_fuzzy(products, similarity_threshold)
    
    def _find_duplicates_ai(self, products: List[Dict[str, Any]], 
                           threshold: float) -> List[List[int]]:
        """Find duplicates using SBERT semantic similarity"""
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create text representations
            texts = []
            for p in products:
                title = p.get('title', '')
                category = p.get('category', '')
                brand = p.get('brand', '')
                text = f"{title} {category} {brand}".strip()
                texts.append(text if text else "unknown product")
            
            # Get embeddings
            embeddings = self.sbert_model.encode(texts)
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find duplicate groups
            duplicate_groups = []
            processed = set()
            
            for i in range(len(products)):
                if i in processed:
                    continue
                
                # Find similar products
                similar_indices = []
                for j in range(i, len(products)):
                    if j in processed:
                        continue
                    
                    if similarity_matrix[i][j] >= threshold:
                        similar_indices.append(j)
                        processed.add(j)
                
                # Only keep groups with more than 1 product
                if len(similar_indices) > 1:
                    duplicate_groups.append(similar_indices)
            
            logger.info(f"Found {len(duplicate_groups)} duplicate groups using SBERT")
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Error in AI duplicate detection: {e}")
            return self._find_duplicates_fuzzy(products, threshold)
    
    def _find_duplicates_fuzzy(self, products: List[Dict[str, Any]], 
                               threshold: float) -> List[List[int]]:
        """Find duplicates using fuzzy string matching (fallback)"""
        try:
            from difflib import SequenceMatcher
            
            duplicate_groups = []
            processed = set()
            
            for i in range(len(products)):
                if i in processed:
                    continue
                
                title_i = str(products[i].get('title', '')).lower()
                price_i = products[i].get('price', 0)
                
                similar_indices = [i]
                processed.add(i)
                
                for j in range(i + 1, len(products)):
                    if j in processed:
                        continue
                    
                    title_j = str(products[j].get('title', '')).lower()
                    price_j = products[j].get('price', 0)
                    
                    # Calculate title similarity
                    title_sim = SequenceMatcher(None, title_i, title_j).ratio()
                    
                    # Calculate price similarity
                    if price_i > 0 and price_j > 0:
                        price_diff = abs(price_i - price_j) / max(price_i, price_j)
                        price_sim = 1 - price_diff
                    else:
                        price_sim = 0.5
                    
                    # Combined similarity
                    combined_sim = title_sim * 0.7 + price_sim * 0.3
                    
                    if combined_sim >= threshold:
                        similar_indices.append(j)
                        processed.add(j)
                
                if len(similar_indices) > 1:
                    duplicate_groups.append(similar_indices)
            
            logger.info(f"Found {len(duplicate_groups)} duplicate groups using fuzzy matching")
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Error in fuzzy duplicate detection: {e}")
            return []
    
    def merge_duplicates(self, products: List[Dict[str, Any]], 
                        duplicate_groups: List[List[int]]) -> List[Dict[str, Any]]:
        """
        Merge duplicate products, keeping best quality data
        
        Returns:
            Deduplicated list of products
        """
        if not duplicate_groups:
            return products
        
        # Create set of all duplicate indices
        all_duplicate_indices = set()
        for group in duplicate_groups:
            all_duplicate_indices.update(group)
        
        # Merge each group
        merged_products = []
        
        for group in duplicate_groups:
            # Select best product from group (highest rating, most reviews)
            best_idx = group[0]
            best_product = products[best_idx]
            
            for idx in group[1:]:
                p = products[idx]
                # Prefer product with more data
                if p.get('rating', 0) > best_product.get('rating', 0):
                    best_product = p
                    best_idx = idx
                elif p.get('reviews_count', 0) > best_product.get('reviews_count', 0):
                    best_product = p
                    best_idx = idx
            
            merged_products.append(best_product)
        
        # Add non-duplicate products
        for i, product in enumerate(products):
            if i not in all_duplicate_indices:
                merged_products.append(product)
        
        logger.info(f"Merged {len(products)} -> {len(merged_products)} products ({len(products) - len(merged_products)} removed)")
        return merged_products
