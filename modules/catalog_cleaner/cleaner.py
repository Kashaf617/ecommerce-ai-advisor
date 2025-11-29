"""
Main Catalog Cleaner Orchestrator
Coordinates all cleaning operations
"""
from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger
from .duplicate_remover import DuplicateRemover
from .title_normalizer import TitleNormalizer
from .attribute_fixer import AttributeFixer
from .price_standardizer import PriceStandardizer

logger = get_logger(__name__)


class CatalogCleaner:
    """
    Main catalog cleaning orchestrator
    Coordinates all data cleaning and normalization operations
    """
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize catalog cleaner
        
        Args:
            use_ai: Whether to use AI-powered cleaning (SBERT for duplicates)
        """
        self.use_ai = use_ai
        
        # Initialize components
        self.duplicate_remover = DuplicateRemover(use_ai=use_ai)
        self.title_normalizer = TitleNormalizer()
        self.attribute_fixer = AttributeFixer()
        self.price_standardizer = PriceStandardizer()
        
        logger.info(f"CatalogCleaner initialized (AI: {use_ai})")
    
    def clean_catalog(self,
                     products: List[Dict[str, Any]],
                     remove_duplicates: bool = True,
                     normalize_titles: bool = True,
                     fix_attributes: bool = True,
                     standardize_prices: bool = True,
                     similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """
        Clean and normalize product catalog
        
        Args:
            products: List of product dictionaries
            remove_duplicates: Whether to remove duplicates
            normalize_titles: Whether to clean titles
            fix_attributes: Whether to fix attributes
            standardize_prices: Whether to standardize prices
            similarity_threshold: Duplicate similarity threshold (0-1)
            
        Returns:
            Dictionary with cleaned products and statistics
        """
        logger.info(f"Starting catalog cleaning for {len(products)} products")
        
        if not products:
            return {
                'products': [],
                'products_before': 0,
                'products_after': 0,
                'duplicates_removed': 0,
                'titles_normalized': 0,
                'attributes_fixed': 0,
                'prices_standardized': 0,
                'ai_technique': '🧹 SBERT + Fuzzy Matching',
                'is_ai_powered': self.use_ai
            }
        
        cleaned_products = products.copy()
        stats = {
            'products_before': len(products),
            'duplicates_removed': 0,
            'titles_normalized': 0,
            'attributes_fixed': 0,
            'prices_standardized': 0
        }
        
        # Step 1: Remove duplicates
        if remove_duplicates:
            logger.info("Step 1/4: Removing duplicates...")
            duplicate_groups = self.duplicate_remover.find_duplicates(
                cleaned_products, 
                similarity_threshold
            )
            
            if duplicate_groups:
                before_count = len(cleaned_products)
                cleaned_products = self.duplicate_remover.merge_duplicates(
                    cleaned_products,
                    duplicate_groups
                )
                stats['duplicates_removed'] = before_count - len(cleaned_products)
                logger.info(f"Removed {stats['duplicates_removed']} duplicates")
        
        # Step 2: Normalize titles
        if normalize_titles:
            logger.info("Step 2/4: Normalizing titles...")
            for product in cleaned_products:
                if 'title' in product:
                    original_title = product['title']
                    product['title'] = self.title_normalizer.normalize_title(original_title)
                    
                    if product['title'] != original_title:
                        stats['titles_normalized'] += 1
                    
                    # Extract brand if not present
                    if not product.get('brand'):
                        brand = self.title_normalizer.extract_brand(product['title'])
                        if brand:
                            product['brand'] = brand
            
            logger.info(f"Normalized {stats['titles_normalized']} titles")
        
        # Step 3: Fix attributes
        if fix_attributes:
            logger.info("Step 3/4: Fixing attributes...")
            for product in cleaned_products:
                before = product.copy()
                product.update(self.attribute_fixer.fix_attributes(product))
                
                # Count if any attribute changed
                if product != before:
                    stats['attributes_fixed'] += 1
            
            logger.info(f"Fixed attributes for {stats['attributes_fixed']} products")
        
        # Step 4: Standardize prices
        if standardize_prices:
            logger.info("Step 4/4: Standardizing prices...")
            for product in cleaned_products:
                if 'price' in product:
                    before_price = product.get('price')
                    product.update(self.price_standardizer.standardize_product_price(product))
                    
                    if product.get('price') != before_price:
                        stats['prices_standardized'] += 1
            
            logger.info(f"Standardized {stats['prices_standardized']} prices")
        
        # Final statistics
        stats['products_after'] = len(cleaned_products)
        
        logger.info(f"Cleaning complete: {stats['products_before']} -> {stats['products_after']} products")
        logger.info(f"  - Duplicates removed: {stats['duplicates_removed']}")
        logger.info(f"  - Titles normalized: {stats['titles_normalized']}")
        logger.info(f"  - Attributes fixed: {stats['attributes_fixed']}")
        logger.info(f"  - Prices standardized: {stats['prices_standardized']}")
        
        return {
            'products': cleaned_products,
            'ai_technique': '🧹 SBERT + Fuzzy Matching',
            'is_ai_powered': self.use_ai,
            **stats
        }
    
    def get_cleaning_recommendations(self, products: List[Dict[str, Any]]) -> List[str]:
        """
        Analyze catalog and provide cleaning recommendations
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not products:
            return ["No products to analyze"]
        
        # Check for potential duplicates
        duplicate_groups = self.duplicate_remover.find_duplicates(products, 0.85)
        if duplicate_groups:
            recommendations.append(
                f"Found {len(duplicate_groups)} potential duplicate groups. "
                "Consider running duplicate removal."
            )
        
        # Check title quality
        messy_titles = sum(1 for p in products 
                          if 'title' in p and any(char in str(p['title']).lower() 
                          for char in ['🔥', '⚡', 'sale', 'discount']))
        if messy_titles > len(products) * 0.1:
            recommendations.append(
                f"{messy_titles} products have promotional text in titles. "
                "Consider running title normalization."
            )
        
        # Check price consistency
        missing_prices = sum(1 for p in products if not p.get('price'))
        if missing_prices:
            recommendations.append(
                f"{missing_prices} products missing price information."
            )
        
        # Check attribute completeness
        missing_category = sum(1 for p in products if not p.get('category'))
        if missing_category > len(products) * 0.2:
            recommendations.append(
                f"{missing_category} products missing category. "
                "Consider categorization."
            )
        
        if not recommendations:
            recommendations.append("Catalog appears clean! 🎉")
        
        return recommendations
