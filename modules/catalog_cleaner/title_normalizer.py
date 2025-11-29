"""
Title Normalizer
Cleans and standardizes product titles
"""
import re
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


class TitleNormalizer:
    """Normalize and clean product titles"""
    
    # Common promotional words to remove
    PROMO_WORDS = [
        'sale', 'discount', 'offer', 'deal', 'new', 'hot', 'best',
        'limited', 'special', 'free shipping', 'fast delivery',
        '🔥', '⚡', '✨', '🎉', '💥', '🌟'
    ]
    
    # Common brands for standardization
    BRAND_MAP = {
        'apple': 'Apple',
        'samsung': 'Samsung',
        'sony': 'Sony',
        'lg': 'LG',
        'dell': 'Dell',
        'hp': 'HP',
        'lenovo': 'Lenovo',
        'nike': 'Nike',
        'adidas': 'Adidas',
        'puma': 'Puma'
    }
    
    def normalize_title(self, title: str) -> str:
        """
        Clean and normalize a product title
        
        Args:
            title: Raw product title
            
        Returns:
            Cleaned title
        """
        if not title:
            return ""
        
        original = title
        
        # Remove HTML tags
        title = re.sub(r'<[^>]+>', '', title)
        
        # Remove special characters but keep alphanumeric, spaces, and basic punctuation
        title = re.sub(r'[^\w\s\-\(\)\/\.]', ' ', title)
        
        # Remove promotional words
        for word in self.PROMO_WORDS:
            title = re.sub(rf'\b{word}\b', '', title, flags=re.IGNORECASE)
        
        # Fix multiple spaces
        title = re.sub(r'\s+', ' ', title)
        
        # Capitalize properly
        title = self._fix_capitalization(title)
        
        # Standardize brands
        title = self._standardize_brands(title)
        
        # Trim
        title = title.strip()
        
        if title != original:
            logger.debug(f"Normalized: '{original}' -> '{title}'")
        
        return title
    
    def _fix_capitalization(self, title: str) -> str:
        """Fix title capitalization"""
        # If all caps or all lowercase, apply title case
        if title.isupper() or title.islower():
            # Keep certain words lowercase
            small_words = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 
                          'in', 'of', 'on', 'or', 'the', 'to', 'with'}
            
            words = title.lower().split()
            result = []
            
            for i, word in enumerate(words):
                # Always capitalize first and last word
                if i == 0 or i == len(words) - 1:
                    result.append(word.capitalize())
                # Keep small words lowercase unless they're first/last
                elif word in small_words:
                    result.append(word)
                else:
                    result.append(word.capitalize())
            
            return ' '.join(result)
        
        return title
    
    def _standardize_brands(self, title: str) -> str:
        """Standardize brand names"""
        words = title.split()
        result = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.BRAND_MAP:
                result.append(self.BRAND_MAP[word_lower])
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def extract_brand(self, title: str) -> str:
        """
        Extract brand name from title
        
        Returns:
            Brand name or empty string
        """
        title_lower = title.lower()
        
        for brand_key, brand_name in self.BRAND_MAP.items():
            if brand_key in title_lower:
                return brand_name
        
        # Try to get first word as brand (common pattern)
        words = title.split()
        if words:
            first_word = words[0]
            # If it's capitalized and not a common word, likely a brand
            if first_word[0].isupper() and len(first_word) > 2:
                return first_word
        
        return ""
