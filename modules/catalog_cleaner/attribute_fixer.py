"""
Attribute Fixer
Standardizes product attributes like colors, sizes, materials
"""
from typing import Dict, Any
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


class AttributeFixer:
    """Fix and standardize product attributes"""
    
    # Color standardization map
    COLOR_MAP = {
        'blk': 'Black', 'black': 'Black', 'noir': 'Black',
        'wht': 'White', 'white': 'White', 'blanc': 'White',
        'red': 'Red', 'rouge': 'Red',
        'blu': 'Blue', 'blue': 'Blue', 'bleu': 'Blue',
        'grn': 'Green', 'green': 'Green', 'vert': 'Green',
        'ylw': 'Yellow', 'yellow': 'Yellow', 'jaune': 'Yellow',
        'slv': 'Silver', 'silver': 'Silver', 'argent': 'Silver',
        'gld': 'Gold', 'gold': 'Gold', 'or': 'Gold',
        'gry': 'Gray', 'gray': 'Gray', 'grey': 'Gray', 'gris': 'Gray',
    }
    
    # Size standardization
    SIZE_MAP = {
        'xs': 'XS', 'extra small': 'XS',
        's': 'S', 'small': 'S',
        'm': 'M', 'medium': 'M', 'med': 'M',
        'l': 'L', 'large': 'L', 'lrg': 'L',
        'xl': 'XL', 'extra large': 'XL',
        'xxl': 'XXL', '2xl': 'XXL',
        'xxxl': 'XXXL', '3xl': 'XXXL'
    }
    
    def fix_attributes(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix all attributes in a product
        
        Args:
            product: Product dictionary
            
        Returns:
            Product with fixed attributes
        """
        fixed = product.copy()
        
        # Fix color
        if 'color' in fixed:
            fixed['color'] = self.normalize_color(fixed['color'])
        
        # Fix size
        if 'size' in fixed:
            fixed['size'] = self.normalize_size(fixed['size'])
        
        # Fix material
        if 'material' in fixed:
            fixed['material'] = self.normalize_material(fixed['material'])
        
        # Extract attributes from title if missing
        if 'title' in fixed:
            self._extract_attributes_from_title(fixed)
        
        return fixed
    
    def normalize_color(self, color: str) -> str:
        """Standardize color name"""
        if not color:
            return ""
        
        color_lower = color.lower().strip()
        
        # Check exact match
        if color_lower in self.COLOR_MAP:
            return self.COLOR_MAP[color_lower]
        
        # Check if it contains a known color
        for key, value in self.COLOR_MAP.items():
            if key in color_lower:
                return value
        
        # Return capitalized version
        return color.capitalize()
    
    def normalize_size(self, size: str) -> str:
        """Standardize size"""
        if not size:
            return ""
        
        size_lower = size.lower().strip()
        
        # Check direct mapping
        if size_lower in self.SIZE_MAP:
            return self.SIZE_MAP[size_lower]
        
        # Handle numeric sizes (keep as is)
        if re.match(r'^\d+(\.\d+)?$', size):
            return size
        
        # Handle shoe sizes like "US 10", "UK 8"
        if re.match(r'^(us|uk|eu)\s*\d+', size_lower):
            return size.upper()
        
        return size.capitalize()
    
    def normalize_material(self, material: str) -> str:
        """Standardize material description"""
        if not material:
            return ""
        
        # Common materials
        material_map = {
            'cotton': 'Cotton',
            'polyester': 'Polyester',
            'leather': 'Leather',
            'plastic': 'Plastic',
            'metal': 'Metal',
            'wood': 'Wood',
            'glass': 'Glass',
            'silk': 'Silk',
            'wool': 'Wool',
            'nylon': 'Nylon'
        }
        
        material_lower = material.lower().strip()
        
        for key, value in material_map.items():
            if key in material_lower:
                return value
        
        return material.capitalize()
    
    def _extract_attributes_from_title(self, product: Dict[str, Any]):
        """Extract color, size from title if not present"""
        title = product.get('title', '').lower()
        
        # Extract color if missing
        if not product.get('color'):
            for color_key in self.COLOR_MAP.keys():
                if color_key in title:
                    product['color'] = self.COLOR_MAP[color_key]
                    break
        
        # Extract size if missing
        if not product.get('size'):
            # Look for size patterns
            size_pattern = r'\b(xs|s|m|l|xl|xxl|xxxl|\d+(\.\d+)?)\b'
            match = re.search(size_pattern, title)
            if match:
                product['size'] = self.normalize_size(match.group(0))
