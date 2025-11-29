"""
Price Standardizer
Standardizes prices and currencies
"""
import re
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


class PriceStandardizer:
    """Standardize prices and currencies"""
    
    # Exchange rates (approximate, for demonstration)
    # In production, use a real-time API
    EXCHANGE_RATES = {
        'USD': 1.0,
        'PKR': 0.0036,  # Pakistani Rupee
        'EUR': 1.10,
        'GBP': 1.27,
        'INR': 0.012,   # Indian Rupee
        'CNY': 0.14,    # Chinese Yuan
        'JPY': 0.0067   # Japanese Yen
    }
    
    def standardize_price(self, price: Any, currency: str = 'USD') -> float:
        """
        Standardize price to USD
        
        Args:
            price: Price value (can be string or number)
            currency: Currency code
            
        Returns:
            Price in USD as float
        """
        # Parse price if it's a string
        if isinstance(price, str):
            price = self.parse_price_string(price)
        
        try:
            price = float(price)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert price to float: {price}")
            return 0.0
        
        # Convert to USD
        currency_upper = currency.upper()
        if currency_upper in self.EXCHANGE_RATES:
            rate = self.EXCHANGE_RATES[currency_upper]
            price_usd = price * rate
        else:
            logger.warning(f"Unknown currency: {currency}, assuming USD")
            price_usd = price
        
        return round(price_usd, 2)
    
    def parse_price_string(self, price_str: str) -> float:
        """
        Parse price from string
        
        Examples:
            "$19.99" -> 19.99
            "Rs. 2,500" -> 2500.0
            "€15.50-€20.00" -> 15.50 (takes minimum)
        """
        if not price_str:
            return 0.0
        
        # Remove currency symbols and common prefixes
        price_str = re.sub(r'[^\d\.,\-]', '', str(price_str))
        
        # Handle price ranges (take minimum)
        if '-' in price_str:
            parts = price_str.split('-')
            if parts:
                price_str = parts[0]
        
        # Remove commas
        price_str = price_str.replace(',', '')
        
        try:
            return float(price_str)
        except ValueError:
            logger.warning(f"Could not parse price: {price_str}")
            return 0.0
    
    def detect_currency(self, price_str: str) -> str:
        """
        Detect currency from price string
        
        Returns:
            Currency code or 'USD' if unknown
        """
        if not isinstance(price_str, str):
            return 'USD'
        
        price_str = price_str.upper()
        
        # Check for currency symbols and codes
        currency_indicators = {
            '$': 'USD',
            'USD': 'USD',
            '₨': 'PKR',
            'RS': 'PKR',
            'PKR': 'PKR',
            '€': 'EUR',
            'EUR': 'EUR',
            '£': 'GBP',
            'GBP': 'GBP',
            '₹': 'INR',
            'INR': 'INR',
            '¥': 'CNY',
            'CNY': 'CNY',
            'JPY': 'JPY'
        }
        
        for indicator, currency in currency_indicators.items():
            if indicator in price_str:
                return currency
        
        return 'USD'
    
    def format_price(self, price: float, currency: str = 'USD') -> str:
        """
        Format price with currency symbol
        
        Returns:
            Formatted price string like "$19.99"
        """
        symbols = {
            'USD': '$',
            'PKR': 'Rs.',
            'EUR': '€',
            'GBP': '£',
            'INR': '₹',
            'CNY': '¥',
            'JPY': '¥'
        }
        
        symbol = symbols.get(currency.upper(), '$')
        return f"{symbol}{price:.2f}"
    
    def standardize_product_price(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize price in product dictionary
        
        Updates price to USD and adds original_price/original_currency
        """
        if 'price' not in product:
            return product
        
        original_price = product['price']
        original_currency = product.get('currency', 'USD')
        
        # If price is a string, try to detect currency
        if isinstance(original_price, str):
            detected_currency = self.detect_currency(original_price)
            if detected_currency != 'USD':
                original_currency = detected_currency
        
        # Standardize to USD
        standardized_price = self.standardize_price(original_price, original_currency)
        
        # Update product
        product['price'] = standardized_price
        product['currency'] = 'USD'
        product['original_price'] = original_price
        product['original_currency'] = original_currency
        
        return product
