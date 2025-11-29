"""
Marketplace Scraper Module
Extracts product data from multiple e-commerce platforms
"""
from .scrapers import MarketplaceScraper
from .storage import DataWarehouse

__all__ = ['MarketplaceScraper', 'DataWarehouse']
