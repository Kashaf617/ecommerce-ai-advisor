"""
Marketplace Scraper - Data extraction from e-commerce platforms
"""
import requests
from bs4 import BeautifulSoup
import time
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import SCRAPER_CONFIG, PLATFORMS, RAW_DATA_DIR
from utils.logger import get_logger
from utils.helpers import clean_text, get_timestamp

logger = get_logger(__name__)


class BaseScraper:
    """Base class for platform-specific scrapers"""
    
    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        self.platform_config = PLATFORMS.get(platform_name, {})
        self.base_url = self.platform_config.get('base_url', '')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': SCRAPER_CONFIG['user_agent']
        })
    
    def _make_request(self, url: str) -> Optional[BeautifulSoup]:
        """Make HTTP request with retry logic"""
        for attempt in range(SCRAPER_CONFIG['retry_attempts']):
            try:
                time.sleep(SCRAPER_CONFIG['delay_between_requests'])
                response = self.session.get(
                    url,
                    timeout=SCRAPER_CONFIG['timeout']
                )
                response.raise_for_status()
                return BeautifulSoup(response.content, 'lxml')
            except Exception as e:
                logger.warning(f"Request attempt {attempt + 1} failed for {url}: {e}")
                if attempt == SCRAPER_CONFIG['retry_attempts'] - 1:
                    logger.error(f"All attempts failed for {url}")
                    return None
                time.sleep(random.uniform(1, 3))
        return None
    
    def search_products(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for products (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def get_product_details(self, product_url: str) -> Dict[str, Any]:
        """Get detailed product information (to be implemented by subclasses)"""
        raise NotImplementedError


class AmazonScraper(BaseScraper):
    """Amazon marketplace scraper"""
    
    def __init__(self):
        super().__init__('amazon')
        logger.info("Initialized Amazon scraper")
    
    def search_products(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Search Amazon for products
        Note: This is a simplified implementation. In production, consider using Amazon API
        """
        logger.info(f"Searching Amazon for: {query}")
        
        # For demo purposes, return sample data
        # In production, implement actual scraping or use Amazon Product Advertising API
        sample_products = self._generate_sample_data(query, max_results)
        logger.info(f"Found {len(sample_products)} products")
        return sample_products
    
    def _generate_sample_data(self, query: str, count: int) -> List[Dict[str, Any]]:
        """Generate sample product data for demonstration"""
        products = []
        categories = ['Electronics', 'Fashion', 'Home & Kitchen', 'Sports', 'Books']
        
        for i in range(min(count, 50)):
            product = {
                'platform': 'amazon',
                'product_id': f'AMZ{i+1:05d}',
                'title': f'{query} Product {i+1}',
                'category': random.choice(categories),
                'price': round(random.uniform(10, 500), 2),
                'original_price': round(random.uniform(15, 600), 2),
                'rating': round(random.uniform(3.5, 5.0), 1),
                'reviews_count': random.randint(10, 5000),
                'availability': random.choice(['In Stock', 'Limited Stock', 'Pre-order']),
                'seller': f'Seller {random.randint(1, 100)}',
                'seller_rating': round(random.uniform(4.0, 5.0), 1),
                'shipping': random.choice(['Free Shipping', 'Paid Shipping']),
                'prime_eligible': random.choice([True, False]),
                'url': f'https://amazon.com/product/{i+1}',
                'image_url': f'https://via.placeholder.com/300x300?text=Product+{i+1}',
                'description': f'High-quality {query} with excellent features and performance.',
                'brand': f'Brand {chr(65 + i % 26)}',
                'scraped_at': get_timestamp()
            }
            
            # Calculate discount percentage
            if product['original_price'] > product['price']:
                product['discount_percent'] = round(
                    ((product['original_price'] - product['price']) / product['original_price']) * 100,
                    1
                )
            else:
                product['discount_percent'] = 0
            
            products.append(product)
        
        return products
    
    def get_product_details(self, product_url: str) -> Dict[str, Any]:
        """Get detailed product information"""
        logger.info(f"Fetching product details from: {product_url}")
        # Implementation would go here
        return {}


class DarazScraper(BaseScraper):
    """Daraz marketplace scraper"""
    
    def __init__(self):
        super().__init__('daraz')
        logger.info("Initialized Daraz scraper")
    
    def search_products(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search Daraz for products"""
        logger.info(f"Searching Daraz for: {query}")
        
        # For demo purposes, return sample data
        sample_products = self._generate_sample_data(query, max_results)
        logger.info(f"Found {len(sample_products)} products")
        return sample_products
    
    def _generate_sample_data(self, query: str, count: int) -> List[Dict[str, Any]]:
        """Generate sample product data for demonstration"""
        products = []
        categories = ['Electronics', 'Fashion', 'Beauty', 'Home', 'Sports']
        
        for i in range(min(count, 50)):
            product = {
                'platform': 'daraz',
                'product_id': f'DRZ{i+1:05d}',
                'title': f'{query} Item {i+1}',
                'category': random.choice(categories),
                'price': round(random.uniform(500, 50000), 2),  # PKR
                'original_price': round(random.uniform(600, 60000), 2),
                'rating': round(random.uniform(3.0, 5.0), 1),
                'reviews_count': random.randint(5, 3000),
                'availability': random.choice(['In Stock', 'Out of Stock', 'Pre-order']),
                'seller': f'Daraz Seller {random.randint(1, 100)}',
                'seller_rating': round(random.uniform(3.5, 5.0), 1),
                'shipping': random.choice(['Free Delivery', 'Cash on Delivery']),
                'verified_seller': random.choice([True, False]),
                'url': f'https://daraz.pk/product/{i+1}',
                'image_url': f'https://via.placeholder.com/300x300?text=Item+{i+1}',
                'description': f'Quality {query} available in Pakistan with fast delivery.',
                'brand': f'Brand {chr(65 + i % 26)}',
                'scraped_at': get_timestamp()
            }
            
            # Calculate discount percentage
            if product['original_price'] > product['price']:
                product['discount_percent'] = round(
                    ((product['original_price'] - product['price']) / product['original_price']) * 100,
                    1
                )
            else:
                product['discount_percent'] = 0
            
            products.append(product)
        
        return products


class MarketplaceScraper:
    """Main scraper class that coordinates all platform scrapers with AI enhancements"""
    
    def __init__(self, use_ai: bool = True):
        self.scrapers = {
            'amazon': AmazonScraper(),
            'daraz': DarazScraper()
        }
        self.use_ai = use_ai
        self.semantic_matcher = None
        
        if use_ai:
            try:
                from .semantic_matcher import SemanticProductMatcher
                self.semantic_matcher = SemanticProductMatcher()
                logger.info("AI-powered semantic matching enabled")
            except Exception as e:
                logger.warning(f"Could not load semantic matcher: {e}. Using traditional search.")
                self.use_ai = False
        
        logger.info(f"Initialized Marketplace Scraper with platforms: Amazon, Daraz (AI: {self.use_ai})")
    
    def scrape_all_platforms(self, query: str, max_results_per_platform: int = 50,
                           use_semantic_search: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scrape product data from all enabled platforms with AI enhancements
        
        Args:
            query: Search query (product category or keyword)
            max_results_per_platform: Maximum products to scrape per platform
            use_semantic_search: Whether to use AI semantic search for filtering
        
        Returns:
            Dictionary with platform names as keys and product lists as values
        """
        all_products = {}
        
        for platform_name, scraper in self.scrapers.items():
            if PLATFORMS[platform_name].get('enabled', False):
                try:
                    logger.info(f"Scraping {platform_name}...")
                    products = scraper.search_products(query, max_results_per_platform)
                    
                    # Apply AI semantic filtering if enabled
                    if self.use_ai and use_semantic_search and self.semantic_matcher and products:
                        logger.info(f"Applying semantic search to {len(products)} products...")
                        products = self.semantic_matcher.find_similar_products(
                            query, 
                            products,
                            top_k=max_results_per_platform,
                            threshold=0.3  # Relatively low threshold to keep relevant items
                        )
                        logger.info(f"Semantic search returned {len(products)} relevant products")
                    
                    all_products[platform_name] = products
                    logger.info(f"Successfully scraped {len(products)} products from {platform_name}")
                except Exception as e:
                    logger.error(f"Error scraping {platform_name}: {e}")
                    all_products[platform_name] = []
        
        return all_products
    
    def scrape_and_deduplicate(self, query: str, max_results_per_platform: int = 50) -> List[Dict[str, Any]]:
        """
        Scrape all platforms and remove duplicates using AI
        
        Args:
            query: Search query
            max_results_per_platform: Max products per platform
            
        Returns:
            Deduplicated list of products from all platforms
        """
        # Scrape all platforms
        platform_products = self.scrape_all_platforms(query, max_results_per_platform)
        
        # Combine all products
        all_products = []
        for platform, products in platform_products.items():
            all_products.extend(products)
        
        logger.info(f"Combined {len(all_products)} products from all platforms")
        
        # Deduplicate using AI if enabled
        if self.use_ai and self.semantic_matcher and all_products:
            logger.info("Deduplicating products using semantic similarity...")
            all_products = self.semantic_matcher.deduplicate_products(
                all_products,
                similarity_threshold=0.85
            )
            logger.info(f"After deduplication: {len(all_products)} unique products")
        
        return all_products
    
    def scrape_platform(self, platform: str, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Scrape product data from a specific platform
        
        Args:
            platform: Platform name ('amazon', 'daraz', etc.)
            query: Search query
            max_results: Maximum products to scrape
        
        Returns:
            List of product dictionaries
        """
        if platform not in self.scrapers:
            logger.error(f"Platform {platform} not supported")
            return []
        
        try:
            logger.info(f"Scraping {platform} for: {query}")
            products = self.scrapers[platform].search_products(query, max_results)
            
            # Apply semantic search if enabled
            if self.use_ai and self.semantic_matcher and products:
                products = self.semantic_matcher.find_similar_products(
                    query,
                    products,
                    top_k=max_results,
                    threshold=0.3
                )
            
            logger.info(f"Successfully scraped {len(products)} products")
            return products
        except Exception as e:
            logger.error(f"Error scraping {platform}: {e}")
            return []
