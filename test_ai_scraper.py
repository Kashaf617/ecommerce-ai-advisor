"""
Test AI-powered marketplace scraper with SBERT semantic search
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from modules.marketplace_scraper.scrapers import MarketplaceScraper
from utils.logger import get_logger

logger = get_logger(__name__)

def test_ai_scraper():
    """Test the AI-powered scraper"""
    print("\n" + "="*60)
    print("Testing AI-Powered Marketplace Scraper")
    print("="*60 + "\n")
    
    # Test 1: Standard scraper (no AI)
    print("Test 1: Traditional Scraper (No AI)")
    print("-" * 60)
    scraper_traditional = MarketplaceScraper(use_ai=False)
    products_traditional = scraper_traditional.scrape_platform('amazon', 'wireless headphones', max_results=10)
    print(f"✓ Traditional scraper found {len(products_traditional)} products\n")
    
    # Test 2: AI-powered scraper with semantic search
    print("Test 2: AI-Powered Scraper (With SBERT Semantic Search)")
    print("-" * 60)
    try:
        scraper_ai = MarketplaceScraper(use_ai=True)
        products_ai = scraper_ai.scrape_platform('amazon', 'wireless headphones', max_results=10)
        print(f"✓ AI scraper found {len(products_ai)} products")
        
        # Show similarity scores if available
        if products_ai and 'similarity_score' in products_ai[0]:
            print("\nTop 3 products by semantic similarity:")
            for i, product in enumerate(products_ai[:3], 1):
                score = product.get('similarity_score', 0)
                print(f"  {i}. {product['title']} (Similarity: {score:.3f})")
        print()
    except Exception as e:
        print(f"✗ AI scraper failed: {e}")
        print("  (This is normal if SBERT models aren't downloaded yet)\n")
    
    # Test 3: Deduplication
    print("Test 3: AI-Powered Deduplication")
    print("-" * 60)
    try:
        all_products = scraper_ai.scrape_and_deduplicate('laptop', max_results_per_platform=15)
        print(f"✓ Found {len(all_products)} unique products after deduplication\n")
    except Exception as e:
        print(f"✗ Deduplication test failed: {e}\n")
    
    print("="*60)
    print("Test Complete!")
    print("="*60)

if __name__ == "__main__":
    test_ai_scraper()
