"""
Test Script - E-Commerce Business Automation Platform
Tests each module independently to verify functionality
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("E-COMMERCE AUTOMATION PLATFORM - MODULE TEST")
print("=" * 60)
print()

# Test 1: Marketplace Scraper
print("Test 1: Marketplace Scraper")
print("-" * 40)
try:
    from modules.marketplace_scraper import MarketplaceScraper, DataWarehouse
    
    scraper = MarketplaceScraper()
    warehouse = DataWarehouse()
    
    # Scrape sample data
    products = scraper.scrape_platform('amazon', 'Electronics', max_results=10)
    print(f"✅ Scraped {len(products)} products from Amazon")
    
    # Store in warehouse
    warehouse.store_products(products, 'amazon', 'Electronics')
    print(f"✅ Stored products in warehouse")
    
    # Get stats
    stats = warehouse.get_statistics()
    print(f"✅ Warehouse stats: {stats.get('total_products', 0)} total products")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Test 2: Trend Forecasting
print("Test 2: Trend and Demand Forecasting")
print("-" * 40)
try:
    from modules.trend_forecasting import TrendAnalyzer
    
    analyzer = TrendAnalyzer()
    trends = analyzer.analyze_category_trends('Electronics')
    
    if trends:
        print(f"✅ Category: {trends.get('category')}")
        print(f"✅ Average Price: ${trends.get('average_price', 0):.2f}")
        print(f"✅ Total Products Analyzed: {trends.get('total_products', 0)}")
    
    forecast = analyzer.forecast_demand('Electronics', days_ahead=30)
    if forecast:
        print(f"✅ Demand Level: {forecast.get('demand_level', 'N/A')}")
        print(f"✅ Demand Score: {forecast.get('current_demand_score', 0)}/100")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Test 3: Competitor Analysis
print("Test 3: Competitor Analysis")
print("-" * 40)
try:
    from modules.competitor_analysis import CompetitorAnalyzer
    
    comp_analyzer = CompetitorAnalyzer()
    competitors = comp_analyzer.identify_competitors('Electronics', (40, 60))
    
    print(f"✅ Found {len(competitors)} competitors")
    if competitors:
        print(f"✅ Top competitor: {competitors[0]['seller_name']}")
        print(f"✅ Competitive score: {competitors[0]['competitive_score']}/100")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Test 4: Supplier Sourcing
print("Test 4: Supplier and Sourcing")
print("-" * 40)
try:
    from modules.supplier_sourcing import SupplierRecommender
    
    supplier_rec = SupplierRecommender()
    suppliers = supplier_rec.recommend_suppliers('Electronics', 1000, 100, 'balanced')
    
    print(f"✅ Found {len(suppliers)} suitable suppliers")
    if suppliers:
        print(f"✅ Top supplier: {suppliers[0]['name']}")
        print(f"✅ Est. cost: ${suppliers[0]['estimated_unit_cost']:.2f}/unit")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Test 5: Pricing Calculator
print("Test 5: Pricing and Profitability")
print("-" * 40)
try:
    from modules.pricing_estimator import PricingCalculator
    
    pricing_calc = PricingCalculator()
    pricing = pricing_calc.calculate_pricing(20, 'amazon', category='Electronics')
    
    print(f"✅ Recommended Price: ${pricing.get('recommended_price', 0):.2f}")
    print(f"✅ Profit Margin: {pricing.get('profit_margin', 0):.1f}%")
    print(f"✅ Profit: ${pricing.get('profit', 0):.2f}")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Test 6: Platform Recommender
print("Test 6: Platform Recommendations")
print("-" * 40)
try:
    from modules.platform_recommender import PlatformRecommender
    
    platform_rec = PlatformRecommender()
    platforms = platform_rec.recommend_platforms({
        'category': 'Electronics',
        'price': 50,
        'target_market': 'pakistan',
        'business_model': 'ecommerce'
    })
    
    print(f"✅ Found {len(platforms)} platform recommendations")
    if platforms:
        print(f"✅ Best platform: {platforms[0]['platform_name']}")
        print(f"✅ Suitability: {platforms[0]['suitability_rating']}")
        print(f"✅ Score: {platforms[0]['recommendation_score']}/100")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Test 7: Audience Profiler
print("Test 7: Target Audience Profiling")
print("-" * 40)
try:
    from modules.audience_recommender import AudienceProfiler
    
    audience_prof = AudienceProfiler()
    profile = audience_prof.create_audience_profile({
        'category': 'Electronics',
        'price': 50,
        'features': []
    })
    
    print(f"✅ Category: {profile.get('product_category')}")
    demographics = profile.get('primary_demographics', {})
    print(f"✅ Age Range: {demographics.get('age_range', 'N/A')}")
    print(f"✅ Income Level: {demographics.get('income_level', 'N/A')}")
    
    personas = profile.get('buyer_personas', [])
    if personas:
        print(f"✅ Primary Persona: {personas[0]['name']}")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Test 8: Marketing Strategy
print("Test 8: Marketing Strategy Generator")
print("-" * 40)
try:
    from modules.marketing_strategy import MarketingStrategyGenerator
    
    marketing_gen = MarketingStrategyGenerator()
    strategy = marketing_gen.generate_strategy(
        {'category': 'Electronics', 'price': 50},
        {'primary_demographics': {'age_range': '18-45'}, 'buyer_personas': []},
        [{'platform': 'amazon', 'recommendation_score': 85}],
        500
    )
    
    overview = strategy.get('overview', {})
    print(f"✅ Budget: ${overview.get('total_budget', 0)}")
    print(f"✅ Budget Tier: {overview.get('budget_tier', 'N/A')}")
    print(f"✅ Strategy Period: {overview.get('strategy_period', 'N/A')}")
    
    channels = strategy.get('channel_strategy', {}).get('channels', [])
    print(f"✅ Marketing Channels: {len(channels)}")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Overall result
print("=" * 60)
print("✅ ALL MODULE TESTS COMPLETE!")
print("=" * 60)
print()
print("Next steps:")
print("1. Run the web application: python app.py")
print("2. Open browser: http://localhost:5000")
print("3. Enter product details and click 'Start Complete Analysis'")
print()
