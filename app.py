"""
E-Commerce Business Automation Platform - Main Flask Application
Run all modules sequentially to generate comprehensive business analysis
"""
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import json
from pathlib import Path
import traceback

# Import all modules
from modules.marketplace_scraper import MarketplaceScraper, DataWarehouse
from modules.trend_forecasting import TrendAnalyzer
from modules.competitor_analysis import CompetitorAnalyzer
from modules.supplier_sourcing import SupplierRecommender
from modules.pricing_estimator import PricingCalculator
from modules.platform_recommender import PlatformRecommender
from modules.audience_recommender import AudienceProfiler
from modules.marketing_strategy import MarketingStrategyGenerator
from modules.catalog_cleaner import CatalogCleaner

from config import FLASK_CONFIG, WAREHOUSE_DIR
from utils.logger import get_logger
from utils.helpers import get_timestamp, save_to_json

# Initialize Flask app
app = Flask(__name__)
app.config.update(FLASK_CONFIG)
CORS(app)

logger = get_logger(__name__)

# Initialize all modules with AI ENABLED (where supported)
scraper = MarketplaceScraper(use_ai=True)
warehouse = DataWarehouse()
trend_analyzer = TrendAnalyzer(use_ai=True)
competitor_analyzer = CompetitorAnalyzer()  # Uses AI internally (BERT sentiment)
supplier_recommender = SupplierRecommender(use_ai=True)
pricing_calculator = PricingCalculator(use_ai=True)
platform_recommender = PlatformRecommender()  # Always uses AI (Random Forest)
audience_profiler = AudienceProfiler()  # Always uses AI (K-Means)
marketing_generator = MarketingStrategyGenerator()  # Uses AI templates
catalog_cleaner = CatalogCleaner(use_ai=True)  # AI-powered catalog cleaning

logger.info("All 9 modules initialized successfully")


def convert_to_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


@app.route('/')
def index():
    """Home page"""
    return render_template('dashboard.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_product():
    """
    Main API endpoint - runs all modules sequentially
    Expects JSON with: category, price, budget, quantity, target_market
    """
    try:
        data = request.json
        logger.info(f"Starting analysis for: {data}")
        
        # Extract parameters
        category = data.get('category', 'Electronics')
        price = float(data.get('price', 50))
        budget = float(data.get('budget', 1000))
        quantity = int(data.get('quantity', 100))
        target_market = data.get('target_market', 'pakistan')
        marketing_budget = float(data.get('marketing_budget', 500))
        product_features = data.get('features', [])
        
        results = {
            'status': 'success',
            'timestamp': get_timestamp(),
            'input': data
        }
        
        # Module 1: Scrape marketplace data
        logger.info("Step 1/9: Scraping marketplace data...")
        scraped_data = scraper.scrape_all_platforms(category, max_results_per_platform=50)
        
        # Store in warehouse
        for platform, products in scraped_data.items():
            warehouse.store_products(products, platform, category)
        
        # Collect all products for cleaning
        all_products = []
        for platform, products in scraped_data.items():
            all_products.extend(products)
        
        results['module_1_scraper'] = {
            'ai_technique': '🤖 SBERT Semantic Search',
            'is_ai_powered': True,
            'total_products_scraped': sum(len(products) for products in scraped_data.values()),
            'platforms': list(scraped_data.keys()),
            'status': 'completed'
        }
        
        # Module 9: Clean catalog data (NEW!)
        logger.info("Step 2/9: Cleaning and normalizing catalog...")
        cleaning_results = catalog_cleaner.clean_catalog(
            all_products,
            remove_duplicates=True,
            normalize_titles=True,
            fix_attributes=True,
            standardize_prices=True
        )
        
        results['module_9_cleaner'] = {
            'ai_technique': '🧹 SBERT + Fuzzy Matching',
            'is_ai_powered': True,
            'products_before': cleaning_results['products_before'],
            'products_after': cleaning_results['products_after'],
            'duplicates_removed': cleaning_results['duplicates_removed'],
            'titles_normalized': cleaning_results['titles_normalized'],
            'attributes_fixed': cleaning_results['attributes_fixed'],
            'prices_standardized': cleaning_results['prices_standardized'],
            'status': 'completed'
        }
        
        # Module 2: Trend and demand forecasting
        logger.info("Step 3/9: Analyzing trends and forecasting demand...")
        trends = trend_analyzer.analyze_category_trends(category)
        demand_forecast = trend_analyzer.forecast_demand(category, days_ahead=30)
        trending_products = trend_analyzer.identify_trending_products(top_n=10)
        
        results['module_2_trends'] = {
            'ai_technique': '📊 LSTM Neural Network',
            'is_ai_powered': demand_forecast.get('is_ml_forecast', False),
            'forecast_method': demand_forecast.get('forecast_method', 'Statistical'),
            'category_trends': trends,
            'demand_forecast': demand_forecast,
            'trending_products': trending_products,
            'status': 'completed'
        }
        
        # Module 3: Competitor analysis
        logger.info("Step 4/9: Analyzing competitors...")
        competitors = competitor_analyzer.identify_competitors(category, (price * 0.7, price * 1.3))
        
        user_product = {
            'category': category,
            'price': price
        }
        comparison = competitor_analyzer.compare_with_competitors(user_product, competitors)
        
        results['module_3_competitors'] = {
            'ai_technique': '💬 BERT Sentiment Analysis',
            'is_ai_powered': True,
            'total_competitors': len(competitors),
            'top_competitors': competitors[:5],
            'competitive_analysis': comparison,
            'status': 'completed'
        }
        
        # Module 4: Supplier sourcing
        logger.info("Step 5/9: Finding suppliers...")
        supplier_data = supplier_recommender.recommend_suppliers(
            category, budget, quantity, priority='balanced'
        )
        
        # Extract suppliers list from the returned dictionary
        suppliers = supplier_data.get('suppliers', []) if isinstance(supplier_data, dict) else supplier_data
        
        sourcing_strategy = supplier_recommender.generate_sourcing_strategy({
            'category': category,
            'budget': budget,
            'quantity': quantity,
            'business_model': 'ecommerce'
        })
        
        results['module_4_suppliers'] = {
            'ai_technique': '🔍 SBERT Semantic Matching',
            'is_ai_powered': True,
            'recommended_suppliers': suppliers[:3],
            'sourcing_strategy': sourcing_strategy,
            'accuracy_metrics': supplier_data.get('accuracy_metrics') if isinstance(supplier_data, dict) else None,
            'status': 'completed'
        }
        
        # Module 5: Pricing and profitability
        logger.info("Step 6/9: Calculating optimal pricing...")
        product_cost = suppliers[0]['estimated_unit_cost'] if suppliers else price * 0.4
        
        platform_profitability = pricing_calculator.compare_platform_profitability(
            product_cost, price
        )
        
        pricing_amazon = pricing_calculator.calculate_pricing(
            product_cost, 'amazon', category=category
        )
        
        pricing_daraz = pricing_calculator.calculate_pricing(
            product_cost, 'daraz', category=category
        )
        
        results['module_5_pricing'] = {
            'ai_technique': '💰 XGBoost ML Pricing',
            'is_ai_powered': True,
            'product_cost': product_cost,
            'platform_comparison': platform_profitability,
            'amazon_pricing': pricing_amazon,
            'daraz_pricing': pricing_daraz,
            'status': 'completed'
        }
        
        # Module 6: Platform recommendations
        logger.info("Step 7/9: Recommending platforms...")
        product_info = {
            'category': category,
            'price': price,
            'target_market': target_market,
            'business_model': 'ecommerce'
        }
        
        platform_recommendations = platform_recommender.recommend_platforms(product_info)
        multi_platform_strategy = platform_recommender.create_multi_platform_strategy(product_info)
        
        results['module_6_platforms'] = {
            'ai_technique': '🌳 Random Forest Classifier',
            'is_ai_powered': platform_recommendations[0].get('is_ml_prediction', False) if platform_recommendations else False,
            'recommended_platforms': platform_recommendations,
            'multi_platform_strategy': multi_platform_strategy,
            'status': 'completed'
        }
        
        # Module 7: Target audience profiling
        logger.info("Step 8/9: Profiling target audience...")
        audience_profile = audience_profiler.create_audience_profile({
            'category': category,
            'price': price,
            'features': product_features
        })
        
        results['module_7_audience'] = {
            'ai_technique': '👥 K-Means Clustering',
            'is_ai_powered': True,
            'audience_profile': audience_profile,
            'status': 'completed'
        }
        
        # Module 8: Marketing strategy
        logger.info("Step 9/9: Generating marketing strategy...")
        marketing_strategy = marketing_generator.generate_strategy(
            product_info,
            audience_profile,
            platform_recommendations,
            marketing_budget
        )
        
        results['module_8_marketing'] = {
            'ai_technique': '📱 AI Template Generation',
            'is_ai_powered': marketing_strategy.get('is_ai_generated', True) if isinstance(marketing_strategy, dict) else True,
            'marketing_strategy': marketing_strategy,
            'status': 'completed'
        }
        
        # Save complete results
        results_file = WAREHOUSE_DIR / f"analysis_results_{get_timestamp().replace(':', '-').replace(' ', '_')}.json"
        save_to_json(results, results_file)
        results['results_file'] = str(results_file)
        
        logger.info("Analysis completed successfully")
        
        # Convert all numpy/pandas types to JSON-serializable types
        serializable_results = convert_to_serializable(results)
        
        return jsonify(serializable_results)
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/warehouse/stats', methods=['GET'])
def warehouse_stats():
    """Get warehouse statistics"""
    try:
        stats = warehouse.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'E-Commerce Automation Platform API is running',
        'modules': [
            'Marketplace Scraper',
            'Trend Forecasting',
            'Competitor Analysis',
            'Supplier Sourcing',
            'Pricing Calculator',
            'Platform Recommender',
            'Audience Profiler',
            'Marketing Strategy Generator'
        ]
    })


if __name__ == '__main__':
    logger.info("Starting E-Commerce Business Automation Platform...")
    logger.info(f"Server running on http://{FLASK_CONFIG['HOST']}:{FLASK_CONFIG['PORT']}")
    app.run(
        host=FLASK_CONFIG['HOST'],
        port=FLASK_CONFIG['PORT'],
        debug=FLASK_CONFIG['DEBUG']
    )
