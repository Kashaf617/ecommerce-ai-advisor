"""
AI Utilities - Clean access to all 9 AI-powered modules
AI-only implementation (no traditional fallbacks)
"""
from typing import Dict, Any, List
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# MODULE 1: MARKETPLACE SCRAPER (SBERT)
# ============================================================================

def ai_scrape_products(query: str, platform: str = 'amazon', max_results: int = 50) -> List[Dict[str, Any]]:
    """
    AI-powered product scraping with semantic search
    
    Args:
        query: Search query
        platform: Platform to scrape (amazon, daraz)
        max_results: Maximum results
        
    Returns:
        List of products with similarity scores
    """
    from modules.marketplace_scraper.scrapers import MarketplaceScraper
    
    scraper = MarketplaceScraper(use_ai=True)
    products = scraper.scrape_platform(platform, query, max_results)
    logger.info(f"AI scraped {len(products)} products for '{query}'")
    return products


def ai_deduplicate_products(products: List[Dict[str, Any]], threshold: float = 0.85) -> List[Dict[str, Any]]:
    """Remove duplicate products using AI similarity"""
    from modules.marketplace_scraper.semantic_matcher import SemanticProductMatcher
    
    matcher = SemanticProductMatcher()
    unique = matcher.deduplicate_products(products, threshold)
    logger.info(f"AI deduplicated: {len(products)} -> {len(unique)} products")
    return unique


# ============================================================================
# MODULE 2: TREND FORECASTING (LSTM)
# ============================================================================

def ai_forecast_trend(historical_data: pd.DataFrame, periods: int = 30) -> Dict[str, Any]:
    """
    AI-powered trend forecasting using LSTM
    
    Args:
        historical_data: DataFrame with historical data
        periods: Number of periods to forecast
        
    Returns:
        Forecast with confidence scores
    """
    from modules.trend_analyzer.lstm_forecaster import SimpleLSTMForecaster
    
    forecaster = SimpleLSTMForecaster()
    forecast = forecaster.forecast_trend(historical_data, periods)
    logger.info(f"AI forecasted {periods} periods using {forecast['method']}")
    return forecast


# ============================================================================
# MODULE 3: COMPETITOR ANALYSIS (BERT SENTIMENT)
# ============================================================================

def ai_analyze_sentiment(reviews: List[str]) -> Dict[str, Any]:
    """
    AI sentiment analysis using BERT
    
    Args:
        reviews: List of review texts
        
    Returns:
        Sentiment analysis results
    """
    from modules.competitor_analysis.bert_sentiment import BERTSentimentAnalyzer
    
    analyzer = BERTSentimentAnalyzer()
    sentiment = analyzer.analyze_reviews(reviews)
    logger.info(f"AI analyzed {sentiment['total_reviews_analyzed']} reviews: {sentiment['overall_sentiment']}")
    return sentiment


def ai_analyze_competitor(competitor_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze competitor with sentiment scoring"""
    from modules.competitor_analysis.bert_sentiment import BERTSentimentAnalyzer
    
    analyzer = BERTSentimentAnalyzer()
    enhanced = analyzer.analyze_competitor_sentiment(competitor_data)
    return enhanced


# ============================================================================
# MODULE 4: SUPPLIER SOURCING (SBERT)
# ============================================================================

def ai_find_suppliers(requirements: str, top_k: int = 5, budget: float = None, 
                     quantity: int = None) -> List[Dict[str, Any]]:
    """
    AI-powered supplier matching using SBERT
    
    Args:
        requirements: Supplier requirements description
        top_k: Number of suppliers to return
        budget: Budget constraint
        quantity: Quantity needed
        
    Returns:
        Ranked suppliers with match scores
    """
    from modules.supplier_sourcing.semantic_supplier_matcher import SemanticSupplierMatcher
    from modules.supplier_sourcing.recommender import SupplierRecommender
    
    recommender = SupplierRecommender(use_ai=True)
    
    if recommender.semantic_matcher:
        suppliers = recommender.semantic_matcher.find_best_suppliers(
            requirements, top_k, budget, quantity
        )
        logger.info(f"AI found {len(suppliers)} suppliers matching '{requirements}'")
        return suppliers
    else:
        logger.warning("SBERT not available for supplier matching")
        return []


# ============================================================================
# MODULE 5: PRICING (XGBOOST)
# ============================================================================

def ai_predict_price(product_cost: float, category: str = 'general',
                    platform: str = 'amazon', quantity: int = 100,
                    market_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    AI-powered price prediction using XGBoost
    
    Args:
        product_cost: Product cost
        category: Product category
        platform: Selling platform
        quantity: Quantity
        market_data: Market statistics (optional)
        
    Returns:
        Price prediction with metrics
    """
    from modules.pricing_estimator.xgboost_pricing import XGBoostPricingModel
    
    model = XGBoostPricingModel()
    
    # Train if not trained
    if not model.is_trained:
        logger.info("Training XGBoost model with synthetic data...")
        training_data = model.generate_synthetic_training_data(100)
        model.train_model(training_data)
    
    product_data = {
        'cost': product_cost,
        'quantity': quantity,
        'target_margin': 0.30,
        'category': category,
        'platform': platform
    }
    
    prediction = model.predict_price(product_data, market_data)
    logger.info(f"AI predicted price: ${prediction['predicted_price']:.2f} (ROI: {prediction['roi_percent']:.1f}%)")
    return prediction


# ============================================================================
# MODULE 6: PLATFORM RECOMMENDATION (RANDOM FOREST)
# ============================================================================

def ai_recommend_platform(product_category: str, price: float,
                         target_market: str = 'international',
                         quantity: int = 100, budget: float = 1000) -> List[Dict[str, Any]]:
    """
    AI-powered platform recommendation using Random Forest
    
    Args:
        product_category: Product category
        price: Product price
        target_market: Target market
        quantity: Quantity to sell
        budget: Available budget
        
    Returns:
        Platform recommendations with probabilities
    """
    from modules.platform_recommender.random_forest_recommender import RandomForestPlatformRecommender
    
    recommender = RandomForestPlatformRecommender()
    
    # Train if not trained
    if not recommender.is_trained:
        logger.info("Training Random Forest model...")
        training_data = recommender.generate_synthetic_training_data(200)
        recommender.train_model(training_data)
    
    product_info = {
        'category': product_category,
        'price': price,
        'target_market': target_market,
        'quantity': quantity,
        'budget': budget
    }
    
    platforms = recommender.predict_platforms(product_info)
    logger.info(f"AI recommends: {platforms[0]['platform']} ({platforms[0]['probability']:.1%})")
    return platforms


# ============================================================================
# MODULE 7: AUDIENCE PROFILING (K-MEANS)
# ============================================================================

def ai_segment_audience(customer_data: pd.DataFrame, n_segments: int = 3) -> Dict[int, Dict[str, Any]]:
    """
    AI-powered audience segmentation using K-Means
    
    Args:
        customer_data: Customer data DataFrame
        n_segments: Number of segments to create
        
    Returns:
        Customer segments with personas
    """
    from modules.audience_recommender.kmeans_profiler import KMeansAudienceProfiler
    
    profiler = KMeansAudienceProfiler(n_clusters=n_segments)
    segments = profiler.segment_audience(customer_data)
    
    logger.info(f"AI created {len(segments)} customer segments")
    return segments


def ai_generate_customer_data(n_samples: int = 300) -> pd.DataFrame:
    """Generate synthetic customer data for testing"""
    from modules.audience_recommender.kmeans_profiler import KMeansAudienceProfiler
    
    profiler = KMeansAudienceProfiler()
    data = profiler.generate_synthetic_customer_data(n_samples)
    logger.info(f"Generated {len(data)} synthetic customer records")
    return data


# ============================================================================
# MODULE 8: MARKETING STRATEGY (AI TEMPLATES + LLM)
# ============================================================================

def ai_generate_marketing_strategy(product_name: str, category: str, price: float,
                                   target_audience: Dict[str, Any], budget: float,
                                   use_llm: bool = False, api_key: str = None) -> Dict[str, Any]:
    """
    AI-powered marketing strategy generation
    
    Args:
        product_name: Product name
        category: Product category
        price: Product price
        target_audience: Target audience profile
        budget: Marketing budget
        use_llm: Use GPT (requires API key)
        api_key: OpenAI API key (optional)
        
    Returns:
        Complete marketing strategy
    """
    from modules.marketing_strategy.ai_strategy_generator import AIMarketingStrategyGenerator
    
    generator = AIMarketingStrategyGenerator(use_llm=use_llm, api_key=api_key)
    
    product_info = {
        'product_name': product_name,
        'category': category,
        'price': price
    }
    
    strategy = generator.generate_strategy(product_info, target_audience, budget)
    logger.info(f"AI generated strategy using {strategy['method']}")
    return strategy


# ============================================================================
# MODULE 9: CATALOG CLEANER (SBERT + Fuzzy Matching)
# ============================================================================

def ai_clean_catalog(products: List[Dict[str, Any]], 
                     remove_duplicates: bool = True,
                     normalize_titles: bool = True,
                     fix_attributes: bool = True,
                     standardize_prices: bool = True,
                     similarity_threshold: float = 0.85) -> Dict[str, Any]:
    """
    AI-powered catalog cleaning and normalization
    
    Args:
        products: List of product dictionaries
        remove_duplicates: Remove duplicate products
        normalize_titles: Clean and standardize titles
        fix_attributes: Fix colors, sizes, materials
        standardize_prices: Convert prices to USD
        similarity_threshold: Duplicate detection threshold (0-1)
        
    Returns:
        Dict with cleaned products and statistics
    """
    from modules.catalog_cleaner import CatalogCleaner
    
    cleaner = CatalogCleaner(use_ai=True)
    result = cleaner.clean_catalog(
        products,
        remove_duplicates=remove_duplicates,
        normalize_titles=normalize_titles,
        fix_attributes=fix_attributes,
        standardize_prices=standardize_prices,
        similarity_threshold=similarity_threshold
    )
    logger.info(f"AI cleaned catalog: {result['products_before']} -> {result['products_after']} products")
    return result


def ai_get_cleaning_recommendations(products: List[Dict[str, Any]]) -> List[str]:
    """Get AI recommendations for cleaning catalog"""
    from modules.catalog_cleaner import CatalogCleaner
    
    cleaner = CatalogCleaner(use_ai=True)
    recommendations = cleaner.get_cleaning_recommendations(products)
    logger.info(f"Generated {len(recommendations)} cleaning recommendations")
    return recommendations


# ============================================================================
# COMBINED AI ANALYSIS
# ============================================================================

def ai_complete_analysis(product_name: str, product_cost: float, category: str,
                        target_market: str = 'international', quantity: int = 100) -> Dict[str, Any]:
    """
    Run complete AI analysis across all 9 modules
    
    Returns:
        Comprehensive AI analysis results
    """
    logger.info(f"Running complete AI analysis for '{product_name}'...")
    
    results = {
        'product': product_name,
        'category': category,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        # Module 1: Scrape similar products
        products = ai_scrape_products(product_name, max_results=20)
        results['similar_products'] = len(products)
        
        # Module 5: AI Pricing
        pricing = ai_predict_price(product_cost, category, quantity=quantity)
        results['ai_pricing'] = pricing
        
        # Module 6: Platform recommendation
        platforms = ai_recommend_platform(category, pricing['predicted_price'], target_market, quantity)
        results['recommended_platforms'] = platforms[:3]
        
        # Module 7: Audience segmentation (with synthetic data)
        customer_data = ai_generate_customer_data(200)
        segments = ai_segment_audience(customer_data, 3)
        results['audience_segments'] = len(segments)
        
        # Module 8: Marketing strategy
        audience = {'primary_demographics': {'age_range': '25-45'}}
        strategy = ai_generate_marketing_strategy(product_name, category, 
                                                 pricing['predicted_price'], 
                                                 audience, budget=5000)
        results['marketing_strategy'] = strategy['method']
        
        logger.info("✅ Complete AI analysis finished successfully")
        
    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        results['error'] = str(e)
    
    return results


# ============================================================================
# QUICK ACCESS FUNCTIONS
# ============================================================================

# Convenience exports
scrape = ai_scrape_products
forecast = ai_forecast_trend
sentiment = ai_analyze_sentiment
suppliers = ai_find_suppliers
pricing = ai_predict_price
platforms = ai_recommend_platform
audience = ai_segment_audience
marketing = ai_generate_marketing_strategy
cleaner = ai_clean_catalog
analyze = ai_complete_analysis
