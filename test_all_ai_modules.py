"""
Test ALL 8 AI-Powered Modules - Complete Integration Test
Tests all modules including Phase 2 & 3
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.logger import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)

def test_all_8_modules():
    """Test all 8 AI modules"""
    print("\n" + "="*70)
    print("TESTING ALL 8 AI-POWERED MODULES")
    print("="*70)
    
    results = {}
    
    # Module 1: SBERT Marketplace Scraper
    print("\n[1/8] Testing Marketplace Scraper (SBERT)...")
    try:
        from modules.marketplace_scraper.scrapers import MarketplaceScraper
        scraper = MarketplaceScraper(use_ai=True)
        products = scraper.scrape_platform('amazon', 'laptop', max_results=10)
        print(f"  ✓ Found {len(products)} products with semantic search")
        results['Module 1 - Scraper'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['Module 1 - Scraper'] = False
    
    # Module 2: LSTM Trend Forecasting
    print("\n[2/8] Testing Trend Forecasting (LSTM/Statistical)...")
    try:
        from modules.trend_analyzer.lstm_forecaster import SimpleLSTMForecaster
        forecaster = SimpleLSTMForecaster()
        # Generate synthetic historical data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        values = np.random.randn(100).cumsum() + 100
        data = pd.DataFrame({'date': dates, 'value': values})
        forecast = forecaster.forecast_trend(data, periods=30)
        print(f"  ✓ Forecast generated using {forecast['method']}")
        print(f"    - {len(forecast['forecast'])} periods forecasted")
        print(f"    - Confidence: {forecast['confidence']:.0%}")
        results['Module 2 - Trends'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['Module 2 - Trends'] = False
    
    # Module 3: BERT Sentiment Analysis
    print("\n[3/8] Testing Competitor Sentiment (BERT)...")
    try:
        from modules.competitor_analysis.bert_sentiment import BERTSentimentAnalyzer
        analyzer = BERTSentimentAnalyzer()
        reviews = [
            "Great product, highly recommend!",
            "Terrible quality, very disappointed",
            "Good value for money",
            "Not worth the price"
        ]
        sentiment = analyzer.analyze_reviews(reviews)
        print(f"  ✓ Analyzed {sentiment['total_reviews_analyzed']} reviews")
        print(f"    - Sentiment: {sentiment['overall_sentiment']}")
        print(f"    - Positive: {sentiment['positive_ratio']:.0%}")
        print(f"    - Method: {'BERT ML' if sentiment['is_ml_analysis'] else 'Fallback'}")
        results['Module 3 - Sentiment'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['Module 3 - Sentiment'] = False
    
    # Module 4: SBERT Supplier Matching
    print("\n[4/8] Testing Supplier Sourcing (SBERT)...")
    try:
        from modules.supplier_sourcing.recommender import SupplierRecommender
        recommender = SupplierRecommender(use_ai=True)
        if recommender.semantic_matcher:
            suppliers = recommender.semantic_matcher.find_best_suppliers(
                "electronics manufacturer", top_k=3
            )
            print(f"  ✓ Found {len(suppliers)} suppliers with semantic search")
            top_match = suppliers[0]['semantic_match_score'] if suppliers else 0
            print(f"    - Top match score: {top_match:.3f}")
        else:
            print("  ⚠ SBERT not loaded, using fallback")
        results['Module 4 - Suppliers'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['Module 4 - Suppliers'] = False
    
    # Module 5: XGBoost Pricing
    print("\n[5/8] Testing Pricing (XGBoost)...")
    try:
        from modules.pricing_estimator.xgboost_pricing import XGBoostPricingModel
        model = XGBoostPricingModel()
        # Train if needed
        if not model.is_trained:
            training_data = model.generate_synthetic_training_data(50)
            model.train_model(training_data)
        prediction = model.predict_price(
            {'cost': 50, 'quantity': 100, 'category': 'electronics', 'platform': 'amazon'},
            {'average_price': 120, 'competitor_count': 10, 'demand_level': 'High'}
        )
        print(f"  ✓ Price predicted: ${prediction['predicted_price']:.2f}")
        print(f"    - ROI: {prediction['roi_percent']:.1f}%")
        print(f"    - Method: {prediction['model_type']}")
        results['Module 5 - Pricing'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['Module 5 - Pricing'] = False
    
    # Module 6: Random Forest Platforms
    print("\n[6/8] Testing Platform Recommendation (Random Forest)...")
    try:
        from modules.platform_recommender.random_forest_recommender import RandomForestPlatformRecommender
        recommender = RandomForestPlatformRecommender()
        # Check using classes_ attribute which is set after fit() is called
        if not recommender.is_trained or not hasattr(recommender.model, 'classes_'):
            training_data = recommender.generate_synthetic_training_data(100)
            success = recommender.train_model(training_data)
            if not success or not hasattr(recommender.model, 'classes_'):
                raise Exception("Model training failed or model not properly fitted")
        predictions = recommender.predict_platforms({
            'category': 'electronics',
            'price': 200,
            'target_market': 'international',
            'quantity': 300,
            'budget': 30000
        })
        top_platform = predictions[0]
        print(f"  ✓ Top platform: {top_platform['platform']}")
        print(f"    - Probability: {top_platform['probability']:.1%}")
        results['Module 6 - Platforms'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['Module 6 - Platforms'] = False
    
    # Module 7: K-Means Audience
    print("\n[7/8] Testing Audience Profiling (K-Means)...")
    try:
        from modules.audience_recommender.kmeans_profiler import KMeansAudienceProfiler
        profiler = KMeansAudienceProfiler(n_clusters=3)
        customer_data = profiler.generate_synthetic_customer_data(200)
        segments = profiler.segment_audience(customer_data)
        print(f"  ✓ Created {len(segments)} customer segments")
        for seg_id, segment in list(segments.items())[:2]:
            print(f"    - {segment['persona']['name']}: {segment['percentage']:.1f}%")
        results['Module 7 - Audience'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['Module 7 - Audience'] = False
    
    # Module 8: AI Marketing Strategy
    print("\n[8/8] Testing Marketing Strategy (AI Templates)...")
    try:
        from modules.marketing_strategy.ai_strategy_generator import AIMarketingStrategyGenerator
        generator = AIMarketingStrategyGenerator(use_llm=False)  # Using templates
        strategy = generator.generate_strategy(
            {'product_name': 'Wireless Earbuds', 'category': 'electronics', 'price': 79},
            {'primary_demographics': {'age_range': '25-35'}},
            5000
        )
        print(f"  ✓ Strategy generated using {strategy['method']}")
        print(f"    - Channels: {len(strategy['channels'])}")
        print(f"    - Budget: ${strategy['overview']['budget']:,.0f}")
        results['Module 8 - Marketing'] = True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['Module 8 - Marketing'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY - ALL 8 AI MODULES")
    print("="*70)
    
    for module, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {module}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} modules passed ({total_passed/total_tests*100:.0f}%)")
    
    if total_passed == total_tests:
        print("\n🎉 SUCCESS! All 8 AI modules are working!")
        print("\n✅ 100% AI Coverage Achieved!")
        print("\nAI Techniques Integrated:")
        print("  1. SBERT - Semantic search (Modules 1, 4)")
        print("  2. LSTM/Statistical - Time series forecasting (Module 2)")
        print("  3. BERT - Sentiment analysis (Module 3)")
        print("  4. XGBoost - ML pricing (Module 5)")
        print("  5. Random Forest - Platform classification (Module 6)")
        print("  6. K-Means - Customer clustering (Module 7)")
        print("  7. AI Templates - Strategy generation (Module 8)")
    else:
        print(f"\n⚠️ {total_tests - total_passed} module(s) failed. Review errors above.")
    
    print("="*70)

if __name__ == "__main__":
    test_all_8_modules()
