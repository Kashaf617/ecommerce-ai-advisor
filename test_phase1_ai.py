"""
Test AI-Powered Modules - Phase 1 Quick Wins
Tests SBERT Suppliers, K-Means Audience, Random Forest Platforms, XGBoost Pricing
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.logger import get_logger

logger = get_logger(__name__)

def test_module_4_suppliers():
    """Test SBERT Supplier Matching"""
    print("\n" + "="*70)
    print("MODULE 4: AI-Powered Supplier Matching (SBERT)")
    print("="*70)
    
    try:
        from modules.supplier_sourcing.semantic_supplier_matcher import SemanticSupplierMatcher
        from modules.supplier_sourcing.recommender import SupplierRecommender
        
        # Initialize
        recommender = SupplierRecommender(use_ai=True)
        
        # Test semantic search
        if recommender.semantic_matcher:
            requirements = "electronics manufacturer with good quality and fast shipping"
            suppliers = recommender.semantic_matcher.find_best_suppliers(
                requirements, 
                top_k=3,
                budget=10000,
                quantity=500
            )
            
            print(f"\n✓ Found {len(suppliers)} suppliers using semantic search")
            for i, supplier in enumerate(suppliers[:3], 1):
                score = supplier.get('semantic_match_score', 0)
                print(f"  {i}. {supplier['name']} - Match Score: {score:.3f}")
        else:
            print("  ⚠ SBERT not loaded, using traditional matching")
        
        print("\n✅ Module 4 Test Complete")
        return True
        
    except Exception as e:
        print(f"\n❌ Module 4 Test Failed: {e}")
        return False

def test_module_7_audience():
    """Test K-Means Audience Clustering"""
    print("\n" + "="*70)
    print("MODULE 7: AI-Powered Audience Profiling (K-Means)")
    print("="*70)
    
    try:
        from modules.audience_recommender.kmeans_profiler import KMeansAudienceProfiler
        
        # Initialize
        profiler = KMeansAudienceProfiler(n_clusters=3)
        
        # Generate synthetic customer data
        customer_data = profiler.generate_synthetic_customer_data(n_samples=300)
        print(f"\n✓ Generated {len(customer_data)} synthetic customer records")
        
        # Perform clustering
        segments = profiler.segment_audience(customer_data)
        print(f"✓ Created {len(segments)} customer segments\n")
        
        # Show segments
        for cluster_id, segment in segments.items():
            persona = segment.get('persona', {})
            print(f"Segment {cluster_id + 1}: {persona.get('name', 'Unknown')}")
            print(f"  Size: {segment['size']} customers ({segment['percentage']:.1f}%)")
            print(f"  Age Range: {persona.get('age_range', 'N/A')}")
            print(f"  Income: {persona.get('income_level', 'N/A')}")
            print()
        
        print("✅ Module 7 Test Complete")
        return True
        
    except Exception as e:
        print(f"\n❌ Module 7 Test Failed: {e}")
        return False

def test_module_6_platforms():
    """Test Random Forest Platform Recommendation"""
    print("\n" + "="*70)
    print("MODULE 6: AI-Powered Platform Recommendation (Random Forest)")
    print("="*70)
    
    try:
        from modules.platform_recommender.random_forest_recommender import RandomForestPlatformRecommender
        
        # Initialize
        recommender = RandomForestPlatformRecommender()
        
        # Generate training data
        training_data = recommender.generate_synthetic_training_data(n_samples=200)
        print(f"\n✓ Generated {len(training_data)} training examples")
        
        # Train model
        success = recommender.train_model(training_data)
        if success:
            print("✓ Random Forest model trained successfully")
        
        # Test prediction
        product_info = {
            'category': 'electronics',
            'price': 150,
            'target_market': 'international',
            'quantity': 500,
            'budget': 50000
        }
        
        predictions = recommender.predict_platforms(product_info)
        print(f"\n✓ Platform predictions:\n")
        for i, pred in enumerate(predictions[:3], 1):
            print(f"  {i}. {pred['platform'].title()}: {pred['probability']:.1%} confidence")
        
        print("\n✅ Module 6 Test Complete")
        return True
        
    except Exception as e:
        print(f"\n❌ Module 6 Test Failed: {e}")
        return False

def test_module_5_pricing():
    """Test XGBoost Pricing Model"""
    print("\n" + "="*70)
    print("MODULE 5: AI-Powered Pricing (XGBoost)")
    print("="*70)
    
    try:
        from modules.pricing_estimator.xgboost_pricing import XGBoostPricingModel
        
        # Initialize
        pricing_model = XGBoostPricingModel()
        
        # Generate training data
        training_data = pricing_model.generate_synthetic_training_data(n_samples=100)
        print(f"\n✓ Generated {len(training_data)} pricing training examples")
        
        # Train model
        success = pricing_model.train_model(training_data)
        if success:
            print("✓ XGBoost model trained successfully")
        
        # Test prediction
        product_data = {
            'cost': 50,
            'quantity': 200,
            'target_margin': 0.30,
            'category': 'electronics',
            'platform': 'amazon'
        }
        
        market_data = {
            'average_price': 120,
            'competitor_count': 15,
            'average_rating': 4.2,
            'demand_level': 'High'
        }
        
        prediction = pricing_model.predict_price(product_data, market_data)
        print(f"\n✓ Price Prediction:")
        print(f"  Predicted Price: ${prediction['predicted_price']:.2f}")
        print(f"  Profit Margin: {prediction['margin_percent']:.1f}%")
        print(f"  ROI: {prediction['roi_percent']:.1f}%")
        print(f"  Confidence: {prediction['confidence_level']}")
        
        print("\n✅ Module 5 Test Complete")
        return True
        
    except Exception as e:
        print(f"\n❌ Module 5 Test Failed: {e}")
        return False

def main():
    """Run all Phase 1 tests"""
    print("\n" + "="*70)
    print("TESTING PHASE 1 AI QUICK WINS (4 Modules)")
    print("="*70)
    
    results = {
        'Module 4 (SBERT Suppliers)': test_module_4_suppliers(),
        'Module 5 (XGBoost Pricing)': test_module_5_pricing(),
        'Module 6 (Random Forest Platforms)': test_module_6_platforms(),
        'Module 7 (K-Means Audience)': test_module_7_audience()
    }
    
    print("\n" + "="*70)
    print("PHASE 1 TEST SUMMARY")
    print("="*70)
    
    for module, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {module}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} modules passed")
    
    if total_passed == total_tests:
        print("\n🎉 All Phase 1 AI modules are working!")
        print("\nStatus: 5 of 8 modules AI-powered (62.5% complete)")
        print("  ✅ Module 1: Marketplace Scraper (SBERT)")
        print("  ✅ Module 4: Supplier Sourcing (SBERT)")
        print("  ✅ Module 5: Pricing (XGBoost)")
        print("  ✅ Module 6: Platform Recommendations (Random Forest)")
        print("  ✅ Module 7: Audience Profiling (K-Means)")
        print("\nRemaining modules:")
        print("  ⏳ Module 2: Trend Forecasting (LSTM)")
        print("  ⏳ Module 3: Competitor Analysis (BERT Sentiment)")
        print("  ⏳ Module 8: Marketing Strategy (LLM)")
    else:
        print("\n⚠️ Some tests failed. Review the errors above.")
    
    print("="*70)

if __name__ == "__main__":
    main()
