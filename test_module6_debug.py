"""Test Module 6 specifically to see error details"""
import traceback

print("="*70)
print("DETAILED MODULE 6 DIAGNOSTIC TEST")
print("="*70)

try:
    print("\n1. Importing...")
    from modules.platform_recommender.random_forest_recommender import RandomForestPlatformRecommender
    print("   ✓ Import successful")
    
    print("\n2. Creating recommender...")
    recommender = RandomForestPlatformRecommender()
    print(f"   ✓ Recommender created")
    print(f"   - Model exists: {recommender.model is not None}")
    print(f"   - Is trained: {recommender.is_trained}")
    if recommender.model:
        print(f"   - Has classes_: {hasattr(recommender.model, 'classes_')}")
    
    print("\n3. Generating training data...")
    data = recommender.generate_synthetic_training_data(100)
    print(f"   ✓ Generated {len(data)} examples")
    
    print("\n4. Training model...")
    success = recommender.train_model(data)
    print(f"   - Training success: {success}")
    if recommender.model:
        print(f"   - Has classes_ after training: {hasattr(recommender.model, 'classes_')}")
        if hasattr(recommender.model, 'classes_'):
            print(f"   - Classes: {recommender.model.classes_}")
    
    if not success:
        raise Exception("Training failed!")
    
    print("\n5. Testing prediction...")
    pred = recommender.predict_platforms({
        'category': 'electronics',
        'price': 200,
        'target_market': 'international',
        'quantity': 300,
        'budget': 30000
    })
    print(f"   ✓ Prediction successful")
    print(f"   - Top platform: {pred[0]['platform']}")
    print(f"   - Probability: {pred[0]['probability']:.1%}")
    
    print("\n" + "="*70)
    print("✅ MODULE 6 TEST PASSED!")
    print("="*70)
    
except Exception as e:
    print("\n" + "="*70)
    print("❌ MODULE 6 TEST FAILED!")
    print("="*70)
    print(f"\nError: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    print("="*70)
