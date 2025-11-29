"""Clear all Python cache and test fresh import"""
import sys
import os
import shutil

print("Clearing all __pycache__ directories...")
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        cache_path = os.path.join(root, '__pycache__')
        print(f"  Removing: {cache_path}")
        shutil.rmtree(cache_path)

print("\nClearing sys.modules for platform_recommender...")
modules_to_remove = [key for key in sys.modules.keys() if 'platform_recommender' in key or 'random_forest' in key]
for mod in modules_to_remove:
    print(f"  Removing from sys.modules: {mod}")
    del sys.modules[mod]

print("\nNow testing fresh import...")
from modules.platform_recommender.random_forest_recommender import RandomForestPlatformRecommender

print("\nCreating recommender...")
recommender = RandomForestPlatformRecommender()

print(f"Model exists: {recommender.model is not None}")
print(f"Is trained: {recommender.is_trained}")

print("\nGenerating training data...")
data = recommender.generate_synthetic_training_data(50)

print("Training model...")
success = recommender.train_model(data)
print(f"Training success: {success}")

if success:
    print("\nTesting prediction...")
    pred = recommender.predict_platforms({
        'category': 'electronics',
        'price': 200,
        'target_market': 'international',
        'quantity': 300,
        'budget': 30000
    })
    print(f"✅ SUCCESS! Top platform: {pred[0]['platform']} ({pred[0]['probability']:.1%})")
else:
    print("❌ Training failed!")
