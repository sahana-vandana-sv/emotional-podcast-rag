# scripts/test_pipeline.py
   
import sys
sys.path.append('..')

from src.rag_pipeline import run_rag_pipeline
from src.data_loader import load_episodes

# Load data
print("Loading episodes...")
df = load_episodes()
print(f"✓ Loaded {len(df)} episodes")

# Test query
query = "I feel stressed about finding a new job"
print(f"\nQuery: '{query}'\n")

# Run pipeline
recommendations = run_rag_pipeline(query, df, num_recommendations=2)

# Display results
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['episode']['metadata']['episode_title']}")
    print(f"   {rec['explanation']}\n")

print("✅ Pipeline test successful!")