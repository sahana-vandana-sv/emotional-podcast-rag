# tests/test_pipeline.py
   
import sys
sys.path.append('..')

from scripts.rag_pipeline import run_pipeline
from src.data_loader import load_episodes
from src.memory import ConversationMemory
from src.config import COLLECTION_NAME
from src.vector_store import get_collection

# Load data
print("Loading episodes...")
df = load_episodes()
print(f"✓ Loaded {len(df)} episodes")

# Test query
query = "I feel stressed about finding a new job"
print(f"\nQuery: '{query}'\n")

collection = get_collection()

# Run pipeline
recommendations = run_pipeline(query,collection,memory=ConversationMemory(), top_k=2)

# Display results
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['episode']['metadata']['episode_title']}")
    print(f"   {rec['explanation']}\n")

print("✅ Pipeline test successful!")