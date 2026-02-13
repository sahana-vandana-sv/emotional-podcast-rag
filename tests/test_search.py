# tests/test_search.py
   
import sys
sys.path.append('..')

from src.search import get_embedding, semantic_search
from src.data_loader import load_episodes

def test_get_embedding():
    """Test embedding generation"""
    embedding = get_embedding("test query")
    assert len(embedding) == 1536
    assert isinstance(embedding[0], float)
    print("✅ Embedding test passed")

def test_semantic_search():
    """Test search functionality"""
    df = load_episodes()
    results = semantic_search("I feel anxious", df, top_k=3)
    
    assert len(results) == 3
    assert 'similarity' in results[0]
    assert 'metadata' in results[0]
    print("✅ Search test passed")

if __name__ == "__main__":
    test_get_embedding()
    test_semantic_search()
    print("\n🎉 All tests passed!")