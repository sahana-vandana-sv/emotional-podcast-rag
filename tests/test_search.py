# tests/test_search.py
   
import sys
sys.path.append('..')

from src.get_embeddings import get_embedding
from src.search import semantic_search
from src.data_loader import load_episodes
from src.vector_store import get_collection



def test_get_embedding():
    """Test embedding generation"""
    embedding = get_embedding("test query")
    assert len(embedding) == 1536
    assert isinstance(embedding[0], float)
    print("✅ Embedding test passed")

def test_semantic_search():
    """Test search functionality"""
    collection = get_collection()
    results = semantic_search("I feel anxious",collection, top_k=3)
    
    assert len(results) == 3
    assert 'similarity' in results[0]
    assert 'metadata' in results[0]
    print("✅ Search test passed")

if __name__ == "__main__":
    test_get_embedding()
    test_semantic_search()
    print("\n🎉 All tests passed!")