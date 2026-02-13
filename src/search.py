# gets the query embeddings and searches the embeddings 

from src.get_embeddings import get_embedding
from src.config import DEFAULT_TOP_K

def semantic_search(query: str, collection, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    query_embedding=get_embedding(query)

    raw       = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    distances = raw['distances'][0]
    min_d     = min(distances) if distances else 0
    max_d     = max(distances) if distances else 1

    results = []
    for i, episode_id in enumerate(raw['ids'][0]):
        d = distances[i]

        if max_d - min_d > 0:
            similarity = 1 - ((d - min_d) / (max_d - min_d))
        else:
            similarity = 1.0

        results.append({
            'episode_id': episode_id,
            'similarity': round(similarity, 4),
            'metadata':   raw['metadatas'][0][i],
            'preview':    raw['documents'][0][i][:300] + "...",
        })

    return results





