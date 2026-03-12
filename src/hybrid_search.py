import numpy as np 
from rank_bm25 import BM25Okapi

from src.embeddings import get_embedding
from src.config import EMBEDDING_MODEL
from src.logging_utils import get_logger

logger = get_logger("hybrid_search")

class HybridSearcher:
    def __init__(self, collection, df):
          self.collection = collection
          self.df = df.copy()

          logger.info("building BM25 index..")

          #tokenize documents 
          self.documents = df['transcript_clean'].fillna("").tolist()
          # Tokenize (simple whitespace split)
          self.tokenized_docs = [doc.lower().split() for doc in self.documents]
          # Build BM25 index
          self.bm25 = BM25Okapi(self.tokenized_docs)

          # Create episode_id mapping
          self.episode_ids = [f"{i+1:03d}" for i in range(len(df))]
        
          logger.info(f"✓ BM25 index built for {len(self.documents)} documents")

    def search(
        self, 
        query: str, 
        top_k: int = 5,
        semantic_weight: float = 0.7,
        return_scores: bool = False,
    ) -> list[dict]:
         
         logger.debug(f"Hybrid search: '{query}' (semantic_weight={semantic_weight})")

        # ──────────────────────────────────────────────────────
        # 1. BM25 Keyword Search
        # ──────────────────────────────────────────────────────
         tokenized_query = query.lower().split()
         bm25_scores = self.bm25.get_scores(tokenized_query)
        
         # Normalize BM25 scores to [0, 1]
         if bm25_scores.max() > 0:
            bm25_scores_norm = bm25_scores / bm25_scores.max()
         else:
            bm25_scores_norm = bm25_scores
        
         logger.debug(f"BM25 top score: {bm25_scores.max():.3f}")

         # ──────────────────────────────────────────────────────
        # 2. Semantic Vector Search
        # ──────────────────────────────────────────────────────
         query_embedding = get_embedding(query, model=EMBEDDING_MODEL)
        
        # Query ChromaDB (get more than top_k for re-ranking)
         chroma_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 3, len(self.documents)),  # Over-fetch for re-ranking
        )
        
        # Extract results
         chroma_ids = chroma_results['ids'][0]
         chroma_distances = chroma_results['distances'][0]

         # Convert distance to similarity (ChromaDB uses cosine distance)
        # Cosine similarity = 1 - cosine distance
         semantic_scores_dict = {}
         for doc_id, distance in zip(chroma_ids, chroma_distances):
            similarity = 1 - distance
            # Extract row index from episode_id (e.g., "001" → 0)
            row_idx = int(doc_id) - 1
            semantic_scores_dict[row_idx] = similarity
        
         logger.debug(f"Semantic top score: {max(semantic_scores_dict.values()):.3f}")

         # ──────────────────────────────────────────────────────
        # 3. Combine Scores
        # ──────────────────────────────────────────────────────
         combined_scores = {}
        
         for idx in range(len(self.documents)):
            # Get scores (default to 0 if not in results)
            bm25_score = bm25_scores_norm[idx]
            semantic_score = semantic_scores_dict.get(idx, 0.0)
            
            # Weighted combination
            combined = (
                semantic_weight * semantic_score +
                (1 - semantic_weight) * bm25_score
            )
            
            combined_scores[idx] = {
                'episode_id': self.episode_ids[idx],
                'combined_score': combined,
                'semantic_score': semantic_score,
                'bm25_score': bm25_score,
            }

            # ──────────────────────────────────────────────────────
        # 4. Sort by Combined Score
        # ──────────────────────────────────────────────────────
         sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
         
         # ──────────────────────────────────────────────────────
        # 5. Format Output
        # ──────────────────────────────────────────────────────
         final_results = []
        
         for idx, scores in sorted_results[:top_k]:
            row = self.df.iloc[idx]
            
            result = {
                'episode_id': scores['episode_id'],
                'similarity': scores['combined_score'],  # For compatibility
                'metadata': {
                    'episode_id': scores['episode_id'],
                    'show_name': row['youtube_channel'],
                    'episode_title': row['youtube_title'],
                    'url': row['url'],
                    'duration_mins': float(row['duration_mins']),
                    'word_count': int(row['word_count']),
                },
                'preview': row['transcript_clean'][:1000],  # Preview
            }
            
            # Optionally include individual scores
            if return_scores:
                result['scores'] = {
                    'combined': scores['combined_score'],
                    'semantic': scores['semantic_score'],
                    'bm25': scores['bm25_score'],
                }
            
            final_results.append(result)
        
         logger.info(
            f"Hybrid search returned {len(final_results)} results "
            f"(top score: {final_results[0]['similarity']:.3f})"
        )
        
         return final_results
    

# ──────────────────────────────────────────────────────
# Convenience function for easy use
# ──────────────────────────────────────────────────────
def hybrid_search(
    query: str,
    collection,
    df,
    top_k: int = 5,
    semantic_weight: float = 0.7,
) -> list[dict]:
    """
    One-shot hybrid search (creates searcher each time).
    
    For repeated searches, create HybridSearcher once and reuse.
    """
    searcher = HybridSearcher(collection, df)
    return searcher.search(query, top_k, semantic_weight)


