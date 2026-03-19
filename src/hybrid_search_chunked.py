# src/hybrid_search_chunked.py
# -----------------------------------------------------------
# Hybrid search specifically for chunked collection
# -----------------------------------------------------------

import numpy as np
from rank_bm25 import BM25Okapi
from src.embeddings import get_embedding
from src.config import EMBEDDING_MODEL
from src.logging_utils import get_logger

logger = get_logger("hybrid_search_chunked")


class HybridSearcherChunked:
    """
    Hybrid search for chunked collection.
    
    Builds BM25 index from chunk documents, not full episodes.
    """
    
    def __init__(self, collection):
        """
        Initialize hybrid searcher for chunks.
        
        Parameters
        ----------
        collection : ChromaDB collection (chunked)
        """
        self.collection = collection
        
        logger.info("Building BM25 index from chunks...")
        
        # Get all chunks from ChromaDB
        all_data = collection.get(include=['documents', 'metadatas'])
        
        self.chunk_ids = all_data['ids']
        self.documents = all_data['documents']
        self.metadatas = all_data['metadatas']
        
        # Tokenize documents for BM25
        self.tokenized_docs = [doc.lower().split() for doc in self.documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        logger.info(f"✓ BM25 index built for {len(self.documents)} chunks")
    
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
    ) -> list[dict]:
        """
        Hybrid search over chunks.
        
        Parameters
        ----------
        query            : Search query
        top_k            : Number of results
        semantic_weight  : Weight for semantic (0.7 = 70% semantic, 30% BM25)
        
        Returns
        -------
        List of chunks with combined scores
        """
        
        logger.debug(f"Hybrid search: '{query}' (semantic_weight={semantic_weight})")
        
        # ──────────────────────────────────────────────────────
        # 1. BM25 Keyword Search
        # ──────────────────────────────────────────────────────
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize to [0, 1]
        if bm25_scores.max() > 0:
            bm25_scores_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_scores_norm = bm25_scores
        
        logger.debug(f"BM25 top score: {bm25_scores.max():.3f}")
        
        # ──────────────────────────────────────────────────────
        # 2. Semantic Vector Search
        # ──────────────────────────────────────────────────────
        query_embedding = get_embedding(query, model=EMBEDDING_MODEL)
        
        chroma_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 3, len(self.documents)),
        )
        
        chunk_ids = chroma_results['ids'][0]
        distances = chroma_results['distances'][0]
        
        # Convert distance to similarity
        semantic_scores_dict = {}
        for chunk_id, distance in zip(chunk_ids, distances):
            similarity = 1 - distance
            # Find index in our documents list
            try:
                idx = self.chunk_ids.index(chunk_id)
                semantic_scores_dict[idx] = similarity
            except ValueError:
                continue
        
        logger.debug(f"Semantic top score: {max(semantic_scores_dict.values()):.3f}")
        
        # ──────────────────────────────────────────────────────
        # 3. Combine Scores
        # ──────────────────────────────────────────────────────
        combined_scores = {}
        
        for idx in range(len(self.documents)):
            bm25_score = bm25_scores_norm[idx]
            semantic_score = semantic_scores_dict.get(idx, 0.0)
            
            combined = (
                semantic_weight * semantic_score +
                (1 - semantic_weight) * bm25_score
            )
            
            combined_scores[idx] = {
                'chunk_id': self.chunk_ids[idx],
                'combined_score': combined,
                'semantic_score': semantic_score,
                'bm25_score': bm25_score,
            }
        
        # ──────────────────────────────────────────────────────
        # 4. Sort and Format
        # ──────────────────────────────────────────────────────
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        final_results = []
        
        for idx, scores in sorted_results[:top_k]:
            result = {
                'chunk_id': scores['chunk_id'],
                'episode_id': self.metadatas[idx]['episode_id'],
                'similarity': round(scores['combined_score'], 4),
                'metadata': self.metadatas[idx],
                'preview': self.documents[idx][:300] + "...",
            }
            
            final_results.append(result)
        
        logger.info(
            f"Hybrid search returned {len(final_results)} chunks "
            f"(top score: {final_results[0]['similarity']:.4f})"
        )
        
        return final_results