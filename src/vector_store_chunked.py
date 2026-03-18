# src/vector_store_chunked.py
# -----------------------------------------------------------
# Build ChromaDB collection with timestamp-aware chunks
# -----------------------------------------------------------

import pandas as pd
from src.chunking import chunk_transcript_with_timestamps, parse_raw_segments
from src.embeddings import get_embeddings_batch
from src.vector_store import _get_chroma_client
from src.logging_utils import get_logger

logger = get_logger("vector_store_chunked")


def build_chunked_collection(
    df: pd.DataFrame,
    collection_name: str = "podcast_chunks",
    force_rebuild: bool = False,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
):
    
    logger.info("="*60)
    logger.info("BUILDING CHUNKED COLLECTION")
    logger.info("="*60)
    
    client = _get_chroma_client()
    
    # Delete old collection if force_rebuild
    if force_rebuild:
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            logger.debug(f"No existing collection to delete: {e}")
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    logger.info(f"Created collection: {collection_name}")
    
    # Generate chunks from all episodes
    logger.info(f"Processing {len(df)} episodes...")
    
    all_chunks = []
    
    for idx, row in df.iterrows():
        episode_id = f"{idx + 1:03d}"
        
        logger.info(f"\n[{idx+1}/{len(df)}] Episode {episode_id}: {row['youtube_title'][:50]}...")
        
        # Parse raw_segments
        raw_segments = parse_raw_segments(row['raw_segments'])
        
        if not raw_segments:
            logger.warning(f"Episode {episode_id}: No segments found, skipping")
            continue
        
        # Prepare metadata
        episode_metadata = {
            'episode_id': episode_id,
            'youtube_title': row['youtube_title'],
            'youtube_channel': row['youtube_channel'],
            'url': row['url'],
            'video_id': row.get('video_id', ''),
        }
        
        # Generate chunks
        chunks = chunk_transcript_with_timestamps(
            raw_segments=raw_segments,
            episode_metadata=episode_metadata,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
        
        logger.info(f"  Created {len(chunks)} chunks")
        
        all_chunks.extend(chunks)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Total chunks generated: {len(all_chunks)}")
    logger.info(f"{'='*60}")
    
    # Generate embeddings for chunks
    logger.info("\nGenerating embeddings for chunks...")
    
    chunk_texts = [chunk['text'] for chunk in all_chunks]
    chunk_indices = list(range(len(all_chunks)))
    
    embeddings = get_embeddings_batch(
        texts=chunk_texts,
        row_indices=chunk_indices,
        delay_between_requests=0.5,
    )
    
    # Filter out failed embeddings
    valid_chunks = []
    valid_embeddings = []
    
    for chunk, emb in zip(all_chunks, embeddings):
        if emb is not None:
            valid_chunks.append(chunk)
            valid_embeddings.append(emb)
        else:
            logger.warning(f"Skipping chunk {chunk['chunk_id']} due to embedding failure")
    
    logger.info(f"Successfully embedded {len(valid_chunks)}/{len(all_chunks)} chunks")
    
    # Prepare data for ChromaDB
    ids = [chunk['chunk_id'] for chunk in valid_chunks]
    documents = [chunk['text'] for chunk in valid_chunks]
    
    # Metadata (exclude 'text' field since it's in documents)
    metadatas = []
    for chunk in valid_chunks:
        metadata = {k: v for k, v in chunk.items() if k != 'text'}
        metadatas.append(metadata)
    
    # Add to ChromaDB
    logger.info("\nAdding chunks to ChromaDB...")
    
    collection.add(
        ids=ids,
        embeddings=valid_embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    
    logger.info("="*60)
    logger.info("CHUNKED COLLECTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Total chunks: {len(valid_chunks)}")
    logger.info(f"Average chunks per episode: {len(valid_chunks) / len(df):.1f}")
    logger.info("="*60)
    
    return collection


def get_chunked_collection(collection_name: str = "podcast_chunks"):
    client = _get_chroma_client()
    
    try:
        collection = client.get_collection(name=collection_name)
        logger.info(f"Loaded collection: {collection_name} ({collection.count()} chunks)")
        return collection
    except Exception as e:
        logger.error(f"Collection not found: {collection_name}")
        raise RuntimeError(
            f"Collection '{collection_name}' not found. "
            f"Run build_chunked_collection() first."
        ) from e