import chromadb
import pandas as pd
from src.config import CHROMA_DIR,COLLECTION_NAME

def _get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True,exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def _prepare_episode_data(df: pd.DataFrame, start_idx: int = 0, episode_ids: set = None):

    ids, embeddings, documents, metadatas = [], [], [], []
    skipped_no_embedding = 0
    skipped_already_exists = 0
    
    for idx, row in df.iterrows():
        episode_id = f"{idx + 1:03d}"
        
        # Skip if already exists (for incremental updates)
        if episode_ids and episode_id in episode_ids:
            skipped_already_exists += 1
            continue
        
        # Skip if no embedding
        emb = row.get('embedding')
        if emb is None or (isinstance(emb, float) and pd.isna(emb)):
            print(f"⚠️  Skipping row {idx + 1} — no embedding")
            emb = row.get('embedding')
            continue
        # Skip if embedding is empty list
        if isinstance(emb, list) and len(emb) == 0:
            print(f"⚠️  Row {idx + 1} has empty embedding — skipping")
            skipped_no_embedding += 1
            continue
        
        ids.append(episode_id)
        embeddings.append(emb)
        documents.append(str(row['transcript_clean'])[:1000])
        metadatas.append({
            'episode_id':    episode_id,
            'show_name':     str(row['youtube_channel']),
            'episode_title': str(row['youtube_title']),
            'url':           str(row['url']),
            'duration_mins': float(row['duration_mins']),
            'word_count':    int(row['word_count']),
            'video_id':      str(row['video_id']),
        })
    
    return ids, embeddings, documents, metadatas,skipped_no_embedding

def _add_episodes_to_collection(collection, ids, embeddings, documents, metadatas):
    """
    Add episodes to ChromaDB collection.
    Shared logic for both build and update operations.
    """
    if not ids:
        return 0
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    
    return len(ids)


def get_collection():
    client = _get_chroma_client()
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✓ Loaded '{COLLECTION_NAME}' ({collection.count()} episodes)")
        return collection
    except Exception:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found.\n"
            "Run build_collection(df) first."
        )


def build_collection(df:pd.DataFrame,force_rebuild:bool = False):
    client = _get_chroma_client()

    existing= None
    try:
        existing= client.get_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    if existing and not force_rebuild:
        print(f"✓ Collection already exists ({existing.count()} episodes).")
        print("  Pass force_rebuild=True to rebuild from scratch.")
        return existing
    
    if existing and force_rebuild:
        #delete the old collection 
        client.delete_collection(name=COLLECTION_NAME)
        print("deleted old collection")

    collection =client.create_collection(
        name = COLLECTION_NAME,
        metadata={
            "description": "Emotional support podcast episodes",
            "hnsw:space": "cosine"
        }
    )

    print(f"created collection: {COLLECTION_NAME}")

    # Prepare and add data (reused logic)
    ids, embeddings, documents, metadatas,skipped = _prepare_episode_data(df)
    
    if not ids:
        raise ValueError("No valid episodes with embeddings found in DataFrame")
    
    added_count = _add_episodes_to_collection(collection, ids, embeddings, documents, metadatas)
    
    print(f"✅ Added {added_count} episodes to ChromaDB")
    print(f"✓ Collection now has {collection.count()} episodes")
    
    return collection

# src/vector_store.py
# Replace ONLY the update_collection function:

def update_collection(df: pd.DataFrame, collection=None):
    """
    Add ONLY NEW episodes to existing collection.
    """
    if collection is None:
        try:
            collection = get_collection()
        except RuntimeError as e:
            print(f"❌ Error: {e}")
            return None
    
    # Get existing IDs
    try:
        result = collection.get()
        existing_ids = set(result.get('ids', []))
        print(f"Collection currently has {len(existing_ids)} episodes")
    except Exception as e:
        print(f"❌ Error getting collection data: {e}")
        return collection
    
    # Prepare new data
    try:
        ids, embeddings, documents, metadatas, skipped = _prepare_episode_data(
            df, 
            existing_ids=existing_ids
        )
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        raise
    
    if not ids:
        print("✓ No new episodes to add")
        return collection
    
    print(f"Found {len(ids)} new episodes to add...")
    
    # Add to collection — THIS IS WHERE IT'S FAILING
    try:
        print(f"DEBUG: About to add {len(ids)} episodes to ChromaDB...")
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        
        print(f"✅ Added {len(ids)} episodes")
        print(f"✓ Collection now has {collection.count()} episodes")
        
    except Exception as e:
        print(f"❌ ERROR during collection.add(): {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Print first few IDs that failed
        print(f"   First 5 IDs attempted: {ids[:5]}")
        
        # This exception is bubbling up to sync_collection()
        raise
    
    return collection
    
   

def sync_collection(df: pd.DataFrame):
    """
    Smart sync: creates collection if missing, updates if exists.
    
    This is the function you want for your regular workflow.
    
    Returns
    -------
    chromadb.Collection
    """
    client = _get_chroma_client()
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✓ Found existing collection ({collection.count()} episodes)")
        # Update with new episodes
        updated = update_collection(df, collection)
        return updated
        
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found — creating new one...")
        return build_collection(df)




