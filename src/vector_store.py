import chromadb
import pandas as pd
from src.config import CHROMA_DIR,COLLECTION_NAME

def _get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True,exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

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

    # prepare and insert data into chromadb 

    ids, embeddings,documents, metadatas = [],[],[],[]
    for idx, row in df.iterrows():
        episode_id = f"{idx + 1:03d}"
        ids.append(episode_id)
        embeddings.append(row['embedding'])
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

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"✅ Added {collection.count()} episodes to ChromaDB")
    return collection



