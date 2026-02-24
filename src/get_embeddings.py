import pandas as pd
from openai import OpenAI
from src.config import OPENAI_API_KEY,EMBEDDING_MODEL

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------------------------------------
# Token estimation & text chunking
# -----------------------------------------------------------
def _clean_text(text: str) -> str:
    return str(text).replace("\n", " ").strip()


def _estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 chars."""
    return max(1, len(text) // 4)


def _chunk_text(text: str, max_tokens: int = 6000) -> list[str]:
    """
    Split a single text into chunks that fit within token limit.
    Uses character-based chunking (not semantic).
    """
    text = _clean_text(text)
    
    # If text fits, return as-is
    if _estimate_tokens(text) <= max_tokens:
        return [text]
    
    # Otherwise, chunk by chars
    max_chars = max_tokens * 4
    chunks = []
    
    for i in range(0, len(text), max_chars):
        chunk = text[i:i + max_chars]
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


# -----------------------------------------------------------
# Single embedding (no chunking)
# -----------------------------------------------------------
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    Generate a single embedding for text.
    
    WARNING: If text exceeds 8,191 tokens, this will fail.
    Use get_embedding_safe() for long texts.
    """
    text = _clean_text(text)
    response = openai_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


# -----------------------------------------------------------
# Safe embedding with auto-chunking
# -----------------------------------------------------------
def get_embedding_safe(
    text: str, 
    model: str = EMBEDDING_MODEL,
    max_tokens_per_chunk: int = 6000,
) -> list[float]:
    """
    Generate embedding for text of ANY length.
    
    If text exceeds max_tokens_per_chunk, it's split into chunks,
    embedded separately, then averaged together.
    
    This prevents the "maximum context length exceeded" error.
    """
    text = _clean_text(text)
    
    # Check if chunking is needed
    if _estimate_tokens(text) <= max_tokens_per_chunk:
        return get_embedding(text, model)
    
    # Chunk and embed each piece
    chunks = _chunk_text(text, max_tokens=max_tokens_per_chunk)
    print(f"  ⚠️  Text too long ({_estimate_tokens(text)} tokens) — split into {len(chunks)} chunks")
    
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk, model)
        chunk_embeddings.append(emb)
    
    # Average the chunk embeddings
    dim = len(chunk_embeddings[0])
    avg_embedding = [0.0] * dim
    
    for emb in chunk_embeddings:
        for j in range(dim):
            avg_embedding[j] += emb[j]
    
    return [v / len(chunk_embeddings) for v in avg_embedding]


# -----------------------------------------------------------
# Batch embedding with token-safe chunking
# -----------------------------------------------------------
def get_embeddings_batch(
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    max_tokens_per_chunk: int = 6000,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    
    Each text is processed independently:
    - If text is short: embed directly
    - If text is long: chunk it, embed chunks, average
    
    Returns embeddings in same order as input texts.
    """
    embeddings = []
    
    for i, text in enumerate(texts):
        print(f"[{i+1}/{len(texts)}] Processing...")
        emb = get_embedding_safe(text, model, max_tokens_per_chunk)
        embeddings.append(emb)
    
    return embeddings


# -----------------------------------------------------------
# Add embeddings to DataFrame
# -----------------------------------------------------------
def add_embeddings_to_df(
    df: pd.DataFrame,
    text_col: str = "transcript_clean",
    embedding_col: str = "embedding",
    model: str = EMBEDDING_MODEL,
    max_tokens_per_chunk: int = 6000,
) -> pd.DataFrame:
    """
    Add embeddings to DataFrame for rows where embedding is missing.
    
    Handles long transcripts by chunking automatically.
    
    Parameters
    ----------
    df                   : Input DataFrame
    text_col             : Column containing text to embed
    embedding_col        : Column to store embeddings
    model                : OpenAI embedding model
    max_tokens_per_chunk : Max tokens per text (texts longer than this get chunked)
    
    Returns
    -------
    DataFrame with embeddings filled in
    """
    out = df.copy()
    
    # Ensure columns exist
    if text_col not in out.columns:
        raise KeyError(f"Column not found: {text_col}")
    
    if embedding_col not in out.columns:
        out[embedding_col] = None
    
    # Find rows that need embeddings
    def _is_missing(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return True
        if isinstance(x, str) and x.strip().lower() in {"", "nan", "none"}:
            return True
        if isinstance(x, list) and len(x) == 0:
            return True
        return False
    
    missing_mask = out[embedding_col].apply(_is_missing)
    
    # Only embed rows with non-empty text
    text_series = out[text_col].fillna("").astype(str)
    nonempty_mask = text_series.str.strip().ne("")
    
    to_embed_mask = missing_mask & nonempty_mask
    idxs = out.index[to_embed_mask].tolist()
    
    if not idxs:
        print("✓ All rows already have embeddings")
        return out
    
    print(f"\nGenerating embeddings for {len(idxs)} rows...")
    
    # Generate embeddings
    texts_to_embed = text_series.loc[idxs].tolist()
    embeddings = get_embeddings_batch(texts_to_embed, model, max_tokens_per_chunk)
    
    # Assign back to DataFrame
    for row_idx, emb in zip(idxs, embeddings):
        out.at[row_idx, embedding_col] = emb
    
    print(f"✅ Done. {len(idxs)} embeddings generated.\n")
    
    return out