# src/config.py

import os 
from pathlib import Path 
from dotenv import load_dotenv

# Load .env file (where your API key lives)
load_dotenv()

# paths 
BASE_DIR = Path(__file__).parent.parent
DATA_DIR=BASE_DIR/"data"
RAW_DATA_DIR=DATA_DIR/"raw"
PROCESSED_DATA_DIR=DATA_DIR/"processed"
CHROMA_DIR = DATA_DIR/"chromadb"

#Your CSV files 
TRANSCRIPTS_CSV = RAW_DATA_DIR/"transcripts_df.csv"
TRANSCRIPT_EMBEDDINGS_CSV=PROCESSED_DATA_DIR/"transcripts_with_embeddings.csv"

#API keys
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

#model settings
EMBEDDING_MODEL ="text-embedding-3-small"
LLM_MODEL="gpt-4o-mini"
EMBEDDING_DIMENSIONS=1536

# -----------------------------------------------------------
# CHROMADB SETTINGS
# -----------------------------------------------------------
COLLECTION_NAME = "podcast_episodes"

   
# Search settings
DEFAULT_TOP_K = 5
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# -----------------------------------------------------------
# CHROMADB SETTINGS
# -----------------------------------------------------------
COLLECTION_NAME = "podcast_episodes"