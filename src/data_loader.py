import ast 
import pandas as pd 
from src.config import TRANSCRIPT_EMBEDDINGS_CSV

def load_episodes()->pd.DataFrame:

    if not TRANSCRIPT_EMBEDDINGS_CSV.exists():
        raise FileNotFoundError(
            f"Data file not found :{TRANSCRIPT_EMBEDDINGS_CSV}\n"
            "Run the embedding generation notebook first"
        )

    df=pd.read_csv(TRANSCRIPT_EMBEDDINGS_CSV)

    # if embeddings stored as strings in CSV → convert back to list
    if df['embedding'].dtype==object:
        df['embedding']=df['embedding'].apply(_safe_parse)
    
    #Drop rows where parsing failed 
    before = len(df)
    df = df[df['embedding'].notna()].reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"⚠️  Dropped {before - after} rows with unparseable embeddings")

    print(f"✓ Loaded {len(df)} episodes")
    return df


def _safe_parse(value):
    if isinstance(value,list):
        return value
    if not isinstance(value,str):
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return None



