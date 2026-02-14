import ast
import pandas as pd
from src.config import TRANSCRIPT_EMBEDDINGS_CSV

def _safe_parse(value):
    if isinstance(value, list):
        try:
            return [float(x) for x in value]
        except Exception:
            return None
    if not isinstance(value, str):
        return None
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
        return None
    except Exception:
        return None

def load_episodes() -> pd.DataFrame:
    if not TRANSCRIPT_EMBEDDINGS_CSV.exists():
        raise FileNotFoundError(
            f"Data file not found :{TRANSCRIPT_EMBEDDINGS_CSV}\n"
            "Run the embedding generation notebook first"
        )

    df = pd.read_csv(TRANSCRIPT_EMBEDDINGS_CSV)

    # Always parse (safe + simplest)
    df["embedding"] = df["embedding"].apply(_safe_parse)

    before = len(df)
    df = df[df["embedding"].notna()].reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"⚠️  Dropped {before - after} rows with unparseable embeddings")

    # sanity
    if len(df):
        print("✓ embedding type:", type(df["embedding"].iloc[0]), type(df["embedding"].iloc[0][0]))

    print(f"✓ Loaded {len(df)} episodes")
    return df
