from openai import OpenAI
from src.config import OPENAI_API_KEY,EMBEDDING_MODEL

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    text = str(text).replace("\n"," ").strip()
    response = openai_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def get_embeddings_batches(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    texts = [str(t).replace("\n", " ").strip() for t in texts]
    # add a check later for it can only add 50 in batch at one go in openai

    response = openai_client.embeddings.create(input=texts, model=model)

    # Sort by index to guarantee order matches input
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]

