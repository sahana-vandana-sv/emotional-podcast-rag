import json
from openai import OpenAI
from src.config import OPENAI_API_KEY, LLM_MODEL

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def interpret_emotional_query(user_query: str, memory_context: str = "") -> dict:
    memory_section = f"\n{memory_context}\n" if memory_context else ""
    prompt = f"""You are an empathetic emotional intelligence assistant.
    {memory_section}
    Analyze the user query below and extract emotional context.
    User query: "{user_query}"
    Respond ONLY in JSON:
    {{
    "primary_emotion":    "one word (e.g. shame, anxiety, anger, grief)",
    "secondary_emotions": ["list of secondary emotions"],
    "situation":          "one sentence describing their situation",
    "underlying_needs":   ["validation", "perspective", "actionable advice"],
    "search_keywords":    ["key concepts to search in podcast transcripts"]
    }}"""

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an empathetic assistant."},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    return json.loads(response.choices[0].message.content)


def generate_explanation(
    user_query:        str,
    episode_result:    dict,
    emotional_context: dict,
) -> str:
    prompt = f"""You are recommending a podcast to someone who is struggling emotionally.
    User's situation:
    - Query: "{user_query}"
    - Primary emotion: {emotional_context['primary_emotion']}
    - Situation: {emotional_context['situation']}
    - Needs: {', '.join(emotional_context['underlying_needs'])}
    
    Episode:
    - Title: {episode_result['metadata']['episode_title']}
    - Show:  {episode_result['metadata']['show_name']}
    - Excerpt: {episode_result['preview']}

    Write 2-3 warm, specific sentences explaining why THIS episode will help THEM.
    Use "you" language. Be concrete, not generic."""

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()
