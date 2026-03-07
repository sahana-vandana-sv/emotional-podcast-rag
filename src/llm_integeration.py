from __future__ import annotations

import json
from openai import OpenAI
from src.config import OPENAI_API_KEY, LLM_MODEL
from src.prompt_loader import load_prompt

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def interpret_emotional_query(
    user_query: str,
    memory_context: str = "",
    prompt_version: str | None = None,
) -> dict:
    prompt_cfg = load_prompt("interpret_emotional_query", version=prompt_version)
    memory_section = f"\n{memory_context}\n" if memory_context else ""
    user_prompt = prompt_cfg["user_template"].format(
        memory_section=memory_section,
        user_query=user_query,
    )

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": prompt_cfg["system_message"]},
            {"role": "user", "content": user_prompt},
        ],
        response_format=prompt_cfg.get("response_format"),
        temperature=prompt_cfg.get("temperature", 0.4),
    )
    return json.loads(response.choices[0].message.content)


def generate_explanation(
    user_query:        str,
    episode_result:    dict,
    emotional_context: dict,
    prompt_version: str | None = None,
) -> str:
    prompt_cfg = load_prompt("generate_explanation", version=prompt_version)

    user_prompt = prompt_cfg["user_template"].format(
        user_query=user_query,
        primary_emotion=emotional_context["primary_emotion"],
        situation=emotional_context["situation"],
        underlying_needs=", ".join(emotional_context["underlying_needs"]),
        episode_title=episode_result["metadata"]["episode_title"],
        show_name=episode_result["metadata"]["show_name"],
        preview=episode_result["preview"],
    )
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
            messages=[
            {"role": "system", "content": prompt_cfg["system_message"]},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=prompt_cfg.get("max_tokens", 150),
        temperature=prompt_cfg.get("temperature", 0.7),)
    return response.choices[0].message.content.strip()


