import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.llm_integeration import interpret_emotional_query, generate_explanation


def main():
    print("\n--- Testing interpret_emotional_query ---")
    emotional_context = interpret_emotional_query(
        user_query="I feel lonely and stuck, like I don't know what direction to take."
    )
    print(emotional_context)

    print("\n--- Testing generate_explanation ---")
    episode_result = {
        "metadata": {
            "episode_title": "Finding Your Way When You Feel Lost",
            "show_name": "The Mindful Life"
        },
        "preview": "A conversation about uncertainty, self-trust, and navigating life transitions."
    }

    explanation = generate_explanation(
        user_query="I feel lonely and stuck, like I don't know what direction to take.",
        episode_result=episode_result,
        emotional_context=emotional_context
    )
    print(explanation)


if __name__ == "__main__":
    main()