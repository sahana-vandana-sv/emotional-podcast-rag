import json
from unittest.mock import patch, MagicMock

from src.llm_integeration import interpret_emotional_query, generate_explanation


@patch("src.llm_integeration.openai_client.chat.completions.create")
def test_interpret_emotional_query(mock_create):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "primary_emotion": "anxiety",
                    "secondary_emotions": ["fear", "stress"],
                    "situation": "The user feels overwhelmed and uncertain.",
                    "underlying_needs": ["validation", "perspective"],
                    "search_keywords": ["anxiety", "overthinking", "stress"]
                })
            )
        )
    ]
    mock_create.return_value = mock_response

    result = interpret_emotional_query(
        user_query="I feel overwhelmed and my mind won't stop racing"
    )

    assert isinstance(result, dict)
    assert result["primary_emotion"] == "anxiety"
    assert "search_keywords" in result
    assert isinstance(result["secondary_emotions"], list)


@patch("src.llm_integeration.openai_client.chat.completions.create")
def test_generate_explanation(mock_create):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="This episode may help you feel less alone and give you practical perspective on what you're going through."
            )
        )
    ]
    mock_create.return_value = mock_response

    episode_result = {
        "metadata": {
            "episode_title": "How to Manage Anxiety",
            "show_name": "The Calm Podcast"
        },
        "preview": "This episode discusses overthinking and how to calm a racing mind."
    }

    emotional_context = {
        "primary_emotion": "anxiety",
        "situation": "The user feels overwhelmed and unable to slow their thoughts.",
        "underlying_needs": ["validation", "actionable advice"]
    }

    result = generate_explanation(
        user_query="I feel overwhelmed and my mind won't stop racing",
        episode_result=episode_result,
        emotional_context=emotional_context
    )

    assert isinstance(result, str)
    assert len(result) > 0