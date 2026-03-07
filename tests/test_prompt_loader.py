from src.prompt_loader import get_active_prompt_version, load_prompt


def test_get_active_prompt_version():
    version = get_active_prompt_version("interpret_emotional_query")
    assert isinstance(version, str)
    assert version.startswith("v")


def test_load_interpret_prompt():
    prompt_cfg = load_prompt("interpret_emotional_query")

    assert isinstance(prompt_cfg, dict)
    assert prompt_cfg["name"] == "interpret_emotional_query"
    assert "system_message" in prompt_cfg
    assert "user_template" in prompt_cfg
    assert "temperature" in prompt_cfg


def test_load_generate_explanation_prompt():
    prompt_cfg = load_prompt("generate_explanation")

    assert isinstance(prompt_cfg, dict)
    assert prompt_cfg["name"] == "generate_explanation"
    assert "system_message" in prompt_cfg
    assert "user_template" in prompt_cfg