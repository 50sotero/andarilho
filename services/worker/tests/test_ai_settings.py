"""Worker settings for scene analysis: the switch, the AI variables and the production rule."""

from __future__ import annotations

import pytest

from andarilho_worker.settings import Settings

BASE_ENV = {"DATABASE_URL": "postgresql+psycopg://u:p@localhost:5432/db"}


@pytest.mark.parametrize("word", ["1", "true", "TRUE", "Yes", " yes "])
def test_scene_analysis_is_on_for_common_true_words(word):
    settings = Settings.from_env({**BASE_ENV, "AI_SCENE_ANALYSIS": word})
    assert settings.ai_scene_analysis is True


@pytest.mark.parametrize("word", ["", "0", "false", "No", "off"])
def test_scene_analysis_is_off_for_false_words_and_by_default(word):
    assert Settings.from_env({**BASE_ENV, "AI_SCENE_ANALYSIS": word}).ai_scene_analysis is False
    assert Settings.from_env(BASE_ENV).ai_scene_analysis is False


def test_an_unrecognised_switch_value_is_refused_by_name():
    with pytest.raises(ValueError, match="AI_SCENE_ANALYSIS"):
        Settings.from_env({**BASE_ENV, "AI_SCENE_ANALYSIS": "maybe"})


def test_ai_settings_have_defaults_and_read_their_variables(tmp_path):
    defaults = Settings.from_env(BASE_ENV)
    assert defaults.ai_base_url == "https://opencode.ai/inference/go/v1"  # OpenCode Go
    assert defaults.ai_model == "deepseek-v4.1-flash"
    assert defaults.opencode_api_key_file is None
    assert defaults.ai_timeout_seconds == 180.0
    assert defaults.ai_max_output_tokens == 8192
    key = tmp_path / "key"
    settings = Settings.from_env(
        {
            **BASE_ENV,
            "AI_BASE_URL": "http://gateway.test/v1",
            "AI_MODEL": "model-x",
            "OPENCODE_API_KEY_FILE": str(key),
            "AI_SCENE_TIMEOUT_SECONDS": "45.5",
            "AI_SCENE_MAX_OUTPUT_TOKENS": "12000",
        }
    )
    assert settings.ai_base_url == "http://gateway.test/v1"
    assert settings.ai_model == "model-x"
    assert settings.opencode_api_key_file == str(key)
    assert settings.ai_timeout_seconds == 45.5
    assert settings.ai_max_output_tokens == 12000


def test_blank_ai_variables_fall_back_to_their_defaults():
    settings = Settings.from_env(
        {
            **BASE_ENV,
            "AI_BASE_URL": "  ",
            "AI_MODEL": "",
            "OPENCODE_API_KEY_FILE": "   ",
            "AI_SCENE_TIMEOUT_SECONDS": "",
        }
    )
    assert settings.ai_base_url == "https://opencode.ai/inference/go/v1"
    assert settings.ai_model == "deepseek-v4.1-flash"
    assert settings.opencode_api_key_file is None
    assert settings.ai_timeout_seconds == 180.0


def test_the_api_ai_variables_do_not_reach_the_scene_analysis():
    """The API's short AI advice has its own timeout and reply budget in the same .env."""
    settings = Settings.from_env(
        {**BASE_ENV, "AI_TIMEOUT_SECONDS": "60", "AI_MAX_OUTPUT_TOKENS": "2048"}
    )
    assert settings.ai_timeout_seconds == 180.0
    assert settings.ai_max_output_tokens == 8192


@pytest.mark.parametrize("value", ["0", "-3"])
def test_the_reply_budget_must_be_positive(value):
    with pytest.raises(ValueError, match="AI_SCENE_MAX_OUTPUT_TOKENS"):
        Settings.from_env({**BASE_ENV, "AI_SCENE_MAX_OUTPUT_TOKENS": value})


@pytest.mark.parametrize("value", ["0", "-3", "nan", "inf", "soon"])
def test_the_ai_timeout_must_be_a_positive_number(value):
    with pytest.raises(ValueError, match="AI_SCENE_TIMEOUT_SECONDS"):
        Settings.from_env({**BASE_ENV, "AI_SCENE_TIMEOUT_SECONDS": value})


def test_production_requires_https_for_the_model_when_scene_analysis_is_on():
    production = {
        **BASE_ENV,
        "APP_ENV": "production",
        "AI_SCENE_ANALYSIS": "true",
        "AI_BASE_URL": "http://gateway.internal/v1",
    }
    with pytest.raises(ValueError, match="AI_BASE_URL must use https"):
        Settings.from_env(production)
    secure = {**production, "AI_BASE_URL": "https://gateway.example/v1"}
    assert Settings.from_env(secure).ai_base_url == "https://gateway.example/v1"


def test_production_allows_a_plain_model_url_while_scene_analysis_is_off():
    env = {
        **BASE_ENV,
        "APP_ENV": "production",
        "AI_SCENE_ANALYSIS": "false",
        "AI_BASE_URL": "http://gateway.internal/v1",
    }
    assert Settings.from_env(env).ai_scene_analysis is False
