from app.config import get_settings


def test_get_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-token")
    monkeypatch.setenv("TELEGRAM_CONTACT_NAME", "Taviz")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("NOTION_API_KEY", "notion-key")
    monkeypatch.setenv("NOTION_DATABASE_ID", "cf61817bf7424e09b0cfb48122716977")
    monkeypatch.setenv("NOTION_TEMPLATE_ID", "b720c6b05ff64f26bab6bdb4e8fe8740")
    monkeypatch.setenv("ALLOWED_CHAT_IDS", "123,456")
    monkeypatch.setenv("DRY_RUN", "true")

    settings = get_settings()

    assert settings.telegram_bot_token == "telegram-token"
    assert settings.telegram_contact_name == "Taviz"
    assert settings.gemini_api_key == "gemini-key"
    assert settings.notion_api_key == "notion-key"
    assert settings.notion_database_id == "cf61817b-f742-4e09-b0cf-b48122716977"
    assert settings.notion_template_id == "b720c6b0-5ff6-4f26-bab6-bdb4e8fe8740"
    assert settings.allowed_chat_ids == {123, 456}
    assert settings.dry_run is True


def test_get_settings_normalizes_notion_ids(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-token")
    monkeypatch.setenv("TELEGRAM_CONTACT_NAME", "Taviz")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("NOTION_API_KEY", "notion-key")
    monkeypatch.setenv("NOTION_DATABASE_ID", "https://www.notion.so/workspace/cf61817bf7424e09b0cfb48122716977")
    monkeypatch.setenv(
        "NOTION_TEMPLATE_ID",
        "New-book-b720c6b05ff64f26bab6bdb4e8fe8740",
    )
    monkeypatch.setenv("ALLOWED_CHAT_IDS", "123")

    settings = get_settings()

    assert settings.notion_database_id == "cf61817b-f742-4e09-b0cf-b48122716977"
    assert settings.notion_template_id == "b720c6b0-5ff6-4f26-bab6-bdb4e8fe8740"


def test_get_settings_rejects_empty_allowed_chat_ids(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-token")
    monkeypatch.setenv("TELEGRAM_CONTACT_NAME", "Taviz")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("NOTION_API_KEY", "notion-key")
    monkeypatch.setenv("NOTION_DATABASE_ID", "cf61817bf7424e09b0cfb48122716977")
    monkeypatch.setenv("NOTION_TEMPLATE_ID", "b720c6b05ff64f26bab6bdb4e8fe8740")
    monkeypatch.setenv("ALLOWED_CHAT_IDS", "   ")

    try:
        get_settings()
    except RuntimeError as exc:
        assert "ALLOWED_CHAT_IDS" in str(exc)
    else:
        raise AssertionError("Expected get_settings() to reject empty ALLOWED_CHAT_IDS")
