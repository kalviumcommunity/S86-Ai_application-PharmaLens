from __future__ import annotations

import os

from dotenv import load_dotenv


def load_settings() -> dict[str, str]:
    """
    Load application settings from the .env file.

    Returns:
        A dictionary containing the LLM API configuration.
    """
    load_dotenv()

    settings = {
        "openai_base_url": os.getenv("OPENAI_BASE_URL", "").strip(),
        "openai_api_key": os.getenv("OPENAI_API_KEY", "").strip(),
        "chat_model": os.getenv("CHAT_MODEL", "").strip(),
        "embed_model": os.getenv("EMBED_MODEL", "").strip(),
    }

    required_settings = {
        "OPENAI_BASE_URL": settings["openai_base_url"],
        "OPENAI_API_KEY": settings["openai_api_key"],
        "CHAT_MODEL": settings["chat_model"],
    }

    missing_settings = [
        name
        for name, value in required_settings.items()
        if not value
    ]

    if missing_settings:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing_settings)
        )

    return settings