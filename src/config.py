from __future__ import annotations

import os

from dotenv import load_dotenv


def load_settings() -> dict[str, str]:
    load_dotenv()
    return {
        "openai_base_url": os.getenv("OPENAI_BASE_URL", "").strip(),
        "openai_api_key": os.getenv("OPENAI_API_KEY", "").strip(),
        "chat_model": os.getenv("CHAT_MODEL", "").strip(),
        "embed_model": os.getenv("EMBED_MODEL", "").strip(),
    }
