from __future__ import annotations

from src.config import load_settings


def main() -> None:
    settings = load_settings()
    print("PharmaLens workspace is ready.")
    print(f"Chat model: {settings['chat_model'] or '(not set)'}")
    print(f"Embedding model: {settings['embed_model'] or '(not set)'}")


if __name__ == "__main__":
    main()
