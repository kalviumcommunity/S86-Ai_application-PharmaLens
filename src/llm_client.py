from __future__ import annotations

import logging
from pathlib import Path

from openai import (
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from src.config import load_settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "llm_first_completion.log"


def setup_logging() -> None:
    """
    Configure logging to both the terminal and an output file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if the function is called more than once.
    if logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        OUTPUT_FILE,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def create_client(settings: dict[str, str]) -> OpenAI:
    """
    Create an OpenAI-compatible client using environment configuration.
    """
    return OpenAI(
        base_url=settings["openai_base_url"],
        api_key=settings["openai_api_key"],
    )


def run_first_completion() -> None:
    """
    Send the first chat completion request and print the response.
    """
    settings = load_settings()

    client = create_client(settings)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise pharmaceutical research assistant. "
                "Answer clearly and do not invent facts."
            ),
        },
        {
            "role": "user",
            "content": (
                "Explain what a clinical trial is in one simple sentence."
            ),
        },
    ]

    logging.info("========== PHARMALENS LLM FIRST COMPLETION ==========")
    logging.info("MODEL: %s", settings["chat_model"])
    logging.info("BASE URL: %s", settings["openai_base_url"])
    logging.info("REQUEST MESSAGES: %s", messages)

    try:
        response = client.chat.completions.create(
            model=settings["chat_model"],
            messages=messages,
        )

        answer = response.choices[0].message.content

        logging.info("RESPONSE: %s", answer)

        if response.usage:
            logging.info("TOKEN USAGE: %s", response.usage)
        else:
            logging.info("TOKEN USAGE: Not available")

        print("\nAssistant:")
        print(answer)

        print(f"\nSample output saved to: {OUTPUT_FILE}")

    except AuthenticationError:
        logging.error(
            "Authentication failed. The API key may be invalid or missing."
        )
        print(
            "\nAuth failed (401): "
            "Check OPENAI_API_KEY in your .env file."
        )

    except RateLimitError:
        logging.error(
            "Rate limit or quota exceeded."
        )
        print(
            "\nRate limited (429): "
            "Slow down and retry. Check your API quota if necessary."
        )

    except Exception as error:
        logging.exception("Unexpected API error: %s", error)
        print(
            "\nUnexpected error occurred. "
            "Check the log file for details."
        )


if __name__ == "__main__":
    setup_logging()
    run_first_completion()