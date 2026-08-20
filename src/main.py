from __future__ import annotations

from src.llm_client import setup_logging, run_first_completion


def main() -> None:
    setup_logging()
    run_first_completion()


if __name__ == "__main__":
    main()