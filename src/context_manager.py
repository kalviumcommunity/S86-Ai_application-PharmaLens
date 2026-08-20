from __future__ import annotations

from pathlib import Path

import tiktoken

from src.config import load_settings
from src.llm_client import create_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "context_history.log"


# Deliberately small budget for demonstration.
# A real production model would use a much larger model-specific limit.
HISTORY_TOKEN_BUDGET = 250


SYSTEM_PROMPT = (
    "You are PharmaLens, a concise pharmaceutical research assistant. "
    "Answer clearly and do not invent facts. "
    "Use the conversation context when answering follow-up questions."
)


def get_tokenizer():
    """Return the tokenizer used for token measurement."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in a single text string."""
    return len(tokenizer.encode(text))


def total_tokens(
    messages: list[dict[str, str]],
    tokenizer,
) -> int:
    """
    Count the total tokens in the current message history.
    """
    return sum(
        count_tokens(message["content"], tokenizer)
        for message in messages
    )


def trim_history(
    messages: list[dict[str, str]],
    tokenizer,
    budget: int,
) -> list[dict[str, str]]:
    """
    Remove the oldest conversation messages while the history
    exceeds the token budget.

    The system message is always preserved.
    """

    removed_messages = 0

    while (
        total_tokens(messages, tokenizer) > budget
        and len(messages) > 1
    ):
        # Always remove the oldest non-system message.
        messages.pop(1)
        removed_messages += 1

    return messages


def ask_with_history(
    client,
    model: str,
    history: list[dict[str, str]],
    user_message: str,
    tokenizer,
    budget: int,
) -> tuple[str, int, int, int]:
    """
    Add a user message, measure the history, trim if necessary,
    call the model, and add the assistant response.
    """

    history.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    before_trim = total_tokens(
        history,
        tokenizer,
    )

    old_length = len(history)

    trim_history(
        history,
        tokenizer,
        budget,
    )

    after_trim = total_tokens(
        history,
        tokenizer,
    )

    removed_messages = old_length - len(history)

    print("\n" + "=" * 70)
    print(f"USER: {user_message}")
    print(f"Tokens before trim : {before_trim}")
    print(f"Messages removed   : {removed_messages}")
    print(f"Tokens after trim  : {after_trim}")
    print(f"Token budget       : {budget}")

    if removed_messages:
        print("ACTION              : Trimmed oldest messages")
    else:
        print("ACTION              : No trimming required")

    response = client.chat.completions.create(
        model=model,
        messages=history,
    )

    answer = response.choices[0].message.content or ""

    history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    final_tokens = total_tokens(
        history,
        tokenizer,
    )

    print(f"Assistant: {answer}")
    print(f"History tokens after response: {final_tokens}")

    return (
        answer,
        before_trim,
        after_trim,
        final_tokens,
    )


def main() -> None:
    settings = load_settings()

    client = create_client(settings)

    model = settings["chat_model"]

    tokenizer = get_tokenizer()

    history: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    conversation = [
        (
            "What is a clinical trial? "
            "Explain its basic purpose in simple terms."
        ),
        (
            "What are the main goals researchers usually evaluate "
            "during a clinical trial?"
        ),
        (
            "Why is participant safety monitored throughout the study?"
        ),
        (
            "What is the difference between safety and effectiveness "
            "when evaluating a treatment?"
        ),
        (
            "Why do researchers compare a treatment with a control group?"
        ),
        (
            "What kind of information can researchers collect from "
            "participants during a clinical study?"
        ),
        (
            "Why are adverse events recorded during clinical trials?"
        ),
        (
            "How can researchers use clinical trial results to support "
            "future research?"
        ),
        (
            "Why is it important to document the methods used in a study?"
        ),
        (
            "How does a clinical trial contribute to evidence-based "
            "decision making?"
        ),
    ]

    print("\n========== PHARMALENS CONTEXT HISTORY DEMO ==========")
    print(f"Model: {model}")
    print(f"History token budget: {HISTORY_TOKEN_BUDGET}")

    results = []

    for turn_number, user_message in enumerate(
        conversation,
        start=1,
    ):
        print(f"\n\nTURN {turn_number}")

        answer, before_trim, after_trim, final_tokens = (
            ask_with_history(
                client=client,
                model=model,
                history=history,
                user_message=user_message,
                tokenizer=tokenizer,
                budget=HISTORY_TOKEN_BUDGET,
            )
        )

        results.append(
            {
                "turn": turn_number,
                "user": user_message,
                "answer": answer,
                "before_trim": before_trim,
                "after_trim": after_trim,
                "final_tokens": final_tokens,
            }
        )

    # ---------------------------------------------------------
    # Save sample run for Task 5
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PHARMALENS - CONTEXT WINDOWS & MESSAGE HISTORY\n"
        )

        file.write("=" * 70 + "\n\n")

        file.write(
            f"MODEL: {model}\n"
        )

        file.write(
            f"TOKEN BUDGET: {HISTORY_TOKEN_BUDGET}\n\n"
        )

        file.write(
            "SYSTEM PROMPT:\n"
        )

        file.write(
            SYSTEM_PROMPT + "\n\n"
        )

        file.write(
            "TURN-BY-TURN RESULTS\n"
        )

        file.write(
            "-" * 70 + "\n\n"
        )

        for result in results:

            file.write(
                f"TURN {result['turn']}\n"
            )

            file.write(
                f"USER: {result['user']}\n"
            )

            file.write(
                f"TOKENS BEFORE TRIM: "
                f"{result['before_trim']}\n"
            )

            file.write(
                f"TOKENS AFTER TRIM: "
                f"{result['after_trim']}\n"
            )

            file.write(
                f"TOKENS AFTER RESPONSE: "
                f"{result['final_tokens']}\n"
            )

            file.write(
                f"ASSISTANT: {result['answer']}\n\n"
            )

    print(
        f"\nSample run saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()