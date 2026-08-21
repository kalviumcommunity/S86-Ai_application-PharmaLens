from __future__ import annotations

import logging
from pathlib import Path

from src.config import load_settings
from src.llm_client import create_client
from src.prompt_templates import build_rag_messages, render_template


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "prompt_comparison.log"


def ask_model(client, model: str, messages: list[dict[str, str]]) -> str:
    """Send messages to the configured chat model."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return response.choices[0].message.content or ""


def main() -> None:
    # Load the same configuration used by the previous assignment.
    settings = load_settings()

    # Create the OpenAI-compatible client using the existing helper.
    client = create_client(settings)

    model = settings["chat_model"]

    logging.info("========== PHARMALENS PROMPT CONSTRUCTION ==========")
    logging.info("MODEL: %s", model)
    logging.info("BASE URL: %s", settings["openai_base_url"])

    # =========================================================
    # TASK 1 & TASK 2
    # Shared template rendered with runtime values
    # =========================================================

    context_text = (
        "Clinical trials are controlled studies in human participants. "
        "Researchers evaluate treatment safety, effectiveness, and side effects."
    )
    user_question = "What is the purpose of a clinical trial?"

    output_instructions = "Respond in two concise sentences."

    system_prompt = render_template(
        "rag_system",
        assistant_name="PharmaLens",
    )

    user_prompt = render_template(
        "rag_user",
        context=context_text,
        question=user_question,
        output_instructions=output_instructions,
    )

    messages = build_rag_messages(
        context=context_text,
        question=user_question,
        output_instructions=output_instructions,
    )

    logging.info("SYSTEM PROMPT: %s", system_prompt)
    logging.info("USER PROMPT: %s", user_prompt)

    answer = ask_model(client, model, messages)

    print("\n===== SYSTEM + USER ROLE EXAMPLE =====")

    print("\nSystem:")
    print(system_prompt)

    print("\nUser:")
    print(user_prompt)

    print("\nAssistant:")
    print(answer)

    # =========================================================
    # TASK 3
    # Compare two prompt variations
    # =========================================================

    comparison_system = system_prompt

    # ---------------------------------------------------------
    # Variation 1: Vague prompt
    # ---------------------------------------------------------

    vague_question = "Explain clinical trials."

    vague_prompt = render_template(
        "rag_user",
        context="No additional context is available.",
        question=vague_question,
        output_instructions="Respond freely.",
    )

    vague_messages = [
        {
            "role": "system",
            "content": comparison_system,
        },
        {
            "role": "user",
            "content": vague_prompt,
        },
    ]

    logging.info("VAGUE PROMPT: %s", vague_prompt)

    vague_answer = ask_model(
        client,
        model,
        vague_messages,
    )

    # ---------------------------------------------------------
    # Variation 2: Clear and constrained prompt
    # ---------------------------------------------------------

    clear_question = (
        "What is the main purpose of a clinical trial for a "
        "pharmaceutical research associate?"
    )

    clear_prompt = render_template(
        "rag_user",
        context=context_text,
        question=clear_question,
        output_instructions=(
            "In exactly two simple sentences, focus only on evaluating "
            "safety and effectiveness in human participants."
        ),
    )

    clear_messages = [
        {
            "role": "system",
            "content": comparison_system,
        },
        {
            "role": "user",
            "content": clear_prompt,
        },
    ]

    logging.info("CLEAR PROMPT: %s", clear_prompt)

    clear_answer = ask_model(
        client,
        model,
        clear_messages,
    )

    print("\n===== PROMPT COMPARISON =====")

    print("\n--- Variation 1: Vague Prompt ---")

    print("\nInput:")
    print(vague_prompt)

    print("\nOutput:")
    print(vague_answer)

    print("\n--- Variation 2: Clear and Constrained Prompt ---")

    print("\nInput:")
    print(clear_prompt)

    print("\nOutput:")
    print(clear_answer)

    # =========================================================
    # TASK 4
    # Document the chosen prompt
    # =========================================================

    prompt_note = (
        "The clear and constrained prompt is better because it "
        "explicitly defines the task, audience, scope, required "
        "length, and information boundaries. These constraints "
        "reduce ambiguity and make the model response more focused, "
        "consistent, and easier to use in the PharmaLens application."
    )

    print("\n===== CHOSEN PROMPT NOTE =====")
    print(prompt_note)

    # =========================================================
    # TASK 5
    # Save example inputs and outputs
    # =========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PHARMALENS - PROMPT CONSTRUCTION ASSIGNMENT\n"
        )
        file.write("=" * 60 + "\n\n")

        file.write(
            "TASK 1 & TASK 2 - SYSTEM AND USER ROLES\n"
        )
        file.write("-" * 60 + "\n\n")

        file.write("SYSTEM:\n")
        file.write(system_prompt)
        file.write("\n\n")

        file.write("USER:\n")
        file.write(user_question)
        file.write("\n\n")

        file.write("ASSISTANT:\n")
        file.write(answer)
        file.write("\n\n")

        file.write(
            "TASK 3 - PROMPT COMPARISON\n"
        )
        file.write("-" * 60 + "\n\n")

        file.write(
            "VARIATION 1 - VAGUE PROMPT\n\n"
        )

        file.write("INPUT:\n")
        file.write(vague_prompt)
        file.write("\n\n")

        file.write("OUTPUT:\n")
        file.write(vague_answer)
        file.write("\n\n")

        file.write(
            "VARIATION 2 - CLEAR AND CONSTRAINED PROMPT\n\n"
        )

        file.write("INPUT:\n")
        file.write(clear_prompt)
        file.write("\n\n")

        file.write("OUTPUT:\n")
        file.write(clear_answer)
        file.write("\n\n")

        file.write(
            "TASK 4 - CHOSEN PROMPT NOTE\n"
        )
        file.write("-" * 60 + "\n\n")

        file.write(prompt_note)
        file.write("\n")

    logging.info(
        "Prompt comparison output saved to: %s",
        OUTPUT_FILE,
    )

    print(
        f"\nPrompt comparison saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()