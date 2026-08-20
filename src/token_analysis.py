from __future__ import annotations

from pathlib import Path

import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "sample_clinical_report.txt"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "token_analysis.log"


# Example rates used only for demonstrating the cost calculation.
# Rates are expressed in USD per 1,000 tokens.
INPUT_COST_PER_1K = 0.0005
OUTPUT_COST_PER_1K = 0.0015


def get_tokenizer():
    """
    Return the tokenizer used for this assignment.
    """
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, tokenizer) -> int:
    """
    Count the number of tokens in a text string.
    """
    return len(tokenizer.encode(text))


def count_words(text: str) -> int:
    """
    Return a simple whitespace-based word count.
    """
    return len(text.split())


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
) -> tuple[float, float, float]:
    """
    Estimate input, output, and total cost.

    The rates used here are example rates for demonstrating
    token-based cost estimation.
    """

    input_cost = (
        input_tokens / 1000
    ) * INPUT_COST_PER_1K

    output_cost = (
        output_tokens / 1000
    ) * OUTPUT_COST_PER_1K

    total_cost = input_cost + output_cost

    return input_cost, output_cost, total_cost


def format_sample_result(
    name: str,
    text: str,
    tokenizer,
) -> str:
    """
    Create a readable token-analysis result for one sample.
    """

    characters = len(text)
    words = count_words(text)
    tokens = count_tokens(text, tokenizer)

    return (
        f"{name}\n"
        f"{'-' * 60}\n"
        f"Characters : {characters}\n"
        f"Words      : {words}\n"
        f"Tokens     : {tokens}\n"
        f"Text:\n{text}\n"
    )


def main() -> None:
    tokenizer = get_tokenizer()

    # =========================================================
    # TASK 1 & TASK 2
    # Count tokens for three samples of different lengths.
    # =========================================================

    short_question = (
        "What adverse events were reported in Study PL-101?"
    )

    paragraph = (
        "Study PL-101 evaluated the safety and effectiveness of "
        "Drug X in adult participants. Researchers monitored the "
        "participants during the clinical trial and recorded "
        "adverse events, clinical outcomes, and safety information."
    )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Sample document not found: {DATA_FILE}"
        )

    full_document = DATA_FILE.read_text(encoding="utf-8")

    samples = [
        ("SAMPLE 1 - SHORT QUESTION", short_question),
        ("SAMPLE 2 - PARAGRAPH", paragraph),
        ("SAMPLE 3 - FULL DOCUMENT", full_document),
    ]

    results = []

    print("\n========== PHARMALENS TOKEN ANALYSIS ==========\n")

    for name, text in samples:
        result = format_sample_result(
            name,
            text,
            tokenizer,
        )

        results.append(result)
        print(result)

    # =========================================================
    # TASK 3
    # Estimate input and output cost separately.
    # =========================================================

    input_text = (
        short_question
        + "\n\n"
        + paragraph
    )

    sample_output = (
        "Study PL-101 reported headache, nausea, and fatigue "
        "as commonly observed adverse events."
    )

    input_tokens = count_tokens(
        input_text,
        tokenizer,
    )

    output_tokens = count_tokens(
        sample_output,
        tokenizer,
    )

    input_cost, output_cost, total_cost = estimate_cost(
        input_tokens,
        output_tokens,
    )

    cost_result = (
        "COST ESTIMATION\n"
        f"{'-' * 60}\n"
        f"Example input rate  : ${INPUT_COST_PER_1K:.6f} per 1K tokens\n"
        f"Example output rate : ${OUTPUT_COST_PER_1K:.6f} per 1K tokens\n\n"
        f"Input tokens  : {input_tokens}\n"
        f"Output tokens : {output_tokens}\n\n"
        f"Estimated input cost  : ${input_cost:.8f}\n"
        f"Estimated output cost : ${output_cost:.8f}\n"
        f"Estimated total cost  : ${total_cost:.8f}\n"
    )

    print(cost_result)

    # =========================================================
    # TASK 4
    # Show that characters, words, and tokens are related
    # but not exactly proportional.
    # =========================================================

    relationship_samples = [
        "drug",
        "pharmacovigilance",
        "Drug X caused mild headache.",
        "patient_id = 1024; adverse_event = True",
        "నమస్కారం",
    ]

    relationship_lines = [
        "LENGTH VS TOKEN RELATIONSHIP",
        "-" * 60,
        f"{'Text':<45} {'Chars':>6} {'Words':>6} {'Tokens':>7}",
        "-" * 70,
    ]

    for text in relationship_samples:
        characters = len(text)
        words = count_words(text)
        tokens = count_tokens(text, tokenizer)

        display_text = text

        if len(display_text) > 42:
            display_text = display_text[:39] + "..."

        relationship_lines.append(
            f"{display_text:<45} "
            f"{characters:>6} "
            f"{words:>6} "
            f"{tokens:>7}"
        )

    relationship_lines.extend(
        [
            "",
            "Observation:",
            (
                "Longer text generally produces more tokens, but "
                "characters, words, and tokens are not exactly "
                "proportional. Long technical words, code-like text, "
                "punctuation, and different languages can be split "
                "into tokens differently."
            ),
        ]
    )

    relationship_result = "\n".join(
        relationship_lines
    )

    print("\n" + relationship_result)

    # =========================================================
    # TASK 5
    # Save sample results.
    # =========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PHARMALENS - TOKENS, TOKENIZATION & COST ESTIMATION\n"
        )

        file.write("=" * 70 + "\n\n")

        for result in results:
            file.write(result)
            file.write("\n")

        file.write(cost_result)
        file.write("\n")

        file.write(relationship_result)
        file.write("\n")

    print(
        f"\nSample results saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()