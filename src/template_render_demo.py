from __future__ import annotations

from pathlib import Path

from src.prompt_templates import build_rag_messages


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "prompt_template_renders.txt"


def format_messages(title: str, messages: list[dict[str, str]]) -> str:
    """Format rendered chat messages for readable output samples."""
    lines = [title, "-" * len(title)]
    for message in messages:
        lines.append(f"{message['role'].upper()}:")
        lines.append(message["content"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    chat_messages = build_rag_messages(
        context=(
            "Clinical trials compare interventions to evaluate safety, "
            "effectiveness, and potential side effects."
        ),
        question="Why are control groups used in clinical trials?",
        output_instructions="Respond in 2 concise sentences.",
    )

    batch_messages = build_rag_messages(
        context=(
            "Phase 1 emphasizes initial safety, phase 2 explores early efficacy, "
            "and phase 3 confirms benefit-risk at scale."
        ),
        question="Summarize the purpose of phases 1, 2, and 3.",
        output_instructions=(
            "Return a bullet list with exactly 3 bullets, one per phase."
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    content = []
    content.append("PHARMALENS TEMPLATE RENDER EXAMPLES")
    content.append("=" * 40)
    content.append("")
    content.append(format_messages("FEATURE 1 - CHAT PATH RENDER", chat_messages))
    content.append("")
    content.append(format_messages("FEATURE 2 - BATCH CLI PATH RENDER", batch_messages))

    OUTPUT_FILE.write_text("\n".join(content), encoding="utf-8")

    print(f"Saved template render examples to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()