from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"

_TEMPLATE_FILES = {
    "rag_system": "rag_system.txt",
    "rag_user": "rag_user.txt",
}


def load_template(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    file_name = _TEMPLATE_FILES.get(name)
    if file_name is None:
        raise ValueError(f"Unknown template name: {name}")

    path = PROMPTS_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {path}")

    return path.read_text(encoding="utf-8")


def render_template(name: str, **values: str) -> str:
    """Render a named template with placeholder values."""
    template = load_template(name)
    try:
        return template.format(**values)
    except KeyError as missing_key:
        key = missing_key.args[0]
        raise ValueError(f"Missing value for placeholder: {key}") from missing_key


def build_rag_messages(
    *,
    context: str,
    question: str,
    output_instructions: str,
    assistant_name: str = "PharmaLens",
) -> list[dict[str, str]]:
    """Build system and user messages from shared RAG templates."""
    system_prompt = render_template(
        "rag_system",
        assistant_name=assistant_name,
    )
    user_prompt = render_template(
        "rag_user",
        context=context,
        question=question,
        output_instructions=output_instructions,
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]