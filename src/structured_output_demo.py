from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import load_settings
from src.llm_client import create_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_OUTPUT_FILE = OUTPUT_DIR / "structured_output_samples.json"

REQUIRED_FIELDS = ("answer", "source")


@dataclass
class ParseAttempt:
    """Represents a safe parse attempt for a raw LLM response."""

    ok: bool
    data: dict[str, Any] | None
    recovered: bool
    error: str | None


def build_messages(question: str) -> list[dict[str, str]]:
    """Prompt the model to return only a strict JSON object."""
    schema_hint = '{"answer": "<short answer>", "source": "<citation or source id>"}'

    return [
        {
            "role": "system",
            "content": (
                "You are a pharmaceutical research assistant. "
                "Return ONLY valid JSON with this exact shape: "
                f"{schema_hint}. "
                "Do not include markdown, prose, or extra keys."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]


def call_model_for_json(question: str) -> str | None:
    """Request JSON mode output from the model when credentials are available."""
    try:
        settings = load_settings()
    except ValueError:
        # Keep demo runnable without secrets by skipping the live API call.
        return None

    client = create_client(settings)
    response = client.chat.completions.create(
        model=settings["chat_model"],
        messages=build_messages(question),
        response_format={"type": "json_object"},
    )

    return response.choices[0].message.content


def extract_json_object(raw_text: str) -> str | None:
    """Extract a likely JSON object from mixed text.

    This is a best-effort recovery path for malformed model output.
    """
    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    return raw_text[start : end + 1]


def safe_parse_json(raw_text: str) -> ParseAttempt:
    """Parse model output without raising unhandled JSON exceptions."""
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return ParseAttempt(ok=True, data=data, recovered=False, error=None)
        return ParseAttempt(
            ok=False,
            data=None,
            recovered=False,
            error="Top-level JSON value must be an object.",
        )
    except json.JSONDecodeError as first_error:
        candidate = extract_json_object(raw_text)
        if candidate is None:
            return ParseAttempt(
                ok=False,
                data=None,
                recovered=False,
                error=f"Malformed JSON: {first_error}",
            )

        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return ParseAttempt(ok=True, data=data, recovered=True, error=None)
        except json.JSONDecodeError:
            pass

        # Fallback for Python-dict-like strings with single quotes.
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return ParseAttempt(
                    ok=True,
                    data=dict(parsed),
                    recovered=True,
                    error=None,
                )
        except (ValueError, SyntaxError):
            pass

        return ParseAttempt(
            ok=False,
            data=None,
            recovered=False,
            error=f"Malformed JSON could not be recovered: {first_error}",
        )


def validate_required_fields(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate required fields before payload is used by downstream code."""
    missing_or_invalid: list[str] = []

    for field in REQUIRED_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            missing_or_invalid.append(field)

    return (len(missing_or_invalid) == 0, missing_or_invalid)


def evaluate_case(case_name: str, raw_text: str) -> dict[str, Any]:
    """Parse + validate one response and return a report-friendly record."""
    parse_result = safe_parse_json(raw_text)

    report: dict[str, Any] = {
        "case": case_name,
        "raw_response": raw_text,
        "parse_ok": parse_result.ok,
        "recovered": parse_result.recovered,
        "parse_error": parse_result.error,
        "validated": False,
        "missing_fields": [],
        "parsed": None,
    }

    if not parse_result.ok or parse_result.data is None:
        return report

    valid, missing = validate_required_fields(parse_result.data)
    report["validated"] = valid
    report["missing_fields"] = missing
    report["parsed"] = parse_result.data

    return report


def build_demo_cases() -> list[tuple[str, str]]:
    """Provide stable examples, including malformed-then-recovered JSON."""
    return [
        (
            "valid_json",
            '{"answer": "Clinical trials test safety and efficacy in people.", '
            '"source": "NCI - Clinical Trials Information"}',
        ),
        (
            "malformed_then_recovered",
            "Model note: {'answer': 'Trial phases evaluate safety and effectiveness.', "
            "'source': 'WHO Clinical Trials Registry'} -- done",
        ),
        (
            "missing_required_field",
            '{"answer": "A trial checks if a treatment works and is safe."}',
        ),
    ]


def main() -> None:
    question = "What is the purpose of a clinical trial? Include one source."

    records: list[dict[str, Any]] = []
    for case_name, raw_text in build_demo_cases():
        records.append(evaluate_case(case_name, raw_text))

    live_raw = call_model_for_json(question)
    if live_raw is not None:
        records.append(evaluate_case("live_model_response", live_raw))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "required_fields": list(REQUIRED_FIELDS),
        "question": question,
        "results": records,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_OUTPUT_FILE.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("Structured output demo complete.")
    print(f"Saved sample results to: {SAMPLE_OUTPUT_FILE}")

    for row in records:
        print(
            f"- {row['case']}: parse_ok={row['parse_ok']} "
            f"recovered={row['recovered']} validated={row['validated']}"
        )
        if row["parse_error"]:
            print(f"  parse_error: {row['parse_error']}")
        if row["missing_fields"]:
            print(f"  missing_fields: {', '.join(row['missing_fields'])}")


if __name__ == "__main__":
    main()