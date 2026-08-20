from __future__ import annotations

from pathlib import Path

from src.config import load_settings
from src.llm_client import create_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "parameter_experiments.log"


SYSTEM_PROMPT = (
    "You are PharmaLens, a pharmaceutical research assistant. "
    "Answer factual clinical research questions clearly. "
    "Do not invent study results or unsupported medical facts."
)

USER_PROMPT = (
    "Explain the purpose of a clinical trial and mention the main "
    "things researchers evaluate."
)


def call_model(
    client,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
):
    """
    Send a chat completion request using the supplied parameters.
    """

    request = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": USER_PROMPT,
            },
        ],
    }

    if temperature is not None:
        request["temperature"] = temperature

    if max_tokens is not None:
        request["max_tokens"] = max_tokens

    if stop is not None:
        request["stop"] = stop

    return client.chat.completions.create(**request)


def main() -> None:
    settings = load_settings()

    client = create_client(settings)

    model = settings["chat_model"]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print("\n========== PHARMALENS MODEL PARAMETERS ==========\n")
    print(f"Model: {model}")

    # =========================================================
    # TASK 1 — TEMPERATURE
    # =========================================================

    print("\n" + "=" * 70)
    print("TASK 1 - TEMPERATURE COMPARISON")
    print("=" * 70)

    temperatures = [0.0, 1.0]

    for temperature in temperatures:

        response = call_model(
            client=client,
            model=model,
            temperature=temperature,
        )

        answer = response.choices[0].message.content or ""

        print(
            f"\n--- temperature={temperature} ---"
        )

        print(answer)

        results.append(
            f"""
TEMPERATURE EXPERIMENT
----------------------
temperature: {temperature}

Prompt:
{USER_PROMPT}

Output:
{answer}

Token usage:
{response.usage}
"""
        )

    # =========================================================
    # TASK 2 — MAX TOKENS
    # =========================================================

    print("\n" + "=" * 70)
    print("TASK 2 - MAX TOKENS")
    print("=" * 70)

    max_token_values = [30, 150]

    for max_tokens in max_token_values:

        response = call_model(
            client=client,
            model=model,
            temperature=0.1,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content or ""

        print(
            f"\n--- max_tokens={max_tokens} ---"
        )

        print(answer)

        results.append(
            f"""
MAX TOKENS EXPERIMENT
---------------------
max_tokens: {max_tokens}

Prompt:
{USER_PROMPT}

Output:
{answer}

Token usage:
{response.usage}
"""
        )

    # =========================================================
    # TASK 3 — STOP
    # =========================================================

    print("\n" + "=" * 70)
    print("TASK 3 - STOP PARAMETER")
    print("=" * 70)

    stop_prompt = (
        "Give three short facts about clinical trials. "
        "After the third fact, write END."
    )

    stop_request = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": stop_prompt,
            },
        ],
        "temperature": 0.1,
        "max_tokens": 150,
    }

    # Call without stop
    response_without_stop = client.chat.completions.create(
        **stop_request
    )

    answer_without_stop = (
        response_without_stop.choices[0].message.content or ""
    )

    print("\n--- Without stop ---")
    print(answer_without_stop)

    results.append(
        f"""
STOP EXPERIMENT - WITHOUT STOP
------------------------------
Prompt:
{stop_prompt}

Output:
{answer_without_stop}

Token usage:
{response_without_stop.usage}
"""
    )

    # Call with stop
    response_with_stop = client.chat.completions.create(
        **stop_request,
        stop=["END"],
    )

    answer_with_stop = (
        response_with_stop.choices[0].message.content or ""
    )

    print("\n--- With stop=['END'] ---")
    print(answer_with_stop)

    results.append(
        f"""
STOP EXPERIMENT - WITH STOP
---------------------------
stop: ["END"]

Prompt:
{stop_prompt}

Output:
{answer_with_stop}

Token usage:
{response_with_stop.usage}
"""
    )

    # =========================================================
    # TASK 4 — RECOMMENDED SETTINGS
    # =========================================================

    recommendation = """
RECOMMENDED SETTINGS FOR PHARMALENS GROUNDED Q&A
================================================

Temperature:
Use a low temperature such as 0.1.

Reason:
A low temperature makes factual answers more focused,
consistent, and less likely to introduce unnecessary creative
content.

Max tokens:
Use a sensible output limit such as 300 tokens.

Reason:
The limit prevents unnecessarily long answers and helps control
output-token usage and cost.

Stop:
Use a stop sequence when the application has a clearly defined
output boundary.

Reason:
A stop sequence can prevent the model from continuing beyond
the required response format.

Top-p:
Temperature and top-p should generally not both be aggressively
tuned at the same time. The main PharmaLens configuration should
start with a low temperature and only introduce additional
sampling controls when needed.

Recommended starting configuration:

temperature = 0.1
max_tokens = 300
stop = optional
"""

    print("\n" + "=" * 70)
    print(recommendation)

    results.append(recommendation)

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PHARMALENS - MODEL PARAMETERS & OUTPUT CONTROL\n"
        )

        file.write("=" * 70 + "\n")

        file.write(
            f"\nMODEL: {model}\n"
        )

        file.write(
            f"\nSYSTEM PROMPT:\n{SYSTEM_PROMPT}\n"
        )

        for result in results:
            file.write("\n")
            file.write(result)
            file.write("\n")

    print(
        f"\nParameter experiment results saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()