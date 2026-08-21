from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "corpus_loader_intake.log"
SAMPLE_CORPUS_DIR = PROJECT_ROOT / "data" / "sample_corpus"


@dataclass
class LoadedDocument:
    """Common plain-text representation with source identity."""

    source_id: str
    text: str


class _HTMLTextExtractor(HTMLParser):
    """Extract visible text from basic HTML content."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._parts)


def to_plain_text(path: Path) -> str:
    """Convert supported file formats to a plain-text string."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    if suffix in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8")
        parser = _HTMLTextExtractor()
        parser.feed(html)
        return parser.get_text()

    raise ValueError(f"Unsupported file type: {suffix or '<no extension>'}")


def load_documents(paths: list[Path]) -> tuple[list[LoadedDocument], list[str]]:
    """Load documents while surviving missing/unreadable/unsupported inputs."""
    loaded: list[LoadedDocument] = []
    skipped: list[str] = []

    for path in paths:
        if not path.exists():
            skipped.append(f"SKIPPED {path.name}: missing file")
            continue

        if not path.is_file():
            skipped.append(f"SKIPPED {path.name}: not a regular file")
            continue

        try:
            plain_text = to_plain_text(path)
        except UnicodeDecodeError as error:
            skipped.append(
                f"SKIPPED {path.name}: unreadable text ({error.__class__.__name__})"
            )
            continue
        except ValueError as error:
            skipped.append(f"SKIPPED {path.name}: {error}")
            continue
        except OSError as error:
            skipped.append(f"SKIPPED {path.name}: file error ({error})")
            continue

        loaded.append(
            LoadedDocument(
                source_id=path.name,
                text=plain_text.strip(),
            )
        )

    return loaded, skipped


def preview(text: str, limit: int = 120) -> str:
    """Return a compact preview for intake confirmation."""
    compact = " ".join(text.split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def collect_sample_paths() -> list[Path]:
    """Build a realistic mixed input list, including bad entries."""
    return [
        SAMPLE_CORPUS_DIR / "clinical_trial_overview.txt",
        SAMPLE_CORPUS_DIR / "eligibility_criteria.md",
        SAMPLE_CORPUS_DIR / "study_export.html",
        SAMPLE_CORPUS_DIR / "fake_scan.pdf",
        SAMPLE_CORPUS_DIR / "missing_notes.txt",
    ]


def main() -> None:
    paths = collect_sample_paths()
    loaded_docs, skipped = load_documents(paths)

    lines: list[str] = []
    lines.append("PHARMALENS CORPUS LOADER INTAKE")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Requested files: {len(paths)}")
    lines.append(f"Loaded files   : {len(loaded_docs)}")
    lines.append(f"Skipped files  : {len(skipped)}")
    lines.append("")

    for message in skipped:
        lines.append(message)

    if skipped:
        lines.append("")

    for doc in loaded_docs:
        lines.append(f"LOADED {doc.source_id}")
        lines.append(f"length={len(doc.text)}")
        lines.append(f"sample={preview(doc.text)}")
        lines.append("")

    report = "\n".join(lines).rstrip() + "\n"

    print(report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(f"Saved intake report to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()