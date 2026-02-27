from pathlib import Path


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pypdf'. Install with: pip install -r requirements.txt"
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"--- PAGE {i} ---\\n{text.strip()}")
    return "\\n\\n".join(pages).strip()
