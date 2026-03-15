"""PDF parser — extract text from PDFs and chunk it."""
import re
from pathlib import Path


def extract_pdfs(paths: list, max_pages: int = 20) -> list[dict]:
    """
    Extract text from a list of PDF file paths.

    Returns list of { 'text': str, 'source': str, 'page': int }
    """
    import pdfplumber

    pages = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        try:
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages[:max_pages]):
                    text = page.extract_text() or ""
                    text = _clean_text(text)
                    if len(text.strip()) > 30:          # skip near-empty pages
                        pages.append({
                            "text":   text,
                            "source": path.name,
                            "page":   i + 1,
                        })
        except Exception as exc:
            print(f"[pdf_parser] Error reading {path.name}: {exc}")

    return pages


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    Split *text* into overlapping character-level chunks.
    Returns list of non-empty strings.
    """
    if not text or not text.strip():
        return []

    chunks = []
    start  = 0
    length = len(text)

    while start < length:
        end   = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start += chunk_size - overlap

    return chunks


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()