"""General utility helpers."""
import shutil
from pathlib import Path


def save_uploaded_file(uploaded_file, upload_dir) -> Path:
    """
    Save a Streamlit UploadedFile to *upload_dir*.
    Returns the saved file Path.
    """
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def format_source(doc: dict) -> str:
    """Return a human-readable source label for a document chunk."""
    source = doc.get("source", "Unknown")
    page   = doc.get("page",   "?")
    score  = doc.get("score")

    label = f"**{source}** — page {page}"
    if score is not None:
        label += f" *(relevance: {score:.2f})*"
    return label