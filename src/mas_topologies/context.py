"""Generic file-ingestion helpers."""

from typing import List

try:
    import PyPDF2
except ImportError:  # pragma: no cover
    PyPDF2 = None  # PyPDF2 is optional; only needed for PDF ingestion


def extract_pdf_text(pdf_path: str) -> str:
    """Extract plain text from a PDF file."""
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is required: pip install PyPDF2")
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as exc:
        print(f"Warning: could not extract PDF {pdf_path}: {exc}")
        return ""


def load_papers(paper_paths: List[str]) -> List[str]:
    """Load paper content from PDF or plain-text paths."""
    papers: List[str] = []
    for path in paper_paths:
        if path.endswith(".pdf"):
            papers.append(extract_pdf_text(path))
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    papers.append(f.read())
            except Exception as exc:
                print(f"Warning: could not read {path}: {exc}")
    return papers


__all__ = ["extract_pdf_text", "load_papers"]

