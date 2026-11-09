"""Document loading module supporting PDF, DOCX, and TXT files.

Extracts text preserving page boundaries, paragraph structure, and metadata.
"""

from dataclasses import dataclass
import io
from pathlib import Path
from docx import Document
from pypdf import PdfReader

from rag.text_cleaner import clean_text


class DocumentHandlingError(Exception):
    """Base exception for document handling errors."""


class UnsupportedFileTypeError(DocumentHandlingError):
    """Raised when an unsupported file format is uploaded."""


class EmptyDocumentError(DocumentHandlingError):
    """Raised when an uploaded document contains no readable text."""


class DocumentExtractionError(DocumentHandlingError):
    """Raised when document extraction fails due to corruption or parsing errors."""


@dataclass
class DocumentPage:
    """Represents an extracted page or structural section of a document."""
    text: str
    source_file: str
    page_number: int


def load_pdf(content: bytes, filename: str) -> list[DocumentPage]:
    """Extract text from a PDF file page by page."""
    pages: list[DocumentPage] = []
    try:
        reader = PdfReader(io.BytesIO(content))
        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""
            cleaned = clean_text(raw_text)
            if cleaned:
                pages.append(DocumentPage(text=cleaned, source_file=filename, page_number=page_idx))
    except Exception as e:
        raise DocumentExtractionError(f"Failed to read PDF file \'{filename}\': {e}") from e


    if not pages:
        raise EmptyDocumentError(f"PDF document '{filename}' contains no readable text.")

    return pages


def load_docx(content: bytes, filename: str) -> list[DocumentPage]:
    """Extract paragraphs and table text from a DOCX file."""
    parts: list[str] = []
    try:
        doc = Document(io.BytesIO(content))
        # Extract paragraphs (preserving headings and text)
        for paragraph in doc.paragraphs:
            p_text = paragraph.text.strip()
            if p_text:
                parts.append(p_text)

        # Extract tables if present
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    parts.append(" | ".join(row_cells))
    except Exception as e:
        raise DocumentExtractionError(f"Failed to read DOCX file \'{filename}\': {e}") from e


    combined_text = "\n\n".join(parts)
    cleaned = clean_text(combined_text)

    if not cleaned:
        raise EmptyDocumentError(f"DOCX document '{filename}' contains no readable text.")

    return [DocumentPage(text=cleaned, source_file=filename, page_number=1)]


def load_txt(content: bytes, filename: str) -> list[DocumentPage]:
    """Extract and decode text from a TXT file."""
    # Attempt UTF-8 decoding, fallback to latin-1 / replace
    try:
        raw_text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            raw_text = content.decode("latin-1")
        except Exception as e:
            raw_text = content.decode("utf-8", errors="replace")

    cleaned = clean_text(raw_text)
    if not cleaned:
        raise EmptyDocumentError(f"TXT document '{filename}' contains no readable text.")

    return [DocumentPage(text=cleaned, source_file=filename, page_number=1)]


def load_document(filename: str, content: bytes) -> list[DocumentPage]:
    """Unified document loader routing by file extension.

    Args:
        filename: Name of the file.
        content: Binary bytes of the document.

    Returns:
        List of DocumentPage objects containing cleaned text and page metadata.

    Raises:
        UnsupportedFileTypeError: If file extension is not supported.
        EmptyDocumentError: If document contains no readable text.
        DocumentExtractionError: If document cannot be parsed.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return load_pdf(content, filename)
    elif suffix == ".docx":
        return load_docx(content, filename)
    elif suffix == ".txt":
        return load_txt(content, filename)
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file format '{suffix}'. Supported formats are: .pdf, .docx, .txt"
        )
