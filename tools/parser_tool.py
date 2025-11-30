import re
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from config.settings import CHUNK_MAX_TOKENS
from rag.rag_system import RagSystem


# ------------------------------------------------------------
# PDF Extraction using PyMuPDF (MUCH faster & cleaner)
# ------------------------------------------------------------
def extract_text_from_pdf(path: str) -> str:
    """
    Extract text from PDF using PyMuPDF (fitz).
    Much faster and more accurate than pdfminer.
    """
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text("text"))
    return "\n".join(text)


# ------------------------------------------------------------
# DOCX Extraction (unchanged, interface same)
# ------------------------------------------------------------
def extract_text_from_docx(path: str) -> str:
    doc = DocxDocument(path)
    paras = [p.text for p in doc.paragraphs if p.text]
    return '\n'.join(paras)


# ------------------------------------------------------------
# TXT Extraction (unchanged)
# ------------------------------------------------------------
def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------
# Auto File Parser (same interface)
# ------------------------------------------------------------
def parse_file(path: str) -> str:
    lpath = path.lower()

    if lpath.endswith(".pdf"):
        return extract_text_from_pdf(path)
    if lpath.endswith(".docx"):
        return extract_text_from_docx(path)
    if lpath.endswith(".txt"):
        return extract_text_from_txt(path)

    # fallback
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------
# Add parsed file to RAG (interface unchanged)
# ------------------------------------------------------------
def parse_file_and_add_to_rag(path: str, rag_system: RagSystem, chunk_tokens: int = CHUNK_MAX_TOKENS):
    """
    Extract text from file (PDF/DOCX/TXT) using the new fast parser,
    then index into the provided RagSystem instance.
    """

    text = parse_file(path)

    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()

    # Index into RAG
    result = rag_system.add_documents(
        [text],
        max_tokens=chunk_tokens,
        overlap=rag_system.tokenizer.model_max_length // 10,
        save=False
    )

    return result
