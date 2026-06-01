"""Data loading and chunk preparation for the textbook retrieval pipeline."""

from collections import defaultdict
import os
import re

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


NON_CONTENT_ROLES = {"pageHeader", "pageFooter", "pageNumber", "footnote"}
CHAPTER_STARTS = [
    (24, "Chapter 1. The Machine Learning Landscape"),
    (54, "Chapter 2. End-to-End Machine Learning Project"),
    (102, "Chapter 3. Classification"),
    (128, "Chapter 4. Training Models"),
    (168, "Chapter 5. Support Vector Machines"),
    (190, "Chapter 6. Decision Trees"),
    (204, "Chapter 7. Ensemble Learning and Random Forests"),
    (228, "Chapter 8. Dimensionality Reduction"),
    (252, "Chapter 9. Up and Running with TensorFlow"),
    (276, "Chapter 10. Introduction to Artificial Neural Networks"),
    (298, "Chapter 11. Training Deep Neural Nets"),
    (338, "Chapter 12. Distributing TensorFlow Across Devices and Servers"),
    (378, "Chapter 13. Convolutional Neural Networks"),
    (404, "Chapter 14. Recurrent Neural Networks"),
    (438, "Chapter 15. Autoencoders"),
    (498, "Appendix A. Exercise Solutions"),
    (524, "Appendix B. Machine Learning Project Checklist"),
    (530, "Appendix C. SVM Dual Problem"),
    (534, "Appendix D. Autodiff"),
    (542, "Appendix E. Other Popular ANN Architectures"),
]


CHAPTER_TITLE_BY_MARKER = {
    re.match(r"^(Chapter\s+\d+|Appendix\s+[A-Z])", title, re.IGNORECASE)
    .group(1)
    .lower(): title
    for _, title in CHAPTER_STARTS
}


def _get_env(name, default=None):
    """Read an environment variable, tolerating whitespace around keys."""
    value = os.getenv(name)
    if value is not None:
        return value.strip()

    for key, candidate in os.environ.items():
        if key.strip() == name:
            return candidate.strip()
    return default


def get_chapter_for_page(page_number):
    """Resolve the chapter title for a zero-based PDF page index."""
    chapter = "Front Matter"
    for start_page, title in CHAPTER_STARTS:
        if page_number >= start_page:
            chapter = title
        else:
            break
    return chapter


def get_chapter_for_marker(marker):
    """Resolve a normalized chapter or appendix marker to the full title."""
    return CHAPTER_TITLE_BY_MARKER.get(marker.lower())


def get_document_intelligence_config():
    """Return Azure Document Intelligence configuration from the environment."""
    endpoint = _get_env("AZURE_DOCUMENT_INTEL_ENDPOINT")
    api_key = _get_env("AZURE_DOCUMENT_INTEL_KEY")
    missing = [
        name
        for name, value in {
            "AZURE_DOCUMENT_INTEL_ENDPOINT": endpoint,
            "AZURE_DOCUMENT_INTEL_KEY": api_key,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(
            f"Missing Azure Document Intelligence environment variables: {missing}"
        )

    return endpoint, api_key


def load_pdf_data(path):
    """Load a PDF as page-level documents using the local PyPDF loader."""
    loader = PyPDFLoader(path)
    return loader.load()


def analyze_textbook_with_layout(path, endpoint=None, api_key=None):
    """Analyze a textbook PDF with Azure Document Intelligence layout model."""
    if not endpoint or not api_key:
        endpoint, api_key = get_document_intelligence_config()

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )
    with open(path, "rb") as file:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            body=file,
        )
    return poller.result()


def detect_chapter_title(text):
    """Detect chapter or appendix titles from cleaned layout text."""
    text = clean_text(text)
    patterns = [
        r"^(Chapter\s+\d+|Appendix\s+[A-Z])\.?\b",
        r"^(?:CHAPTER|Chapter)\s*(\d+)\b",
        r"^(?:APPENDIX|Appendix)\s*([A-Z])\b",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if not match:
            continue

        marker = match.group(1)
        if marker.isdigit():
            marker = f"Chapter {marker}"
        elif len(marker) == 1 and marker.isalpha():
            marker = f"Appendix {marker.upper()}"

        return get_chapter_for_marker(marker) or text

    return None


def detect_book_page_number(text):
    """Detect an Arabic book page number from a page number/footer paragraph."""
    text = clean_text(text)
    match = re.fullmatch(r"(?:page\s*)?(\d{1,4})", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _new_page_record():
    return {
        "content": [],
        "headers": [],
        "footers": [],
        "page_numbers": [],
        "chapter_candidates": [],
    }


def _infer_page_chapters(page_records):
    """Infer chapter for each page from candidates, neighbors, then page index."""
    page_chapters = {}
    current_chapter = None
    for page_index in sorted(page_records):
        candidates = page_records[page_index]["chapter_candidates"]
        if candidates:
            current_chapter = candidates[-1]
        if current_chapter:
            page_chapters[page_index] = current_chapter

    next_chapter = None
    for page_index in sorted(page_records, reverse=True):
        if page_index in page_chapters:
            next_chapter = page_chapters[page_index]
            continue
        if next_chapter:
            page_chapters[page_index] = next_chapter

    for page_index in page_records:
        page_chapters.setdefault(page_index, get_chapter_for_page(page_index))

    return page_chapters


def _infer_book_page_label(page_index, page_record):
    """Infer the page number printed in the book."""
    for text in page_record["page_numbers"] + page_record["footers"]:
        page_number = detect_book_page_number(text)
        if page_number is not None:
            return page_number
    return page_index + 1


def build_docs_from_layout_result(result):
    """Convert a Document Intelligence layout result into page documents."""
    page_records = defaultdict(_new_page_record)

    for paragraph in result.paragraphs or []:
        role = getattr(paragraph, "role", None)

        text = clean_text(getattr(paragraph, "content", ""))
        if not text:
            continue

        bounding_regions = getattr(paragraph, "bounding_regions", None)
        if not bounding_regions:
            continue

        page_number = bounding_regions[0].page_number
        pdf_page_index = page_number - 1
        page_record = page_records[pdf_page_index]

        detected_chapter = detect_chapter_title(text)
        if detected_chapter:
            page_record["chapter_candidates"].append(detected_chapter)

        if role == "pageHeader":
            page_record["headers"].append(text)
            continue
        if role == "pageFooter":
            page_record["footers"].append(text)
            continue
        if role == "pageNumber":
            page_record["page_numbers"].append(text)
            continue
        if role == "footnote":
            continue

        page_record["content"].append(text)

    page_chapters = _infer_page_chapters(page_records)

    docs = []
    for pdf_page_index in sorted(page_records):
        page_record = page_records[pdf_page_index]
        if not page_record["content"]:
            continue

        chapter = page_chapters[pdf_page_index]
        page_label = _infer_book_page_label(pdf_page_index, page_record)
        docs.append(
            Document(
                page_content="\n\n".join(page_record["content"]),
                metadata={
                    "page": page_label,
                    "pdf_page": pdf_page_index,
                    "page_label": page_label,
                    "chapter": chapter,
                    "metadata_label": f"{chapter} | page {page_label}",
                },
            )
        )

    return docs


def load_data(path):
    """Load the source PDF as page-level documents with Azure layout extraction.

    This is the entry point for the retrieval data pipeline. Downstream stages
    preserve page metadata so generated answers can cite the original source.

    Args:
        path: Path to the textbook PDF.

    Returns:
        list: Page-level LangChain documents produced from layout paragraphs.
    """
    result = analyze_textbook_with_layout(path)
    return build_docs_from_layout_result(result)


def split_chunk(document):
    """Attach chapter metadata to each page before chunking.

    Chapter labels provide a stable, human-readable source identifier for both
    retrieval inspection and final answer citations.

    Args:
        document: Page-level documents returned by :func:`load_data`.

    Returns:
        list: Documents enriched with chapter and display metadata.
    """

    filtered_docs = []
    for page in document:
        pdf_page_index = page.metadata.get("pdf_page", page.metadata.get("page", 0))
        page_label = page.metadata.get("page_label", page.metadata.get("page", 0))
        chapter = page.metadata.get("chapter") or get_chapter_for_page(pdf_page_index)
        page.metadata["pdf_page"] = pdf_page_index
        page.metadata["page"] = page_label
        page.metadata["page_label"] = page_label
        page.metadata["chapter"] = chapter
        page.metadata["metadata_label"] = page.metadata.get(
            "metadata_label",
            f"{chapter} | page {page_label}",
        )
        filtered_docs.append(page)

    return filtered_docs


def clean_text(text):
    """Normalize PDF extraction artifacts before embedding.

    Collapsing whitespace reduces embedding noise caused by PDF line wrapping
    and inconsistent spacing, while preserving the semantic content needed for
    retrieval.

    Args:
        text: Raw text extracted from a PDF page or chunk.

    Returns:
        str: Cleaned text with normalized whitespace.
    """
    cleaned_text = text.encode("utf-8", "ignore").decode("utf-8")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text


def split_clean_chunks(filtered_docs):
    """Split textbook pages into retrieval-friendly chunks.

    The chunking strategy balances semantic fidelity and retrieval efficiency.
    A chunk size of 800 characters is small enough to keep each embedding
    focused on a narrow topic, while 150 characters of overlap preserves
    context when an explanation spans a chunk boundary. The ordered separators
    prefer paragraph, line, and sentence boundaries before falling back to
    whitespace and raw character splits.

    Args:
        filtered_docs: Page-level documents with chapter metadata attached.

    Returns:
        list: Cleaned chunk documents ready for embedding.

    Notes:
        Smaller chunks usually improve retrieval precision, but they increase
        index size and may reduce recall if overlap is too low.
    """
    # Prefer natural text boundaries so chunk embeddings map more cleanly to a
    # single concept or explanation.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(filtered_docs)

    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)
    return chunks
