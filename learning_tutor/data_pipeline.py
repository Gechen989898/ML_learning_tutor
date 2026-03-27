"""Data loading and chunk preparation for the textbook retrieval pipeline."""

import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def load_data(path):
    """Load the source PDF as page-level documents.

    This is the entry point for the retrieval data pipeline. Downstream stages
    preserve page metadata so generated answers can cite the original source.

    Args:
        path: Path to the textbook PDF.

    Returns:
        list: Page-level LangChain documents produced by the PDF loader.
    """
    loader = PyPDFLoader(path)
    return loader.load()


def split_chunk(document):
    """Attach chapter metadata to each page before chunking.

    Chapter labels provide a stable, human-readable source identifier for both
    retrieval inspection and final answer citations.

    Args:
        document: Page-level documents returned by :func:`load_data`.

    Returns:
        list: Documents enriched with chapter and display metadata.
    """

    def get_chapter(page_number):
        """Resolve the chapter title for a PDF page index.

        Args:
            page_number: Zero-based page index from the PDF loader.

        Returns:
            str: Chapter title that contains the page.
        """
        chapter = "Front Matter"
        for start_page, title in CHAPTER_STARTS:
            if page_number >= start_page:
                chapter = title
            else:
                break
        return chapter

    filtered_docs = []
    for page in document:
        chapter = get_chapter(page.metadata["page"])
        page_label = page.metadata.get("page_label", page.metadata["page"] + 1)
        page.metadata["chapter"] = chapter
        page.metadata["metadata_label"] = f"{chapter} | page {page_label}"
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
