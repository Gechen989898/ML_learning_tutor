"""Unit tests for Azure RAG migration helpers."""

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from langchain_core.documents import Document

from learning_tutor.azure_openai import get_azure_openai_config
from learning_tutor.azure_search import get_blob_config
from learning_tutor.data_pipeline import (
    build_docs_from_layout_result,
    detect_chapter_title,
    split_chunk,
    split_clean_chunks,
)


@contextmanager
def patched_env(values, clear=()):
    original = {name: os.environ.get(name) for name in set(values) | set(clear)}
    try:
        for name in clear:
            os.environ.pop(name, None)
        for name, value in values.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def paragraph(content, page_number, role=None):
    return SimpleNamespace(
        content=content,
        role=role,
        bounding_regions=[SimpleNamespace(page_number=page_number)],
    )


class AzureMigrationTests(unittest.TestCase):
    def test_blob_config_uses_account_endpoint_alias(self):
        with patched_env(
            {
                "AZURE_ACCOUNT_ENDPOINT": "https://storage.example.net",
                "AZURE_STORAGE_CONTAINER": "documents",
                "AZURE_STORAGE_BLOB_NAME": "book.pdf",
            },
            clear=("AZURE_STORAGE_ACCOUNT_URL",),
        ):
            self.assertEqual(
                get_blob_config(),
                ("https://storage.example.net", "documents", "book.pdf"),
            )

    def test_layout_result_becomes_page_documents(self):
        result = SimpleNamespace(
            paragraphs=[
                paragraph("ignored header", 1, role="pageHeader"),
                paragraph("Chapter 1. The Machine Learning Landscape", 1, role="title"),
                paragraph("Useful textbook content.", 1),
                paragraph("More useful content.", 2),
            ]
        )

        docs = build_docs_from_layout_result(result)

        self.assertEqual(len(docs), 2)
        self.assertEqual(
            docs[0].metadata["chapter"],
            "Chapter 1. The Machine Learning Landscape",
        )
        self.assertEqual(
            docs[1].metadata["metadata_label"],
            "Chapter 1. The Machine Learning Landscape | page 2",
        )
        self.assertNotIn("ignored header", docs[0].page_content)

    def test_layout_header_can_set_chapter_without_entering_page_content(self):
        result = SimpleNamespace(
            paragraphs=[
                paragraph("Chapter 2", 55, role="pageHeader"),
                paragraph("Chapter body content.", 55),
            ]
        )

        docs = build_docs_from_layout_result(result)

        self.assertEqual(
            docs[0].metadata["chapter"],
            "Chapter 2. End-to-End Machine Learning Project",
        )
        self.assertNotIn("Chapter 2", docs[0].page_content)

    def test_layout_docs_fall_back_to_page_chapter_map(self):
        result = SimpleNamespace(
            paragraphs=[
                paragraph("A page without an explicit heading.", 25),
            ]
        )

        docs = build_docs_from_layout_result(result)

        self.assertEqual(
            docs[0].metadata["chapter"],
            "Chapter 1. The Machine Learning Landscape",
        )

    def test_layout_docs_use_printed_book_page_number(self):
        result = SimpleNamespace(
            paragraphs=[
                paragraph("Chapter 3", 103, role="pageFooter"),
                paragraph("91", 103, role="pageNumber"),
                paragraph("Classification content.", 103),
            ]
        )

        docs = build_docs_from_layout_result(result)

        self.assertEqual(docs[0].metadata["page"], 91)
        self.assertEqual(docs[0].metadata["pdf_page"], 102)
        self.assertEqual(
            docs[0].metadata["metadata_label"],
            "Chapter 3. Classification | page 91",
        )

    def test_layout_docs_infer_missing_chapter_from_neighbor_pages(self):
        result = SimpleNamespace(
            paragraphs=[
                paragraph("Chapter 4", 128, role="pageFooter"),
                paragraph("Training model content.", 128),
                paragraph("A page whose footer only has the page number.", 129),
            ]
        )

        docs = build_docs_from_layout_result(result)

        self.assertEqual(
            docs[1].metadata["chapter"],
            "Chapter 4. Training Models",
        )

    def test_detect_chapter_title_accepts_marker_without_period(self):
        self.assertEqual(
            detect_chapter_title("CHAPTER 3"),
            "Chapter 3. Classification",
        )

    def test_chunking_preserves_document_intelligence_metadata(self):
        docs = [
            Document(
                page_content="This paragraph should stay associated with its source page.",
                metadata={
                    "page": 4,
                    "page_label": 5,
                    "chapter": "Chapter 2. End-to-End Machine Learning Project",
                    "metadata_label": (
                        "Chapter 2. End-to-End Machine Learning Project | page 5"
                    ),
                },
            )
        ]

        enriched_docs = split_chunk(docs)
        chunks = split_clean_chunks(enriched_docs)

        self.assertEqual(chunks[0].metadata["page"], 5)
        self.assertEqual(
            chunks[0].metadata["chapter"],
            "Chapter 2. End-to-End Machine Learning Project",
        )
        self.assertEqual(
            chunks[0].metadata["metadata_label"],
            "Chapter 2. End-to-End Machine Learning Project | page 5",
        )

    def test_azure_openai_config_reports_missing_chat_deployment(self):
        with patched_env(
            {
                "AZURE_OPENAI_ENDPOINT": "https://openai.example.net",
                "AZURE_OPENAI_API_KEY": "secret",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
            },
            clear=(
                "AZURE_OPENAI_CHAT_DEPLOYMENT",
                "AZURE_OPEN_AI_LLM_DEPLOYMENT",
            ),
        ):
            with self.assertRaisesRegex(ValueError, "AZURE_OPENAI_CHAT_DEPLOYMENT"):
                get_azure_openai_config(require_chat=True)

    def test_azure_openai_config_supports_separate_embedding_and_chat_names(self):
        with patched_env(
            {
                "AZURE_OPEN_AI_ENDPOINT_EMBEDDING": "https://embed.openai.azure.com/",
                "AZURE_OPEN_AI_KEY_EMBEDDING": "embedding-secret",
                "AZURE_OPENAI_ENDPOINT_CHAT": (
                    "https://chat.cognitiveservices.azure.com/openai/responses"
                    "?api-version=2025-04-01-preview"
                ),
                "AZURE_OPEN_AI_KEY_CHAT": "chat-secret",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
                "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-5.4-mini-gc",
            },
            clear=(
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPEN_AI_ENDPOINT",
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPEN_AI_KEY",
                "AZURE_OPENAI_CHAT_ENDPOINT",
                "AZURE_OPENAI_CHAT_API_KEY",
            ),
        ):
            config = get_azure_openai_config(require_chat=True)

        self.assertEqual(config["endpoint"], "https://embed.openai.azure.com")
        self.assertEqual(
            config["chat_endpoint"],
            "https://chat.cognitiveservices.azure.com",
        )
        self.assertEqual(config["api_key"], "embedding-secret")
        self.assertEqual(config["chat_api_key"], "chat-secret")
        self.assertEqual(config["chat_api_version"], "2025-04-01-preview")


if __name__ == "__main__":
    unittest.main()
