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
    loader = PyPDFLoader(path)
    return loader.load()


def split_chunk(document):
    def get_chapter(page_number):
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
    return text.encode("utf-8", "ignore").decode("utf-8")


def split_clean_chunks(filtered_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(filtered_docs)

    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)
    return chunks
