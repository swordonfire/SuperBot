# conftest.py (create if missing)
import pytest

from src.rag.vector_store import VectorStore


@pytest.fixture(autouse=True)
def clean_chroma_after_tests():
    """Clean ChromaDB before and after each test"""
    VectorStore().client.reset()
