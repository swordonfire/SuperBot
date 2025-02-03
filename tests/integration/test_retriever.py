'''import pytest
from src.rag.retriever import Retriever
from src.rag.chunker import Chunker

class TestRetriever:
    @pytest.fixture
    def retriever(self):
        return Retriever()

    @pytest.fixture
    def test_chunks(self):
        return [
            "Superteam Vietnam focuses on web3 education",
            "Community-driven initiatives in Southeast Asia"
        ]

    def test_add_and_query(self, retriever, test_chunks):
        # Test document insertion
        retriever.add_documents(test_chunks)

        # Test query
        results = retriever.query("web3 education")
        assert len(results) > 0, "No results returned"
        assert "web3" in results[0].lower(), "Relevant result not found"'''

import pytest

from src.rag.chunker import Chunker
from src.rag.retriever import Retriever


class TestRetriever:
    @pytest.fixture
    def retriever(self):
        return Retriever()

    @pytest.fixture
    def test_chunks(self):
        chunker = Chunker()
        sample_pdf = 'tests/data/sample.pdf'  # Replace with actual path
        chunks = chunker.process_pdf(sample_pdf)
        return chunks

    def test_add_and_query_with_chunking(self, retriever, test_chunks):
        retriever.add_documents(test_chunks)
        results = retriever.query('Steps of Research Process')
        assert len(results) > 0, 'No results returned'
        assert results[0].score >= 0.8, 'Top result should have high relevance'
        # Add more assertions based on expected retrieval behavior
