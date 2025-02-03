# src/rag/embedder.py
import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from .base import DocumentChunk

load_dotenv()


class Embedder:
    def __init__(self, model_name: str | None = os.getenv('EMBEDDING_MODEL')):
        # def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        # def __init__(self, model_name: str = "jxm/cde-small-v2"): BAAI/bge-small-en-v1.5
        # def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Convert text chunks to embeddings"""
        texts = [chunk.text for chunk in chunks]
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed search query"""
        return self.model.encode(query, convert_to_numpy=True)


# Test with:
# embedder = Embedder()
# embeddings = embedder.embed_chunks(chunks)
# print(f"Embeddings shape: {embeddings.shape}")
