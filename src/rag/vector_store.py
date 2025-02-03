# src/rag/vector_store.py
from pathlib import Path
from typing import List

import chromadb
import numpy as np

from .base import DocumentChunk


class VectorStore:
    def __init__(self, persist_dir: str = 'data/vector_db/chroma'):
        # Create directory with proper permissions
        self.persist_path = Path(persist_dir).absolute()
        self.persist_path.mkdir(parents=True, exist_ok=True, mode=0o777)

        # self.client = chromadb.PersistentClient(path=persist_dir)
        # Initialize Chroma with explicit settings
        self.client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=chromadb.config.Settings(allow_reset=True, is_persistent=True),
        )

        self.collection = self.client.get_or_create_collection('superteam_docs')

    '''
    def add_documents(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Store documents in ChromaDB"""
        ids = [f"{ch.metadata['source']}-{ch.metadata['page']}-{ch.metadata['chunk_num']}" 
              for ch in chunks]
        metadatas = [ch.metadata for ch in chunks]
        documents = [ch.text for ch in chunks]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )'''

    def add_documents(self, chunks: list[DocumentChunk], embeddings: np.ndarray):
        """Store DocumentChunk objects properly"""
        ids = [
            f'{ch.metadata["source"]}-p{ch.metadata["page"]}-c{ch.metadata["chunk_num"]}'
            for ch in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=[ch.text for ch in chunks],
            metadatas=[ch.metadata for ch in chunks],
            embeddings=embeddings.tolist(),
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """Vector similarity search"""
        results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)
        return self._format_results(results)

    def _format_results(self, raw_results: dict) -> List[dict]:
        """Convert Chroma results to standardized format"""
        return [
            {'text': doc, 'metadata': meta, 'score': score}
            for doc, meta, score in zip(
                raw_results['documents'][0],
                raw_results['metadatas'][0],
                raw_results['distances'][0],
                strict=False,
            )
        ]


# Test with:
# vector_store = VectorStore()
# vector_store.add_documents(chunks, embeddings)
# results = vector_store.search(query_embedding)
