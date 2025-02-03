# src/rag/base.py
from typing import List

from pydantic import BaseModel


class DocumentChunk(BaseModel):
    text: str
    metadata: dict  # {source: "file.pdf", page: 3, ...}


class RetrievedResult(BaseModel):
    chunks: List[DocumentChunk]
    scores: List[float]
    query: str
