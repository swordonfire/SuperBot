from pathlib import Path

import fitz
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    text: str
    metadata: dict


class Chunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, file_path: str | Path) -> list[DocumentChunk]:
        """Process PDFs into validated chunks"""

        path = Path(file_path)
        if not path.suffix.lower() == '.pdf':
            raise NotImplementedError('Only PDF files are supported')

        chunks = []

        try:
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text(sort=True)  # Sort by layout
                    if text:
                        chunks.append(
                            DocumentChunk(
                                text=text,
                                metadata={'source': Path(file_path).name, 'page': page_num + 1},
                            )
                        )
        except fitz.FileDataError as err:
            raise ValueError('Invalid or corrupted PDF file') from err

        return self._split_chunks(chunks)

    '''
    def _split_chunks(self, page_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Split pages into smaller chunks"""
        split_chunks = []
        for chunk in page_chunks:
            words = chunk.text.split()
            current_chunk = []
            current_len = 0
            
            for word in words:
                if current_len + len(word) + 1 > self.chunk_size:
                    split_chunks.append(DocumentChunk(
                        text=" ".join(current_chunk),
                        metadata=chunk.metadata
                    ))
                    current_chunk = []
                    current_len = 0
                
                current_chunk.append(word)
                current_len += len(word) + 1
            
            if current_chunk:
                split_chunks.append(DocumentChunk(
                    text=" ".join(current_chunk),
                    metadata=chunk.metadata
                ))
                
        return split_chunks'''

    def _split_chunks(self, page_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        split_chunks = []
        for original_chunk in page_chunks:
            words = original_chunk.text.split()
            current_chunk = []
            current_len = 0
            chunk_num = 1  # Track chunk number per page

            for word in words:
                if current_len + len(word) + 1 > self.chunk_size:
                    split_chunks.append(
                        DocumentChunk(
                            text=' '.join(current_chunk),
                            metadata={**original_chunk.metadata, 'chunk_num': chunk_num},
                        )
                    )
                    current_chunk = []
                    current_len = 0
                    chunk_num += 1

                current_chunk.append(word)
                current_len += len(word) + 1

            if current_chunk:
                split_chunks.append(
                    DocumentChunk(
                        text=' '.join(current_chunk),
                        metadata={**original_chunk.metadata, 'chunk_num': chunk_num},
                    )
                )

        return split_chunks
