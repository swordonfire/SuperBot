# src/rag/retriever.py
from typing import List

from rank_bm25 import BM25Okapi

from .base import DocumentChunk, RetrievedResult


class Retriever:
    def __init__(self, vector_store, embedder, hybrid_search: bool = True):
        self.vector_store = vector_store
        self.embedder = embedder
        self.hybrid_search = hybrid_search

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.5) -> RetrievedResult:
        """Hybrid search with confidence threshold"""
        query_embedding = self.embedder.embed_query(query)

        # Vector search
        vector_results = self.vector_store.search(query_embedding, top_k)

        if self.hybrid_search:
            # Add keyword search (BM25) results
            keyword_results = self._bm25_search(query, top_k)
            combined = self._combine_results(vector_results, keyword_results)
        else:
            combined = vector_results

        # Filter by score threshold
        filtered = [res for res in combined if res['score'] >= score_threshold]

        return RetrievedResult(
            chunks=[DocumentChunk(text=res['text'], metadata=res['metadata']) for res in filtered],
            scores=[res['score'] for res in filtered],
            query=query,
        )

    '''
    def _bm25_search(self, query: str, top_k: int) -> List[dict]:
        """Simple keyword matching (to be replaced with proper BM25)"""
        # MVP implementation - search document texts directly
        all_docs = self.vector_store.collection.get()["documents"]
        matches = [doc for doc in all_docs if query.lower() in doc.lower()]
        # return [{"text": doc, "metadata": {}, "score": 0.7} for doc in matches[:top_k]]

        return [{
            "text": doc,
            "metadata": self.vector_store.collection.get(ids=[id])["metadatas"][0] or {},
            "score": 0.7
        } for doc, id in zip(matches[:top_k], ids)]'''

    '''
    def _bm25_search(self, query: str, top_k: int) -> List[dict]:
        """Keyword search with metadata fallback"""
        # Get all documents and their metadata
        all_docs = self.vector_store.collection.get()
        documents = all_docs["documents"]
        metadatas = all_docs["metadatas"]
    
        # Simple keyword match
        matches = []
        for idx, doc in enumerate(documents):
            if query.lower() in doc.lower():
                matches.append({
                    "text": doc,
                    "metadata": metadatas[idx] if metadatas else {},
                    "score": 0.7  # Fixed score for MVP
                })
    
        return sorted(matches, key=lambda x: x["score"], reverse=True) '''

    def _bm25_search(self, query: str, top_k: int) -> List[dict]:
        """Proper BM25 implementation with metadata"""
        collection = self.vector_store.collection.get()
        documents = collection['documents']
        metadatas = collection['metadatas'] or [{}] * len(documents)

        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        # Search
        tokenized_query = query.lower().split()
        doc_scores = bm25.get_scores(tokenized_query)

        # Combine with metadata
        results = []
        for idx, score in enumerate(doc_scores):
            results.append({'text': documents[idx], 'metadata': metadatas[idx], 'score': score})

        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]

    '''
    def _combine_results(self, vector_results, keyword_results):
        """Simple reciprocal rank fusion"""
        combined = {}
        for i, res in enumerate(vector_results):
            combined[res["text"]] = res["score"] + (1 / (i + 1))
        for i, res in enumerate(keyword_results):
            if res["text"] in combined:
                combined[res["text"]] += res["score"] + (1 / (i + 1))
            else:
                combined[res["text"]] = res["score"] + (1 / (i + 1))
        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [{"text": k, "score": v} for k, v in sorted_items]
    '''

    '''
    # src/rag/retriever.py
    def _combine_results(self, vector_results, keyword_results):
        """Combine results while preserving metadata"""
        combined = {}
    
        # Track metadata from both sources
        for res in vector_results:
            key = res["text"]
            combined[key] = {
                "score": res["score"] + 1/(combined.get(key, {}).get("index", 0) + 1),
                "metadata": res["metadata"],
                "index": 0
            }
    
        for res in keyword_results:
            key = res["text"]
            if key in combined:
                combined[key]["score"] += res["score"] + 1/(combined[key]["index"] + 1)
                combined[key]["index"] += 1
            else:
                combined[key] = {
                    "score": res["score"],
                    "metadata": res["metadata"],
                    "index": 0
                }
    
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )
    
        return [{
            "text": res["metadata"]["source"],  # Use original text from vector store
            "score": res["score"],
            "metadata": res["metadata"]
        } for res in sorted_results]'''

    '''
    def _combine_results(self, vector_results, keyword_results):
        """Normalize scores between 0-1 before combining"""
        vector_scores = [res["score"] for res in vector_results]
        max_vector = max(vector_scores) if vector_scores else 1
        keyword_scores = [res["score"] for res in keyword_results]
        max_keyword = max(keyword_scores) if keyword_scores else 1
    
        return [
            {
                "text": res["text"],
                "score": (res["score"]/max_vector * 0.6) + (res["score"]/max_keyword * 0.4),
                "metadata": res["metadata"]
            }
            for res in vector_results + keyword_results 
        ]
    '''

    def _combine_results(self, vector_results, keyword_results):
        """Safe score combination with clamping"""
        vector_scores = [res['score'] for res in vector_results]
        max_vector = max(vector_scores) if vector_scores else 1e-9  # Prevent zero-division
        keyword_scores = [res['score'] for res in keyword_results]
        max_keyword = max(keyword_scores) if keyword_scores else 1e-9

        combined = []
        for res in vector_results:
            score = res['score'] / max_vector * 0.6
            combined.append({**res, 'score': min(max(score, 0), 1)})  # Clamp 0-1

        for res in keyword_results:
            score = res['score'] / max_keyword * 0.4
            combined.append({**res, 'score': min(max(score, 0), 1)})

        return sorted(combined, key=lambda x: x['score'], reverse=True)


# Test with:
# retriever = Retriever(vector_store, embedder)
# results = retriever.retrieve("What is Superteam Vietnam?")
# print(results.chunks[0].text)
