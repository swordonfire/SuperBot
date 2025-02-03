# src/rag/pipeline.py
from ..llm.engine import generate_response  # Now works with both access patterns
from .base import RetrievedResult
from .retriever import Retriever


class RAGPipeline:
    '''def __init__(self, retriever: Retriever, llm_prompt_template: str = None):
        self.retriever = retriever
        self.llm_prompt_template = llm_prompt_template or \
            "Context: {context}\n\nQuestion: {question}\nAnswer:"'''

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.llm_prompt_template = """Answer ONLY with facts from the context. 
        If unsure, say "I don't know".
        
        Context: {context}
        
        Question: {question}
        Answer:"""

    def generate_response(self, query: str) -> str:
        """End-to-end RAG pipeline"""
        result = self.retriever.retrieve(query)

        if not result.chunks:
            return (
                "I don't have enough information to answer that. Please contact a Superteam admin."
            )

        context = '\n\n'.join([chunk.text for chunk in result.chunks])
        prompt = self.llm_prompt_template.format(context=context, question=query)

        # Use your existing LLM integration here
        # from ..llm.engine import generate_response

        return generate_response(prompt)  # Direct function call

    def format_for_telegram(self, result: RetrievedResult) -> str:
        """Format results for Telegram bot output"""
        if not result.chunks:
            return "Sorry, I couldn't find relevant information."

        response = [
            f'🔍 Found {len(result.chunks)} relevant results:',
            *[f'📄 {chunk.text[:200]}...' for chunk in result.chunks],
            '\nPlease ask for details if you need more information!',
        ]
        return '\n\n'.join(response)


# Test with:
# pipeline = RAGPipeline(retriever)
# print(pipeline.generate_response("How to apply for grants?"))
