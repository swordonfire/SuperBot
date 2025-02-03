from fastapi import APIRouter, HTTPException
from llama_cpp import Llama
from pydantic import BaseModel

from src.llm.engine import get_llm_engine

router = APIRouter()
llm_engine = get_llm_engine()


class PromptRequest(BaseModel):
    prompt: str


@router.post('/generate')
async def generate_text(request: PromptRequest):
    try:
        model: Llama = llm_engine.model
        response = model.create_chat_completion(
            messages=[{'role': 'user', 'content': request.prompt}]
        )
        return {'response': response['choices'][0]['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
