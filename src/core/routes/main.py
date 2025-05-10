from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
from conversation import initialize_models, generate_response
from multi_agent_rag import create_multi_agent_rag
import asyncio

model, tokenizer, vectordb = initialize_models()
# Initialize the multi-agent RAG system
multi_agent_system = create_multi_agent_rag(model, tokenizer, vectordb)

router = APIRouter()


class ResponseMessage(BaseModel):
    content: str
    type: int
    id: str


class RequestQuery(BaseModel):
    id: str
    content: str


class ResponseQuery(BaseModel):
    id: str
    message: ResponseMessage


class DetailedResponseQuery(ResponseQuery):
    metadata: dict = None


@router.post("/query", response_model=ResponseQuery)
async def answer_query(item: RequestQuery) -> ResponseQuery:
    try:
        response = generate_response("1", model, tokenizer, item.content, vectordb)
        return ResponseQuery(
            id=item.id,
            message=ResponseMessage(
                content=response,
                type=1,
                id=str(uuid.uuid4()),
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/multi-agent", response_model=DetailedResponseQuery)
async def answer_query_multi_agent(item: RequestQuery) -> DetailedResponseQuery:
    """
    Generate a response using the multi-agent RAG system.
    
    This endpoint processes the query through the full multi-agent pipeline,
    providing a more contextually relevant and evaluated response.
    """
    try:
        # Generate response using the multi-agent system
        result = await multi_agent_system.generate_response(
            query=item.content,
            session_id=item.id
        )
        
        return DetailedResponseQuery(
            id=item.id,
            message=ResponseMessage(
                content=result["response"],
                type=1,
                id=str(uuid.uuid4()),
            ),
            metadata=result.get("metadata", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
