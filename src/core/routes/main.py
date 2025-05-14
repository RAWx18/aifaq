from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import uuid
from conversation import initialize_models, generate_response, initialize_guardrails
from multi_agent_rag import create_multi_agent_rag
from guardrails import GuardrailProcessor, GuardrailConfig
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models and guardrails
model, tokenizer, vectordb, guardrails_processor = initialize_models()  # Changed here to unpack 4 values
# Initialize the multi-agent RAG system
multi_agent_system = create_multi_agent_rag(model, tokenizer, vectordb)
# No need to initialize guardrails separately, we now get it from initialize_models()
# guardrails_processor = initialize_guardrails()  # Comment out or remove this line

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
        # Apply guardrails to the query first
        should_process, custom_response = guardrails_processor.check_query(item.content)
        if not should_process:
            logger.info(f"Query blocked by guardrails: {item.content}")
            # Return the custom response without further processing
            return ResponseQuery(
                id=item.id,
                message=ResponseMessage(
                    content=custom_response,
                    type=1,
                    id=str(uuid.uuid4()),
                ),
            )
        
        # If query passes guardrails, process normally
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
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/multi-agent", response_model=DetailedResponseQuery)
async def answer_query_multi_agent(item: RequestQuery) -> DetailedResponseQuery:
    """
    Generate a response using the multi-agent RAG system.
    
    This endpoint processes the query through the full multi-agent pipeline,
    providing a more contextually relevant and evaluated response.
    """
    try:
        # Apply guardrails to the query first
        should_process, custom_response = guardrails_processor.check_query(item.content)
        if not should_process:
            logger.info(f"Multi-agent query blocked by guardrails: {item.content}")
            # Return the custom response without further processing
            return DetailedResponseQuery(
                id=item.id,
                message=ResponseMessage(
                    content=custom_response,
                    type=1,
                    id=str(uuid.uuid4()),
                ),
                metadata={"guardrails_blocked": True}
            )
            
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
        logger.error(f"Error processing multi-agent query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Simple endpoint for direct text queries (for curl testing)
@router.post("/text-query")
async def text_query(request: Request):
    try:
        data = await request.json()
        query_text = data.get("text", "")
        session_id = data.get("session_id", "default")
        
        # Apply guardrails to the query first
        should_process, custom_response = guardrails_processor.check_query(query_text)
        if not should_process:
            logger.info(f"Text query blocked by guardrails: {query_text}")
            # Return the custom response without further processing
            return {"response": custom_response}
        
        # If query passes guardrails, process normally
        response = generate_response(session_id, model, tokenizer, query_text, vectordb)
        
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing text query: {e}")
        raise HTTPException(status_code=500, detail=str(e))