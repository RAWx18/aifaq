import asyncio
import sys
import os

# Add the core directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import (
    QueryUnderstandingAgent, 
    RetrievalAgent, 
    ContextIntegrationAgent, 
    ResponseGenerationAgent,
    EvaluationAgent,
    AgentCoordinator
)
from multi_agent_rag import create_multi_agent_rag
from conversation import initialize_models

async def test_multi_agent_rag():
    """
    Test the multi-agent RAG system with a sample query.
    
    This function initializes the models and multi-agent system, 
    then runs a sample query through the pipeline to verify functionality.
    """
    print("Initializing models...")
    model, tokenizer, vectordb = initialize_models()
    
    print("Creating multi-agent RAG system...")
    multi_agent_system = create_multi_agent_rag(model, tokenizer, vectordb)
    
    # Sample query
    test_query = "What is Hyperledger Fabric and how does it work?"
    session_id = "test-session-1"
    
    print(f"Processing query: '{test_query}'")
    result = await multi_agent_system.generate_response(test_query, session_id)
    
    print("\nResponse:")
    print(result["response"])
    
    print("\nMetadata:")
    for key, value in result["metadata"].items():
        print(f"- {key}: {value}")
    
    # Test a follow-up question to check if conversation history is used
    print("\nProcessing follow-up query...")
    follow_up_query = "What are its main components?"
    result = await multi_agent_system.generate_response(follow_up_query, session_id)
    
    print("\nFollow-up Response:")
    print(result["response"])
    
    return "Multi-agent RAG test completed"


if __name__ == "__main__":
    print("Starting multi-agent RAG system test...")
    test_result = asyncio.run(test_multi_agent_rag())
    print(f"\n{test_result}")
