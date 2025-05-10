#!/usr/bin/env python3
# filepath: /home/raw/Documents/workspace/lfx/aifaq/src/core/scripts/test_agent_coordination.py
import asyncio
import sys
import os
from unittest.mock import MagicMock

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

async def test_agent_coordination():
    """
    Test the multi-agent coordination without requiring the full model.
    
    This function uses mocks for the model dependencies to test the agent coordination logic.
    """
    print("Testing agent coordination...")
    
    # Create mock agents
    query_agent = QueryUnderstandingAgent()
    
    # Mock the vectordb and other components
    mock_vectordb = MagicMock()
    mock_vectordb.similarity_search.return_value = [
        MagicMock(page_content="Hyperledger Fabric is an enterprise blockchain platform.")
    ]
    
    retrieval_agent = RetrievalAgent(mock_vectordb)
    context_agent = ContextIntegrationAgent()
    
    # Mock the model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    
    # Set up response generation to return a test response
    mock_response = "Hyperledger Fabric is an enterprise blockchain platform for developing solutions with a modular architecture."
    response_agent = ResponseGenerationAgent(mock_model, mock_tokenizer)
    
    # Patch the process method to return a mock response
    original_process = response_agent.process
    
    async def mock_process(data):
        result = await original_process(data)
        result["response"] = mock_response
        return result
    
    response_agent.process = mock_process
    
    evaluation_agent = EvaluationAgent()
    
    # Create the agent coordinator
    coordinator = AgentCoordinator([
        query_agent,
        retrieval_agent,
        context_agent,
        response_agent,
        evaluation_agent
    ])
    
    # Sample query
    test_query = "What is Hyperledger Fabric?"
    session_id = "test-session-1"
    
    # Prepare the initial input
    initial_input = {
        "query": test_query,
        "session_id": session_id,
        "chat_history": []
    }
    
    print(f"Processing query: '{test_query}'")
    result = await coordinator.run_pipeline(initial_input)
    
    print("\nResponse:")
    print(result.get("response", "No response generated"))
    
    print("\nAgent Processing History:")
    for stage in result.get("processing_history", []):
        print(f"- {stage['agent']}")
    
    return "Agent coordination test completed"


if __name__ == "__main__":
    print("Starting agent coordination test...")
    test_result = asyncio.run(test_agent_coordination())
    print(f"\n{test_result}")
