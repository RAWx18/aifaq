#!/usr/bin/env python3
"""
Simple test script to verify the multi-agent RAG system.
This script uses mocks to avoid downloading the large language model.
"""
import sys
import os
import asyncio
from unittest.mock import MagicMock

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the agent classes
from agents.query_agent import QueryUnderstandingAgent
from agents.retrieval_agent import RetrievalAgent
from agents.context_agent import ContextIntegrationAgent
from agents.response_agent import ResponseGenerationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.agent_coordinator import AgentCoordinator

async def test_multi_agent_coordination():
    print("Starting multi-agent coordination test...")
    
    # Create mocks of required components
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_vectordb = MagicMock()
    mock_vectordb.similarity_search.return_value = [
        MagicMock(page_content="Hyperledger Fabric is an enterprise blockchain platform.")
    ]
    
    # Initialize agents
    print("Initializing agents...")
    query_agent = QueryUnderstandingAgent()
    retrieval_agent = RetrievalAgent(mock_vectordb)
    context_agent = ContextIntegrationAgent()
    response_agent = ResponseGenerationAgent(mock_model, mock_tokenizer)
    evaluation_agent = EvaluationAgent()
    
    # Create coordinator
    coordinator = AgentCoordinator([
        query_agent,
        retrieval_agent,
        context_agent, 
        response_agent,
        evaluation_agent
    ])
    
    # Mock the response generation
    original_process = response_agent.process
    async def mock_process(data):
        result = await original_process(data)
        result["response"] = "Hyperledger Fabric is an enterprise blockchain platform for developing solutions with a modular architecture."
        return result
    response_agent.process = mock_process
    
    # Test with a sample query
    test_query = "What is Hyperledger Fabric?"
    print(f"Processing query: '{test_query}'")
    
    # Run the pipeline
    result = await coordinator.run_pipeline({
        "query": test_query,
        "session_id": "test-session",
        "chat_history": []
    })
    
    # Print results
    print("\nResponse:")
    print(result.get("response", "No response generated"))
    
    print("\nAgent Processing History:")
    for stage in result.get("processing_history", []):
        print(f"- {stage['agent']}")
    
    return "Test completed successfully"

if __name__ == "__main__":
    result = asyncio.run(test_multi_agent_coordination())
    print(f"\n{result}")
