#!/usr/bin/env python3
# filepath: /home/raw/Documents/workspace/lfx/aifaq/src/core/scripts/test_multi_agent_mock.py

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
from multi_agent_rag import MultiAgentRAG

class MockVectorDB:
    def similarity_search(self, query, k=3):
        # Create mock documents
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Hyperledger Fabric is an enterprise-grade permissioned distributed ledger framework."
        
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Hyperledger Fabric uses a modular architecture with components like peers, orderers, and channels."
        
        mock_doc3 = MagicMock()
        mock_doc3.page_content = "Hyperledger Fabric supports smart contracts called chaincode that can be written in languages like Go, Node.js, and Java."
        
        return [mock_doc1, mock_doc2, mock_doc3]
    
    def similarity_search_with_score(self, query, k=3):
        # Return documents with similarity scores
        docs = self.similarity_search(query, k)
        # Return pairs of (document, score)
        return [(doc, 0.85 - i * 0.1) for i, doc in enumerate(docs)]

async def test_multi_agent_rag():
    """
    Test the multi-agent RAG system with a sample query using mock components.
    """
    print("Setting up mock components...")
    
    # Set up mock model and tokenizer
    mock_model = MagicMock()
    mock_model.generate.return_value = [1, 2, 3, 4, 5]
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "Hyperledger Fabric is an enterprise-grade permissioned blockchain platform. It uses a modular architecture with components such as peers, orderers, and channels. The platform supports smart contracts called chaincode that can be written in various languages. Its permissioned nature means that participants have known identities, making it suitable for enterprise use cases where privacy and identity are important."
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    # Set up mock vector database
    mock_vectordb = MockVectorDB()
    
    print("Creating multi-agent RAG system...")
    multi_agent_system = MultiAgentRAG(mock_model, mock_tokenizer, mock_vectordb)
    
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
    
    return "Multi-agent RAG test completed successfully with mock components"


if __name__ == "__main__":
    print("Starting multi-agent RAG system test with mock components...")
    test_result = asyncio.run(test_multi_agent_rag())
    print(f"\n{test_result}")
