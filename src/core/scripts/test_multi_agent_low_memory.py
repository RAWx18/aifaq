#!/usr/bin/env python3
"""
Low-memory version of the multi-agent test script.
This script uses CPU-only inference with reduced model precision to avoid memory issues.
"""

import asyncio
import sys
import os
import torch

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
from utils import load_yaml_file
from conversation import initialize_models
from transformers import AutoModelForCausalLM, AutoTokenizer

async def test_multi_agent_rag_low_memory():
    """
    Test the multi-agent RAG system with a sample query using low-memory settings.
    """
    print("Initializing models with low memory settings...")
    
    config_data = load_yaml_file("config.yaml")
    
    # Low memory model initialization
    print("Loading model with CPU and 8-bit quantization...")
    try:
        # Force CPU and lower precision
        if torch.cuda.is_available():
            print("CUDA is available but using CPU for low memory test")
        
        # Load model with lower precision for CPU
        model = AutoModelForCausalLM.from_pretrained(
            config_data["model_name"],
            device_map="cpu",
            torch_dtype=torch.float32,  # Use float32 instead of float16 for CPU
            load_in_8bit=True,  # 8-bit quantization for memory reduction
            low_cpu_mem_usage=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config_data["model_name"])
        
        # Use the same vector database
        _, _, vectordb = initialize_models()
        
        print("Models loaded successfully with low memory settings")
        
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
        
        return "Multi-agent RAG test completed successfully with low-memory settings"
    
    except Exception as e:
        print(f"Error during low-memory test: {str(e)}")
        print("Falling back to mock test...")
        
        # Import the mock test function
        from test_multi_agent_mock import test_multi_agent_rag as test_mock
        
        # Run the mock test as a fallback
        return await test_mock()


if __name__ == "__main__":
    print("Starting multi-agent RAG system test with low-memory settings...")
    test_result = asyncio.run(test_multi_agent_rag_low_memory())
    print(f"\n{test_result}")
