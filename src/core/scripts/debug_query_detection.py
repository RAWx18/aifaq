#!/usr/bin/env python3
"""
Debug script for query detection.
"""

import sys
import os
import re

# Add the core directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.query_agent import QueryUnderstandingAgent

async def test_actual_agent():
    """Test query detection with the actual agent."""
    
    print("Testing query detection with actual agent...")
    
    # Create the agent
    agent = QueryUnderstandingAgent(use_spacy=False)
    
    # Test different query types
    test_queries = [
        "What is Hyperledger Fabric?",
        "How do I install Hyperledger Fabric?",
        "How does Hyperledger Fabric compare to Ethereum?",
        "Compare Hyperledger Fabric and Ethereum",
        "What's the difference between Fabric and Ethereum?",
        "Explain the endorsement process in Hyperledger Fabric",
        "Why is Hyperledger Fabric permissioned?"
    ]
    
    for query in test_queries:
        # Process the query with the agent
        input_data = {"query": query}
        result = await agent.process(input_data)
        
        # Get the query type
        query_type = result["query_type"]
        
        # Check for pattern matches
        query_lower = query.lower()
        print(f"Query: '{query}'")
        print(f"  Detected: {query_type}")
        
        # Define patterns
        comparative_pattern = r'\bhow\b.+\bcompare\b|\bcompare\b|\bdifference\b|\bdistinguish\b|\bversus\b|\bvs\b|\bsimilar\b|\bdifferent\b'
        procedural_pattern = r'\bhow\b.+\bdo\b|\bhow\b.+\bcan\b|\bhow\b.+\bto\b'
        
        # Check matches
        comparative_match = bool(re.search(comparative_pattern, query_lower))
        procedural_match = bool(re.search(procedural_pattern, query_lower))
        
        print(f"  Pattern match for comparative: {comparative_match}")
        print(f"  Pattern match for procedural: {procedural_match}")
        print("")
    
    print("\nDebug completed.")

if __name__ == "__main__":
    import asyncio
    print("Starting query detection debug...")
    asyncio.run(test_actual_agent())
