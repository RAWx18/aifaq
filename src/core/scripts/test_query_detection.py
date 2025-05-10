#!/usr/bin/env python3
"""
Test script for comparative query detection.
"""

import sys
import os
import asyncio
import re

# Add the core directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simple query detection without loading the full agent
def determine_query_type(query: str) -> str:
    """
    Determine the type of query using patterns.
    """
    query_lower = query.lower()
    
    # Check for question patterns
    if re.search(r'\bhow\b.+\bcompare\b|\bcompare\b|\bdifference\b|\bdistinguish\b|\bversus\b|\bvs\b|\bsimilar\b|\bdifferent\b', query_lower):
        return "comparative"
    elif re.search(r'\bhow\b.+\bdo\b|\bhow\b.+\bcan\b|\bhow\b.+\bto\b', query_lower):
        return "procedural"
    elif re.search(r'\bhow\b|\bexplain\b|\bdescribe\b|\belaborate\b', query_lower):
        return "explanatory"
    elif re.search(r'\bwhat\s+is\b|\bdefine\b|\bmeaning\s+of\b|\bdefinition\b', query_lower):
        return "definitional"
        return "definitional"
    elif re.search(r'\bwhy\b|\bcause\b|\breason\b', query_lower):
        return "causal"
    elif re.search(r'\bwhen\b|\bwhere\b|\bwho\b|\bwhich\b', query_lower):
        return "factual"
    elif re.search(r'\blist\b|\bname\b|\bgive\b.+\bexamples\b', query_lower):
        return "enumerative"
    elif re.search(r'\badvantages\b|\bbenefits\b|\bdrawbacks\b|\blimitations\b', query_lower):
        return "evaluative"
    else:
        return "general"

def test_query_detection():
    """Test query detection for comparative queries."""
    
    print("Testing query type detection...")
    
    # Test different query types
    test_queries = [
        ("What is Hyperledger Fabric?", "definitional"),
        ("How do I install Hyperledger Fabric?", "procedural"),
        ("How does Hyperledger Fabric compare to Ethereum?", "comparative"),
        ("Compare Hyperledger Fabric and Ethereum", "comparative"),
        ("What's the difference between Fabric and Ethereum?", "comparative"),
        ("Explain the endorsement process in Hyperledger Fabric", "explanatory"),
        ("Why is Hyperledger Fabric permissioned?", "causal")
    ]
    
    all_pass = True
    
    for query, expected_type in test_queries:
        # Process the query
        detected_type = determine_query_type(query)
        
        # Check the result
        status = "✓" if detected_type == expected_type else "✗"
        print(f"{status} Query: '{query}'")
        print(f"  Expected: {expected_type}, Detected: {detected_type}")
        
        if detected_type != expected_type:
            print(f"  FAIL: Query type detection failed!")
            all_pass = False
    
    print("\nTesting completed.")
    
    if all_pass:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")

if __name__ == "__main__":
    print("Starting query detection test...")
    test_query_detection()
