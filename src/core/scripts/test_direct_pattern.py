#!/usr/bin/env python3
"""
Direct test for query detection in the agent.
"""

import re

def test_direct_pattern():
    """Test the pattern matching directly."""
    
    print("Testing direct pattern matching...")
    
    query = "How does Hyperledger Fabric compare to Ethereum?"
    query_lower = query.lower()
    
    # Define patterns
    comparative_pattern = r'\bhow\b.+\bcompare\b|\bcompare\b|\bdifference\b|\bdistinguish\b|\bversus\b|\bvs\b|\bsimilar\b|\bdifferent\b'
    procedural_pattern = r'\bhow\b.+\bdo\b|\bhow\b.+\bcan\b|\bhow\b.+\bto\b'
    
    # Check matches
    comparative_match = bool(re.search(comparative_pattern, query_lower))
    procedural_match = bool(re.search(procedural_pattern, query_lower))
    
    print(f"Query: '{query}'")
    print(f"  Lower case: '{query_lower}'")
    print(f"  Pattern match for comparative: {comparative_match}")
    print(f"  Pattern match for procedural: {procedural_match}")
    
    # Print additional debug information
    if re.search(r'\bhow\b', query_lower):
        print("  Match for 'how': YES")
    else:
        print("  Match for 'how': NO")
        
    if re.search(r'\bcompare\b', query_lower):
        print("  Match for 'compare': YES")
    else:
        print("  Match for 'compare': NO")
        
    if re.search(r'\bhow\b.+\bcompare\b', query_lower):
        print("  Match for 'how...compare': YES")
    else:
        print("  Match for 'how...compare': NO")
        
    # Try with a different pattern order
    print("\nTrying a different detection order...")
    
    # First check procedural
    if re.search(procedural_pattern, query_lower):
        query_type = "procedural"
    # Then check comparative
    elif re.search(comparative_pattern, query_lower):
        query_type = "comparative"
    else:
        query_type = "other"
        
    print(f"  Detection result with procedural-first: {query_type}")
    
    # First check comparative
    if re.search(comparative_pattern, query_lower):
        query_type = "comparative"
    # Then check procedural
    elif re.search(procedural_pattern, query_lower):
        query_type = "procedural"
    else:
        query_type = "other"
        
    print(f"  Detection result with comparative-first: {query_type}")

if __name__ == "__main__":
    print("Starting direct pattern testing...")
    test_direct_pattern()
