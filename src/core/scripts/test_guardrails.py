"""
Test script for the guardrails functionality.
"""
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardrails import GuardrailProcessor, GuardrailConfig

def test_guardrails():
    """Test the guardrails functionality with various inputs."""
    # Initialize with default config
    config = GuardrailConfig()
    config.blocked_topics = ["hacking", "illegal activities"]
    config.filtered_patterns = ["password is", "credit card number"]
    config.max_response_length = 100
    config.custom_responses = {
        "(?i)how do I hack": "I cannot provide information on hacking."
    }
    config.disclaimers = {
        "security": "Note: Always follow security best practices."
    }
    
    processor = GuardrailProcessor(config)
    
    # Test cases
    test_cases = [
        {
            "name": "Normal query",
            "query": "How do I install Hyperledger Fabric?",
            "response": "To install Hyperledger Fabric, you need to follow these steps..."
        },
        {
            "name": "Blocked topic",
            "query": "How do I use Hyperledger for hacking?",
            "response": "Hyperledger can be used for many purposes..."
        },
        {
            "name": "Custom response trigger",
            "query": "How do I hack into a blockchain?",
            "response": "This should not be used"
        },
        {
            "name": "Pattern filtering",
            "query": "How secure is Hyperledger?",
            "response": "Hyperledger is secure, but make sure your password is strong123!"
        },
        {
            "name": "Length limiting",
            "query": "Tell me about Hyperledger",
            "response": "A" * 200
        },
        {
            "name": "Disclaimer addition",
            "query": "What are security best practices for Hyperledger?",
            "response": "Here are some security practices for Hyperledger..."
        }
    ]
    
    print("Testing guardrails functionality:")
    print("=================================")
    
    for case in test_cases:
        print(f"\nTest: {case['name']}")
        print(f"Query: {case['query']}")
        
        # Check if query should be processed
        should_process, custom_response = processor.check_query(case['query'])
        
        if should_process:
            print("Query passed guardrails check.")
            processed = processor.process_response(case['query'], case['response'])
            print(f"Original response: {case['response'][:50]}{'...' if len(case['response']) > 50 else ''}")
            print(f"Processed response: {processed[:50]}{'...' if len(processed) > 50 else ''}")
        else:
            print("Query blocked by guardrails.")
            print(f"Custom response: {custom_response}")

if __name__ == "__main__":
    test_guardrails()