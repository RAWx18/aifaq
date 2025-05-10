#!/usr/bin/env python3
"""
Comprehensive Test Suite for the Multi-Agent RAG System.

This script provides comprehensive testing for all aspects of the multi-agent RAG system:
- Agent coordination
- Document retrieval
- Response generation
- Query understanding
- Error handling
"""

import asyncio
import sys
import os
import json
from datetime import datetime
import traceback

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
from agents.logger import setup_logger

# Setup logging
logger = setup_logger(name="comprehensive_test", level="info")

# Test cases covering various query types and scenarios
TEST_CASES = [
    {
        "name": "Basic definition query",
        "query": "What is Hyperledger Fabric?",
        "expected_query_type": "definitional",
        "min_doc_count": 1
    },
    {
        "name": "How-to query",
        "query": "How do I install Hyperledger Fabric?",
        "expected_query_type": "procedural",
        "min_doc_count": 1
    },
    {
        "name": "Comparison query",
        "query": "How does Hyperledger Fabric compare to Ethereum?",
        "expected_query_type": "comparative",
        "min_doc_count": 2
    },
    {
        "name": "Conceptual query",
        "query": "Explain the endorsement process in Hyperledger Fabric",
        "expected_query_type": "explanatory",
        "min_doc_count": 1
    },
    {
        "name": "Unknown topic query",
        "query": "What is the theory of relativity?",
        "expected_query_type": None,  # We don't care about the query type for this test
        "min_doc_count": 0  # Expect no relevant documents
    },
    {
        "name": "Malformed query",
        "query": "hlf?????",
        "expected_query_type": None,
        "min_doc_count": 0
    },
    {
        "name": "Multi-part query",
        "query": "What are channels in Hyperledger Fabric and how are they used?",
        "expected_query_type": None,
        "min_doc_count": 1
    }
]

class TestResult:
    """Class to store test results."""
    
    def __init__(self, test_case, success=False, query_type=None, doc_count=0, 
                 response_length=0, duration=0, error=None):
        self.test_case = test_case
        self.success = success
        self.query_type = query_type
        self.doc_count = doc_count
        self.response_length = response_length
        self.duration = duration
        self.error = error
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "test_name": self.test_case["name"],
            "query": self.test_case["query"],
            "success": self.success,
            "query_type": self.query_type,
            "expected_query_type": self.test_case.get("expected_query_type"),
            "doc_count": self.doc_count,
            "min_doc_count": self.test_case.get("min_doc_count", 0),
            "response_length": self.response_length,
            "duration": self.duration,
            "error": self.error
        }

async def run_tests(mode="mock"):
    """
    Run all test cases.
    
    Args:
        mode: Test mode - "mock" or "full"
    """
    logger.info(f"Running comprehensive tests in {mode} mode")
    
    # Initialize system based on mode
    try:
        if mode == "mock":
            # Import and use the mock test function for setup
            from test_multi_agent_mock import test_multi_agent_rag as setup_mock
            multi_agent_system = None  # Will be created later
        else:
            # Use the real system with actual models
            from conversation import initialize_models
            logger.info("Initializing models...")
            model, tokenizer, vectordb = initialize_models()
            logger.info("Creating multi-agent RAG system...")
            multi_agent_system = create_multi_agent_rag(model, tokenizer, vectordb)
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return []
    
    results = []
    
    # Run each test case
    for test_case in TEST_CASES:
        logger.info(f"\nRunning test: {test_case['name']}")
        logger.info(f"Query: '{test_case['query']}'")
        
        try:
            start_time = datetime.now()
            
            if mode == "mock":
                # For mock mode, we need to create a fresh system for each test
                # because we haven't imported MultiAgentRAG directly
                from test_multi_agent_mock import MockVectorDB
                from unittest.mock import MagicMock
                
                # Set up mock components
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_tokenizer.decode.return_value = f"Mock response for: {test_case['query']}"
                mock_tokenizer.pad_token_id = 0
                mock_tokenizer.eos_token_id = 0
                
                # Import the MultiAgentRAG class
                from multi_agent_rag import MultiAgentRAG
                multi_agent_system = MultiAgentRAG(mock_model, mock_tokenizer, MockVectorDB())
            
            # Process the query
            result = await multi_agent_system.generate_response(
                test_case["query"], f"test-session-{hash(test_case['name'])}"
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Extract data for validation
            query_type = result["metadata"]["query_understanding"]["query_type"]
            doc_count = result["metadata"]["retrieval"]["document_count"]
            response = result["response"]
            
            # Validate results
            success = True
            
            # Check query type if expected is specified
            if test_case.get("expected_query_type") and query_type != test_case["expected_query_type"]:
                logger.warning(f"Query type mismatch. Expected: {test_case['expected_query_type']}, Got: {query_type}")
                success = False
            
            # Check minimum document count
            if doc_count < test_case.get("min_doc_count", 0):
                logger.warning(f"Document count below minimum. Expected at least: {test_case.get('min_doc_count', 0)}, Got: {doc_count}")
                success = False
            
            # Check if response was generated
            if not response or len(response) < 10:
                logger.warning(f"Response too short or empty: '{response}'")
                success = False
            
            # Create test result
            test_result = TestResult(
                test_case=test_case,
                success=success,
                query_type=query_type,
                doc_count=doc_count,
                response_length=len(response),
                duration=duration
            )
            
            status = "PASSED" if success else "FAILED"
            logger.info(f"Test {status} in {duration:.2f}s")
            logger.info(f"Query type: {query_type}, Documents: {doc_count}")
            
        except Exception as e:
            logger.error(f"Error during test: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Create failed test result
            test_result = TestResult(
                test_case=test_case,
                success=False,
                error=str(e)
            )
            
            logger.info("Test FAILED due to error")
        
        results.append(test_result)
    
    return results

def save_results(results, mode):
    """Save test results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{mode}_{timestamp}.json"
    
    # Convert results to dictionaries
    results_dict = [result.to_dict() for result in results]
    
    # Calculate summary statistics
    total_tests = len(results)
    passed_tests = sum(1 for result in results if result.success)
    
    summary = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
        "avg_duration": sum(result.duration for result in results) / total_tests if total_tests > 0 else 0,
        "timestamp": timestamp,
        "mode": mode
    }
    
    # Create final output
    output = {
        "summary": summary,
        "results": results_dict
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nTest results saved to {filename}")
    
    # Print summary
    logger.info("\nTest Summary:")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed tests: {passed_tests}")
    logger.info(f"Success rate: {summary['success_rate'] * 100:.1f}%")
    logger.info(f"Average duration: {summary['avg_duration']:.2f}s")
    
    return filename

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive tests for the multi-agent RAG system")
    parser.add_argument("--mode", choices=["mock", "full"], default="mock",
                      help="Test mode - 'mock' for mock components or 'full' for full system")
    args = parser.parse_args()
    
    logger.info(f"Starting comprehensive tests in {args.mode} mode...")
    
    try:
        results = await run_tests(args.mode)
        save_results(results, args.mode)
        
        # Print a summary to stdout
        total = len(results)
        passed = sum(1 for r in results if r.success)
        print(f"\nTest Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        # Print failed tests
        if passed < total:
            print("\nFailed tests:")
            for r in results:
                if not r.success:
                    print(f"- {r.test_case['name']}: {r.test_case['query']}")
                    if r.error:
                        print(f"  Error: {r.error}")
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info("Comprehensive testing completed")

if __name__ == "__main__":
    asyncio.run(main())
