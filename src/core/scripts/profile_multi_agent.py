#!/usr/bin/env python3
"""
Performance profiling for the multi-agent RAG system.

This script measures performance metrics for the multi-agent system:
- Overall response time
- Time spent in each agent
- Memory usage during processing
- Document retrieval effectiveness

Usage:
    python3 profile_multi_agent.py
"""

import asyncio
import sys
import os
import time
import tracemalloc
import json
import numpy as np
from datetime import datetime

# Add the core directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock
from multi_agent_rag import MultiAgentRAG
from agents.logger import setup_logger

# Setup logging
logger = setup_logger(name="profiler", level="info")

# Sample queries for testing
SAMPLE_QUERIES = [
    "What is Hyperledger Fabric?",
    "How does consensus work in Hyperledger Fabric?",
    "What are the key components of Hyperledger Fabric?",
    "How does Hyperledger Fabric compare to Ethereum?",
    "Can you explain the endorsement process in Hyperledger Fabric?",
    "What programming languages can be used with Hyperledger Fabric?",
    "How do I install Hyperledger Fabric?",
    "What is the difference between channels and chaincodes?",
    "What security features does Hyperledger Fabric provide?",
    "How does Hyperledger Fabric handle private data?"
]

class ProfilerResults:
    """Class to store and analyze profiling results."""
    
    def __init__(self):
        self.queries = []
        self.total_times = []
        self.agent_times = {}
        self.memory_usages = []
        self.doc_counts = []
        self.errors = []
    
    def add_result(self, query, total_time, agent_times, memory_usage, doc_count, error=None):
        """Add a result from a single query run."""
        self.queries.append(query)
        self.total_times.append(total_time)
        
        # Add agent times
        for agent, time_taken in agent_times.items():
            if agent not in self.agent_times:
                self.agent_times[agent] = []
            self.agent_times[agent].append(time_taken)
        
        self.memory_usages.append(memory_usage)
        self.doc_counts.append(doc_count)
        
        if error:
            self.errors.append({"query": query, "error": error})
    
    def get_summary(self):
        """Get a summary of the profiling results."""
        summary = {
            "total_queries": len(self.queries),
            "avg_response_time": np.mean(self.total_times) if self.total_times else 0,
            "max_response_time": np.max(self.total_times) if self.total_times else 0,
            "min_response_time": np.min(self.total_times) if self.total_times else 0,
            "avg_memory_usage_mb": np.mean(self.memory_usages) / (1024 * 1024) if self.memory_usages else 0,
            "avg_doc_count": np.mean(self.doc_counts) if self.doc_counts else 0,
            "error_count": len(self.errors),
            "agent_times": {}
        }
        
        # Calculate statistics for each agent
        for agent, times in self.agent_times.items():
            summary["agent_times"][agent] = {
                "avg_time": np.mean(times) if times else 0,
                "max_time": np.max(times) if times else 0,
                "min_time": np.min(times) if times else 0,
                "percentage": (np.mean(times) / np.mean(self.total_times) * 100) if times and self.total_times else 0
            }
        
        return summary
    
    def save_to_file(self, filename):
        """Save the results to a JSON file."""
        results = {
            "summary": self.get_summary(),
            "detailed_results": [
                {
                    "query": query,
                    "response_time": time,
                    "memory_usage_mb": mem / (1024 * 1024),
                    "doc_count": doc
                }
                for query, time, mem, doc in zip(
                    self.queries, self.total_times, self.memory_usages, self.doc_counts
                )
            ],
            "errors": self.errors
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")

async def profile_mock_system():
    """Profile the multi-agent system with mock components."""
    logger.info("Setting up mock components for profiling...")
    
    # Create mock model, tokenizer, and vectordb
    mock_model = MagicMock()
    mock_model.generate.return_value = [1, 2, 3, 4, 5]
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "This is a mock response for profiling purposes."
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    # Create mock vector database with docs
    class MockVectorDB:
        def similarity_search_with_score(self, query, k=3):
            # Create mock documents
            mock_docs = []
            for i in range(k):
                doc = MagicMock()
                doc.page_content = f"Mock document {i} for query: {query}"
                doc.metadata = {"source": f"source{i}", "title": f"title{i}"}
                mock_docs.append((doc, 0.9 - (i * 0.1)))
            return mock_docs
    
    # Create the multiagent system
    multi_agent_system = MultiAgentRAG(mock_model, mock_tokenizer, MockVectorDB())
    
    # Create results container
    results = ProfilerResults()
    
    # Profile each query
    for query in SAMPLE_QUERIES:
        try:
            logger.info(f"Profiling query: '{query}'")
            
            # Start memory tracking
            tracemalloc.start()
            start_time = time.time()
            
            # Process the query
            result = await multi_agent_system.generate_response(query, "profile-session")
            
            # Record time and memory usage
            total_time = time.time() - start_time
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Extract agent times from the result
            agent_times = {}
            for stage in result.get("processing_history", []):
                agent_times[stage["agent"]] = stage.get("duration", 0)
            
            # Record document count
            doc_count = len(result.get("retrieved_documents", []))
            
            # Add to results
            results.add_result(query, total_time, agent_times, peak_mem, doc_count)
            
            logger.info(f"Query processed in {total_time:.2f}s, peak memory: {peak_mem / (1024 * 1024):.2f}MB")
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            tracemalloc.stop()
            results.add_result(query, 0, {}, 0, 0, str(e))
    
    # Save the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results.save_to_file(f"profiling_results_{timestamp}.json")
    
    # Print summary
    summary = results.get_summary()
    logger.info("\nProfiling Summary:")
    logger.info(f"Total queries: {summary['total_queries']}")
    logger.info(f"Average response time: {summary['avg_response_time']:.2f}s")
    logger.info(f"Average memory usage: {summary['avg_memory_usage_mb']:.2f}MB")
    logger.info(f"Average document count: {summary['avg_doc_count']:.2f}")
    logger.info(f"Errors: {summary['error_count']}")
    
    logger.info("\nAgent Timing Breakdown:")
    for agent, stats in summary["agent_times"].items():
        logger.info(f"  {agent}: {stats['avg_time']:.2f}s ({stats['percentage']:.1f}% of total time)")
    
    return "Profiling completed"

if __name__ == "__main__":
    logger.info("Starting multi-agent RAG system profiling...")
    result = asyncio.run(profile_mock_system())
    logger.info(result)
