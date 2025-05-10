#!/usr/bin/env python3
"""
Vector Database Diagnostics Script.

This script checks the health and content of the vector database.
"""

import sys
import os
import json

# Add the core directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_yaml_file
from agents.logger import setup_logger
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Setup logging
logger = setup_logger(name="vectordb_check", level="info")

def initialize_vectordb_only():
    """Initialize only the vector database without loading the full model."""
    config_data = load_yaml_file("config.yaml")
    persist_directory = config_data.get("persist_directory", "chromadb")
    embedding_model_name = config_data.get("embedding_model_name", "sentence-transformers/all-mpnet-base-v2")
    
    logger.info(f"Loading embedding model: {embedding_model_name}")
    
    # Load embedding model (smaller memory footprint)
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Load vector database
        logger.info(f"Loading vector database from: {persist_directory}")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        
        return vectordb
    except Exception as e:
        logger.error(f"Error initializing vector database: {str(e)}")
        return None

def check_vectordb_content():
    """Check the content of the vector database."""
    logger.info("Initializing vector database...")
    vectordb = initialize_vectordb_only()
    
    if not vectordb:
        logger.error("Failed to initialize vector database")
        return
    
    # Check if the vectordb has collections
    logger.info("Checking vector database collections...")
    
    try:
        # Check the number of documents in the default collection
        doc_count = len(vectordb.get()['ids'])
        logger.info(f"Vector database contains {doc_count} documents")
        
        if doc_count == 0:
            logger.warning("Vector database is empty. Run ingest.py to populate it.")
            return
        
        # Sample some documents to verify content
        logger.info("Sampling documents from the vector database:")
        sample_query = "Hyperledger"
        results = vectordb.similarity_search(sample_query, k=3)
        
        if not results:
            logger.warning(f"No documents found for sample query '{sample_query}'")
        else:
            for i, doc in enumerate(results):
                logger.info(f"Document {i+1}:")
                logger.info(f"Content (truncated): {doc.page_content[:100]}...")
                logger.info(f"Metadata: {doc.metadata}")
                logger.info("---")
        
        # Test various queries to check retrieval
        test_queries = [
            "What is Hyperledger?",
            "How does consensus work?",
            "Chaincode",
            "Security in Hyperledger"
        ]
        
        logger.info("\nTesting retrieval with sample queries:")
        for query in test_queries:
            results = vectordb.similarity_search_with_score(query, k=2)
            logger.info(f"Query: '{query}'")
            logger.info(f"Found {len(results)} documents")
            
            if results:
                # Show the top result's content and score
                doc, score = results[0]
                logger.info(f"Top result score: {score}")
                logger.info(f"Top result content (truncated): {doc.page_content[:100]}...")
            
            logger.info("---")
        
    except Exception as e:
        logger.error(f"Error checking vector database: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting vector database diagnostics...")
    check_vectordb_content()
    logger.info("Vector database diagnostics completed")
