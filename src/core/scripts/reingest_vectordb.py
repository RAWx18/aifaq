#!/usr/bin/env python3
"""
Vector Database Reingest Script.

This script cleans and reingests documents into the vector database.
"""

import sys
import os
import shutil
import argparse

# Add the core directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_yaml_file
from agents.logger import setup_logger

# Setup logging
logger = setup_logger(name="vectordb_reingest", level="info")

def reingest_documents(force=False):
    """
    Clean and reingest documents into the vector database.
    
    Args:
        force: If True, proceed without confirmation
    """
    config_data = load_yaml_file("config.yaml")
    vectordb_path = config_data.get("persist_directory", "chromadb")
    
    # Check if vectordb path exists
    if os.path.exists(vectordb_path):
        if not force:
            confirm = input(f"This will delete the existing vector database at '{vectordb_path}'. Continue? (y/n): ")
            if confirm.lower() != 'y':
                logger.info("Operation cancelled")
                return
        
        logger.info(f"Removing existing vector database at '{vectordb_path}'")
        try:
            shutil.rmtree(vectordb_path)
            logger.info("Existing database removed successfully")
        except Exception as e:
            logger.error(f"Error removing existing database: {str(e)}")
            return
    
    # Run the ingest script
    logger.info("Starting document ingestion...")
    ingest_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ingest.py")
    
    try:
        # Execute the ingest script
        os.system(f"python3 {ingest_script}")
        logger.info("Document ingestion completed")
    except Exception as e:
        logger.error(f"Error during document ingestion: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reingest documents into the vector database")
    parser.add_argument("--force", action="store_true", help="Force reingestion without confirmation")
    args = parser.parse_args()
    
    logger.info("Starting vector database reingestion...")
    reingest_documents(args.force)
    logger.info("Vector database reingestion process completed")
