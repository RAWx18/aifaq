#!/usr/bin/env python3
"""
NLP Dependencies Setup Script.

This script ensures that all required NLP dependencies are installed,
including:
- NLTK data packages
- SpaCy models

Run this script after installing the Python packages via requirements.txt.
"""

import os
import sys
import subprocess
import nltk
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download required NLTK data packages."""
    logger.info("Downloading NLTK data packages...")
    
    nltk_packages = [
        'punkt',      # For tokenization
        'stopwords',  # Common stopwords
        'wordnet',    # For word meanings and relationships
        'averaged_perceptron_tagger'  # For part-of-speech tagging
    ]
    
    for package in nltk_packages:
        try:
            logger.info(f"Downloading NLTK package: {package}")
            nltk.download(package, quiet=False)
        except Exception as e:
            logger.error(f"Error downloading NLTK package {package}: {str(e)}")

def download_spacy_models():
    """Download required SpaCy models."""
    logger.info("Downloading SpaCy models...")
    
    spacy_models = [
        'en_core_web_sm'  # Small English model
    ]
    
    for model in spacy_models:
        try:
            logger.info(f"Checking for SpaCy model: {model}")
            # Try to import the model to see if it's already installed
            try:
                __import__(model)
                logger.info(f"SpaCy model {model} is already installed.")
            except ImportError:
                # Model not installed, download it
                logger.info(f"Downloading SpaCy model: {model}")
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
        except Exception as e:
            logger.error(f"Error installing SpaCy model {model}: {str(e)}")

def check_vectordb():
    """Check if the vector database exists."""
    logger.info("Checking vector database...")
    
    chromadb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'chromadb')
    
    if os.path.exists(chromadb_path) and os.path.isdir(chromadb_path):
        db_files = os.listdir(chromadb_path)
        if db_files:
            logger.info(f"Vector database exists at {chromadb_path} with {len(db_files)} files.")
        else:
            logger.warning(f"Vector database directory exists at {chromadb_path}, but is empty.")
    else:
        logger.warning(f"Vector database directory does not exist at {chromadb_path}.")
        logger.info("Run the ingest.py script to create and populate the vector database.")

def main():
    """Run all setup tasks."""
    logger.info("Starting NLP dependencies setup...")
    
    download_nltk_data()
    download_spacy_models()
    check_vectordb()
    
    logger.info("NLP dependencies setup completed!")

if __name__ == "__main__":
    main()
