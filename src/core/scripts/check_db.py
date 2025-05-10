#!/usr/bin/env python3
"""
Simple Vector Database Check Script.

This script checks the vector database without loading the large language model.
"""

import sys
import os
import sqlite3

# Add the core directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_yaml_file

def check_vectordb_sqlite():
    """Check the vector database directly using SQLite."""
    print("Checking vector database using SQLite...")
    
    # Load config
    config_data = load_yaml_file("config.yaml")
    persist_directory = config_data.get("persist_directory", "chromadb")
    
    # SQLite database file path
    db_path = os.path.join(persist_directory, "chroma.sqlite3")
    
    if not os.path.exists(db_path):
        print(f"ERROR: Vector database file not found at {db_path}")
        return False
    
    try:
        # Connect to SQLite database
        print(f"Connecting to SQLite database at {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Database tables: {[table[0] for table in tables]}")
        
        # Check embeddings table
        if ('embeddings',) in tables:
            cursor.execute("SELECT COUNT(*) FROM embeddings;")
            count = cursor.fetchone()[0]
            print(f"Number of embeddings: {count}")
            
            if count == 0:
                print("WARNING: No embeddings found in the database!")
                return False
                
            # Check if we have document content in the fulltext search table
            if ('embedding_fulltext_search',) in tables:
                cursor.execute("SELECT COUNT(*) FROM embedding_fulltext_search;")
                doc_count = cursor.fetchone()[0]
                print(f"Number of document contents: {doc_count}")
                
                # Sample some document contents
                cursor.execute("SELECT string_value FROM embedding_fulltext_search LIMIT 3;")
                samples = cursor.fetchall()
                print("\nSample documents:")
                for i, (doc_content,) in enumerate(samples):
                    print(f"Document {i+1}:")
                    # Truncate document content if too long
                    if len(doc_content) > 200:
                        print(f"Content (truncated): {doc_content[:200]}...")
                    else:
                        print(f"Content: {doc_content}")
                    print("---")
                
                conn.close()
                return count > 0 and doc_count > 0
            else:
                print("ERROR: No 'embedding_fulltext_search' table found in the database!")
                return False
        else:
            print("ERROR: No 'embeddings' table found in the database!")
            return False
            
    except Exception as e:
        print(f"ERROR checking database: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting simple vector database check...")
    
    if check_vectordb_sqlite():
        print("\nVector database check completed successfully. Documents found in the database.")
    else:
        print("\nVector database check failed. The database may be empty or corrupted.")
        print("Run the ingest.py script to populate the vector database.")
