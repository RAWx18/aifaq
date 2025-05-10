import os
import torch
from utils import load_yaml_file
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from embeddings import embedding_function
from langchain_community.vectorstores.utils import filter_complex_metadata

config_data = load_yaml_file("config.yaml")

persist_directory = config_data["persist_directory"]
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
loader_web = RecursiveUrlLoader(url=config_data["url"])

loader_rtdocs = ReadTheDocsLoader(config_data["folder_path"], encoding="utf-8")

loader = MergedDataLoader(loaders=[loader_web, loader_rtdocs])

embeddings = embedding_function()

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(f"Number of chunks created: {len(splits)}")

# Process in smaller batches to avoid memory issues
batch_size = 20  # Reduced batch size
total_batches = (len(splits) - 1) // batch_size + 1
print(f"Processing documents in {total_batches} batches of {batch_size} documents each")

for i in range(0, len(splits), batch_size):
    batch = splits[i:i+batch_size]
    current_batch = i // batch_size + 1
    print(f"Processing batch {current_batch}/{total_batches} ({len(batch)} documents)")
    
    try:
        if i == 0:
            # Create vectorstore with first batch
            print("Creating new vectorstore...")
            vectorstore = Chroma.from_documents(
                filter_complex_metadata(batch), embeddings, persist_directory=persist_directory
            )
        else:
            # Add subsequent batches
            print("Adding documents to existing vectorstore...")
            vectorstore.add_documents(filter_complex_metadata(batch))
        
        # Persist after each batch
        vectorstore.persist()
        print(f"Batch {current_batch}/{total_batches} completed and persisted.")
    except Exception as e:
        print(f"Error processing batch {current_batch}: {str(e)}")
        # Continue with next batch

print("Vectorstore creation and persistence completed.")
