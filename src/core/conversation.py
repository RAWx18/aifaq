import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from model import load_quantized_model
from utils import load_yaml_file
from session_history import get_session_history
from guardrails import GuardrailProcessor, GuardrailConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_data = load_yaml_file("config.yaml")

def initialize_models():
    """Initialize models, tokenizer, vectordb, and guardrails."""
    model = load_quantized_model(config_data["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config_data["model_name"])
    embeddings = HuggingFaceEmbeddings(model_name=config_data["embedding_model_name"])
    
    persist_directory = config_data.get("persist_directory", "chromadb")
    
    # Ensure directory exists
    import os
    os.makedirs(persist_directory, exist_ok=True)
    
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # Initialize guardrails with singleton pattern
    if not hasattr(initialize_models, '_guardrails'):
        initialize_models._guardrails = initialize_guardrails()
    
    return model, tokenizer, vectordb, initialize_models._guardrails

def initialize_guardrails(config_path="config/guardrails.yaml"):
    """Initialize the guardrails processor."""
    config = GuardrailConfig()
    try:
        config.load_from_file(config_path)
        logger.info(f"Loaded guardrails configuration from {config_path}")
    except FileNotFoundError:
        logger.warning(f"Guardrails config file not found at {config_path}. Using default configuration.")
        config._set_default_config()
    
    logger.info(f"Guardrails initialized with blocked topics: {config.blocked_topics}")
    logger.info(f"Max response length: {config.max_response_length}")
    
    return GuardrailProcessor(config)

def retrieve_relevant_context(query, vectordb, top_k=3):
    """Retrieve relevant context from the vector database."""
    try:
        results = vectordb.similarity_search(query, k=top_k)
        logger.info(f"Retrieved {len(results)} documents for query: {query}")
        
        if not results:
            return "No relevant documents were found in the knowledge base."
        
        context = "\n\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return "Unable to retrieve context from the knowledge base due to an error."

def generate_response(session_id, model, tokenizer, query, vectordb):
    """Generate a response with guardrails applied."""
    # Use the singleton guardrails instance
    guardrails_processor = initialize_models._guardrails
    
    # Apply guardrails to the query
    should_process, custom_response = guardrails_processor.check_query(query)
    if not should_process:
        conversation_history = get_session_history(session_id)
        conversation_history.add_user_message(query)
        conversation_history.add_ai_message(custom_response)
        return custom_response
    
    conversation_history = get_session_history(session_id)
    context = retrieve_relevant_context(query, vectordb)

    qa_system_prompt = """You are a concise assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Provide a very concise answer in no more than three short sentences."""

    full_prompt = f"{qa_system_prompt}\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"

    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True).to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = dict(
        inputs, streamer=streamer, max_new_tokens=1000, do_sample=True, temperature=0.7
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    response = ""
    for token in streamer:
        response += token
        print(token)
    
    # Apply guardrails to the generated response
    processed_response = guardrails_processor.process_response(query, response)
    
    # Store the processed response in conversation history
    conversation_history.add_user_message(query)
    conversation_history.add_ai_message(processed_response)
    
    return processed_response