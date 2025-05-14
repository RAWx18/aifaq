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
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_data = load_yaml_file("config.yaml")

def initialize_models():
    """Initialize models, tokenizer, vectordb, and guardrails."""
    model = load_quantized_model(config_data["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config_data["model_name"])
    embeddings = HuggingFaceEmbeddings(model_name=config_data["embedding_model_name"])
    
    # Use the correct path for ChromaDB - hardcoded to ensure it works
    chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromadb")
    logger.info(f"Using ChromaDB path: {chroma_path}")
    
    # Ensure directory exists
    os.makedirs(chroma_path, exist_ok=True)
    
    vectordb = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )
    
    # Initialize guardrails with singleton pattern
    if not hasattr(initialize_models, '_guardrails'):
        initialize_models._guardrails = initialize_guardrails()
    
    return model, tokenizer, vectordb, initialize_models._guardrails

def initialize_guardrails(config_path=None):
    """Initialize the guardrails processor with optional custom config path."""
    config = GuardrailConfig()
    if config_path:
        try:
            config.load_from_file(config_path)
            logger.info(f"Loaded guardrails configuration from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Guardrails config file not found at {config_path}. Using default configuration.")
            config._set_default_config()
    else:
        # Try to load from default locations
        default_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/guardrails.yaml"),
            "config/guardrails.yaml",  # Relative path
            "src/core/config/guardrails.yaml"  # Another common path
        ]
        
        config_loaded = False
        for path in default_paths:
            try:
                config.load_from_file(path)
                logger.info(f"Loaded guardrails configuration from {path}")
                config_loaded = True
                break
            except FileNotFoundError:
                continue
        
        if not config_loaded:
            logger.warning("No guardrails config file found. Using default configuration.")
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
            logger.warning(f"No documents found for query: {query}")
            return "No relevant documents were found in the knowledge base."
        
        context = "\n\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return "Unable to retrieve context from the knowledge base due to an error."

def generate_response(session_id, model, tokenizer, query, vectordb):
    """Generate a response with guardrails applied."""
    # Always use the singleton guardrails instance
    if not hasattr(initialize_models, '_guardrails'):
        initialize_models._guardrails = initialize_guardrails()
    guardrails_processor = initialize_models._guardrails
    
    # Apply guardrails to the query first
    should_process, custom_response = guardrails_processor.check_query(query)
    if not should_process:
        logger.info(f"Query blocked by guardrails: {query}")
        conversation_history = get_session_history(session_id)
        conversation_history.add_user_message(query)
        conversation_history.add_ai_message(custom_response)
        return custom_response
    
    conversation_history = get_session_history(session_id)
    context = retrieve_relevant_context(query, vectordb)
    logger.info(f"Retrieved context length: {len(context)}")

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
        print(token, end="", flush=True)
    print()
    
    # Apply guardrails to the generated response
    processed_response = guardrails_processor.process_response(query, response)
    
    # Store the processed response in conversation history
    conversation_history.add_user_message(query)
    conversation_history.add_ai_message(processed_response)
    
    return processed_response