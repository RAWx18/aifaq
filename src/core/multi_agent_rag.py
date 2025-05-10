from typing import Dict, Any, List, Optional
import os
import asyncio
from langchain_community.vectorstores import Chroma

from agents import (
    AgentCoordinator,
    QueryUnderstandingAgent,
    RetrievalAgent,
    ContextIntegrationAgent,
    ResponseGenerationAgent,
    EvaluationAgent
)


class MultiAgentRAG:
    """
    Integrates the multi-agent system for Retrieval Augmented Generation.
    
    This class provides a high-level interface to the multi-agent RAG system,
    handling the coordination between agents and providing a simple API for
    generating responses to user queries.
    """
    
    def __init__(self, model, tokenizer, vectordb: Chroma, session_history: Optional[List[Dict]] = None):
        """
        Initialize the multi-agent RAG system.
        
        Args:
            model: The language model for response generation
            tokenizer: The tokenizer for the language model
            vectordb: The vector database for document retrieval
            session_history: Optional conversation history
        """
        self.model = model
        self.tokenizer = tokenizer
        self.vectordb = vectordb
        self.session_history = session_history or []
        
        # Initialize the agents
        self.query_agent = QueryUnderstandingAgent()
        self.retrieval_agent = RetrievalAgent(vectordb)
        self.context_agent = ContextIntegrationAgent()
        self.response_agent = ResponseGenerationAgent(model, tokenizer)
        self.evaluation_agent = EvaluationAgent()
        
        # Create the agent coordinator
        self.coordinator = AgentCoordinator([
            self.query_agent,
            self.retrieval_agent,
            self.context_agent,
            self.response_agent,
            self.evaluation_agent
        ])
        
    async def generate_response(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Generate a response to a user query using the multi-agent system.
        
        Args:
            query: The user's query
            session_id: The ID of the conversation session
            
        Returns:
            A dictionary containing the response and metadata
        """
        # Prepare the initial input for the pipeline
        initial_input = {
            "query": query,
            "session_id": session_id,
            "chat_history": self.session_history
        }
        
        # Run the multi-agent pipeline
        result = await self.coordinator.run_pipeline(initial_input)
        
        # Extract the final response
        response = result.get("response", "")
        
        # Update session history
        self.session_history.append({
            "user": query,
            "assistant": response
        })
        
        # Return the final result
        return {
            "response": response,
            "metadata": {
                "query_understanding": {
                    "query_type": result.get("query_type"),
                    "key_terms": result.get("key_terms")
                },
                "retrieval": {
                    "document_count": len(result.get("retrieved_documents", [])),
                },
                "evaluation": result.get("evaluation_scores", {}),
                "processing_stages": [stage["agent"] for stage in result.get("processing_history", [])]
            }
        }


# Factory function to create a MultiAgentRAG instance
def create_multi_agent_rag(model, tokenizer, vectordb, session_history=None):
    """Create a MultiAgentRAG instance."""
    return MultiAgentRAG(model, tokenizer, vectordb, session_history)
