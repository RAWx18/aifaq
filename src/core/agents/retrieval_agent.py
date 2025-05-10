from typing import Dict, Any, List, Tuple, Optional
from .base_agent import BaseAgent
from langchain_community.vectorstores import Chroma
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant documents from the vector database.
    
    This agent:
    1. Retrieves candidate documents based on the query
    2. Ranks documents by relevance
    3. Filters out low-quality or irrelevant documents
    4. Returns a set of high-quality context documents
    """
    
    def __init__(self, vectordb: Chroma, name: str = "retrieval_agent"):
        super().__init__(name)
        self.vectordb = vectordb
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve and rank relevant documents for the query.
        
        Args:
            input_data: A dictionary containing:
                - expanded_query: The processed query from the QueryUnderstandingAgent
                - query_type: The type of query being asked
                - key_terms: Important terms from the query
            
        Returns:
            A dictionary containing:
                - retrieved_documents: List of relevant documents
                - source_documents: Information about the source of each document
                - context_text: Combined text from all relevant documents
                - relevance_scores: Scores indicating document relevance
        """
        # Get the query information
        query = input_data.get("expanded_query", input_data.get("query", ""))
        query_type = input_data.get("query_type", "general")
        key_terms = input_data.get("key_terms", [])
        
        logger.info(f"RetrievalAgent processing query: '{query}' (type: {query_type})")
        logger.info(f"Key terms: {key_terms}")
        
        # Retrieve documents from vector database
        # The k value can be dynamically adjusted based on query type
        k_values = {
            "explanatory": 5,  # More context for explanations
            "definitional": 3,  # Fewer, more precise documents for definitions
            "comparative": 6,   # More documents for comparisons
            "causal": 4,        # Several documents for causal explanations
            "factual": 3,       # Fewer documents for factual queries
            "general": 4        # Default number of documents
        }
        
        # Increase k to improve chances of finding relevant documents
        k = k_values.get(query_type, 4) * 2  # Double the initial k value
        
        # First try with expanded query
        results = self._try_retrieval(query, k)
        
        # If no results, try with key terms
        if not results and key_terms:
            logger.info(f"No results with expanded query, trying with key terms...")
            key_terms_query = " ".join(key_terms)
            results = self._try_retrieval(key_terms_query, k)
        
        # If still no results, try to increase k
        if not results:
            logger.info(f"No results with k={k}, trying with increased k...")
            results = self._try_retrieval(query, k * 2)
        
        # Process and rank the results
        documents = []
        scores = []
        sources = []
        
        for doc, score in results:
            documents.append(doc.page_content)
            scores.append(float(score))  # Convert to float for serialization
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", "Unknown")
            })
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Advanced ranking could be implemented here
        ranked_indices = self._rank_documents(documents, scores, key_terms)
        
        # Reorder documents, scores, and sources based on the ranking
        documents = [documents[i] for i in ranked_indices]
        scores = [scores[i] for i in ranked_indices]
        sources = [sources[i] for i in ranked_indices]
        
        # Combine documents into a single context text
        context_text = "\n\n".join(documents)
        
        return {
            "retrieved_documents": documents,
            "source_documents": sources,
            "context_text": context_text,
            "relevance_scores": scores
        }
    
    def _try_retrieval(self, query: str, k: int = 4) -> List[Tuple]:
        """
        Try to retrieve documents with error handling.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Use the vector database to retrieve documents
            results = self.vectordb.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def _rank_documents(self, documents: List[str], scores: List[float], key_terms: List[str]) -> List[int]:
        """
        Rank documents based on relevance scores and key term presence.
        
        Args:
            documents: List of document texts
            scores: Initial relevance scores from vector similarity
            key_terms: Important terms from the query
            
        Returns:
            List of indices representing the ranked order of documents
        """
        # This is a simple ranking that weighs vector similarity and key term presence
        # More sophisticated ranking could be implemented
        
        # Create a copy of the initial scores
        final_scores = scores.copy()
        
        # Boost scores based on key term presence
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            term_count = sum(1 for term in key_terms if term in doc_lower)
            
            # Boost the score based on term presence
            # This is a simple heuristic that can be refined
            term_boost = term_count * 0.05
            final_scores[i] += term_boost
        
        # Get indices that would sort the array in descending order
        ranked_indices = np.argsort(final_scores).tolist()
        ranked_indices.reverse()  # Highest scores first
        
        return ranked_indices
