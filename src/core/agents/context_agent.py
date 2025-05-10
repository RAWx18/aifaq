from typing import Dict, Any, List
from .base_agent import BaseAgent
import re


class ContextIntegrationAgent(BaseAgent):
    """
    Agent responsible for integrating retrieved context with conversation history.
    
    This agent:
    1. Combines retrieved documents with relevant conversation history
    2. Organizes the context in a coherent way
    3. Prioritizes information based on relevance
    4. Prepares the final context for the response generation
    """
    
    def __init__(self, name: str = "context_agent", max_context_length: int = 4000):
        super().__init__(name)
        self.max_context_length = max_context_length
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate retrieved documents with conversation history.
        
        Args:
            input_data: A dictionary containing:
                - retrieved_documents: Documents from the RetrievalAgent
                - relevance_scores: Scores for each document
                - query: The original user query
                - chat_history: The conversation history (optional)
                - query_type: The type of query being asked
                - key_terms: Important terms extracted from the query
            
        Returns:
            A dictionary containing:
                - integrated_context: The final context for response generation
                - context_summary: A summary of the integrated context (optional)
        """
        # Get the necessary data
        documents = input_data.get("retrieved_documents", [])
        scores = input_data.get("relevance_scores", [])
        query = input_data.get("query", "")
        chat_history = input_data.get("chat_history", [])
        query_type = input_data.get("query_type", "general")
        key_terms = input_data.get("key_terms", [])
        
        # Match documents with scores, sorted by relevance
        if len(scores) > 0:
            doc_score_pairs = sorted(
                zip(documents, scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            documents = [doc for doc, _ in doc_score_pairs]
        
        # Determine how much context to include based on the query type
        # Different query types may benefit from different context structures
        context_templates = {
            "explanatory": self._create_explanatory_context,
            "definitional": self._create_definitional_context,
            "comparative": self._create_comparative_context,
            "causal": self._create_causal_context,
            "factual": self._create_factual_context,
            "procedural": self._create_procedural_context,
            "enumerative": self._create_enumerative_context,
            "evaluative": self._create_evaluative_context,
            "general": self._create_general_context
        }
        
        # Get the appropriate context creation function
        context_fn = context_templates.get(query_type, self._create_general_context)
        
        # Create the context
        integrated_context = context_fn(query, documents, chat_history, key_terms)
        
        # Create a brief summary of the context
        context_summary = self._summarize_context(integrated_context, query_type)
        
        return {
            "integrated_context": integrated_context,
            "context_summary": context_summary
        }
    
    def _create_general_context(self, query: str, documents: List[str], 
                               chat_history: List[Dict[str, Any]], key_terms: List[str]) -> str:
        """Create a general-purpose context combining documents and history."""
        # Start with the most relevant information
        context_parts = []
        
        # Add relevant documents, prioritizing by relevance
        for i, doc in enumerate(documents):
            if i == 0:
                context_parts.append(f"Primary information related to the query:\n{doc}")
            else:
                context_parts.append(f"Additional information:\n{doc}")
        
        # Add recent and relevant history if available
        relevant_history = self._filter_relevant_history(chat_history, key_terms)
        if relevant_history:
            history_text = "\n".join([f"User: {h['user']}\nAnswer: {h['assistant']}" 
                                     for h in relevant_history[-2:]])
            context_parts.append(f"Relevant conversation history:\n{history_text}")
        
        # Combine all parts
        integrated_context = "\n\n".join(context_parts)
        
        # Ensure the context isn't too long
        integrated_context = self._trim_context(integrated_context)
        
        return integrated_context
    
    def _create_explanatory_context(self, query: str, documents: List[str], 
                                   chat_history: List[Dict[str, Any]], key_terms: List[str]) -> str:
        """Create a context optimized for explanatory queries."""
        context_parts = []
        
        # For explanations, we want comprehensive information
        background_info = []
        detailed_info = []
        
        for i, doc in enumerate(documents):
            if i < 2:  # First two documents are considered most relevant
                detailed_info.append(doc)
            else:
                background_info.append(doc)
        
        if detailed_info:
            context_parts.append(f"Detailed explanation:\n{' '.join(detailed_info)}")
        
        if background_info:
            context_parts.append(f"Background information:\n{' '.join(background_info[:3])}")  # Limit background
        
        # Add any relevant previous explanations from history
        explanation_history = self._filter_explanatory_history(chat_history, key_terms)
        if explanation_history:
            history_text = "\n".join([f"Previous explanation on this topic:\n{h['assistant']}" 
                                     for h in explanation_history[:1]])
            context_parts.append(history_text)
        
        integrated_context = "\n\n".join(context_parts)
        return self._trim_context(integrated_context)
    
    def _create_definitional_context(self, query: str, documents: List[str], 
                                    chat_history: List[Dict[str, Any]], key_terms: List[str]) -> str:
        """Create a context optimized for definitional queries."""
        # For definitions, we want precise information, not too verbose
        if not documents:
            return ""
        
        # Start with the most relevant definition
        primary_def = documents[0]
        
        # Add context for the definition if available
        additional_context = []
        for doc in documents[1:3]:  # Limit to next 2 documents
            additional_context.append(doc)
        
        # Construct the context
        context_parts = [f"Definition:\n{primary_def}"]
        
        if additional_context:
            context_parts.append(f"Additional context:\n{' '.join(additional_context)}")
        
        # Check if there were previous questions about this definition
        previous_definitions = self._filter_definitional_history(chat_history, key_terms)
        if previous_definitions:
            prev_def = f"Previously provided definition:\n{previous_definitions[0]['assistant']}"
            context_parts.append(prev_def)
        
        integrated_context = "\n\n".join(context_parts)
        return self._trim_context(integrated_context)
    
    def _create_comparative_context(self, query: str, documents: List[str], 
                                   chat_history: List[Dict[str, Any]], key_terms: List[str]) -> str:
        """Create a context optimized for comparative queries."""
        # For comparisons, we need information about all entities being compared
        context_parts = []
        
        # Organize documents that might contain information about each entity
        comparison_docs = {}
        for term in key_terms:
            term_docs = []
            for doc in documents:
                if term.lower() in doc.lower():
                    term_docs.append(doc)
            if term_docs:
                comparison_docs[term] = term_docs
        
        # If we found documents for specific terms
        if comparison_docs:
            for term, docs in comparison_docs.items():
                context_parts.append(f"Information about {term}:\n{' '.join(docs[:2])}")
        else:
            # Otherwise just use the most relevant documents
            for i, doc in enumerate(documents[:4]):
                context_parts.append(f"Comparison information {i+1}:\n{doc}")
        
        integrated_context = "\n\n".join(context_parts)
        return self._trim_context(integrated_context)
    
    def _create_causal_context(self, query: str, documents: List[str], 
                              chat_history: List[Dict[str, Any]], key_terms: List[str]) -> str:
        """Create a context optimized for causal (why) queries."""
        # For causal queries, we need explanations of causes and effects
        context_parts = []
        
        # Look for cause-effect language in documents
        causal_docs = []
        general_docs = []
        
        causal_indicators = ["because", "since", "as a result", "therefore", "consequently", 
                            "due to", "leads to", "causes", "effect of", "impact of"]
        
        for doc in documents:
            if any(indicator in doc.lower() for indicator in causal_indicators):
                causal_docs.append(doc)
            else:
                general_docs.append(doc)
        
        if causal_docs:
            context_parts.append(f"Causal explanation:\n{' '.join(causal_docs[:3])}")
        
        if general_docs:
            context_parts.append(f"Related information:\n{' '.join(general_docs[:2])}")
        
        integrated_context = "\n\n".join(context_parts)
        return self._trim_context(integrated_context)
    
    def _create_factual_context(self, query: str, documents: List[str], 
                               chat_history: List[Dict[str, Any]], key_terms: List[str]) -> str:
        """Create a context optimized for factual queries."""
        # For factual queries, we want precise information
        # Often, less context is better as long as it's accurate
        if not documents:
            return ""
        
        context_parts = []
        
        # Just use the top 2-3 most relevant documents
        for i, doc in enumerate(documents[:3]):
            if i == 0:
                context_parts.append(f"Key facts:\n{doc}")
            else:
                context_parts.append(f"Additional facts:\n{doc}")
        
        integrated_context = "\n\n".join(context_parts)
        return self._trim_context(integrated_context)
    
    def _create_procedural_context(self, query: str, documents: List[str], 
                                  chat_history: List[Dict[str, Any]], key_terms: List[str]) -> str:
        """Create a context optimized for procedural (how-to) queries."""
        context_parts = []
        
        # Look for step-by-step instructions in the documents
        step_indicators = ["step", "first", "then", "next", "finally", "1.", "2.", "3.", "-", "*"]
        procedural_docs = []
        other_docs = []
        
        for doc in documents:
            if any(indicator in doc.lower() for indicator in step_indicators):
                procedural_docs.append(doc)
            else:
                other_docs.append(doc)
        
        if procedural_docs:
            context_parts.append(f"Step-by-step instructions:\n{' '.join(procedural_docs[:2])}")
        
        if other_docs:
            context_parts.append(f"Additional guidance:\n{' '.join(other_docs[:2])}")
        
        integrated_context = "\n\n".join(context_parts)
        return self._trim_context(integrated_context)
    
    def _create_enumerative_context(self, query: str, documents: List[str], 
                                   chat_history: List[Dict[str, Any]], key_terms: List[str]) -> str:
        """Create a context optimized for list or enumeration queries."""
        context_parts = []
        
        # Look for lists in the documents
        list_docs = []
        other_docs = []
        
        # Check for list patterns
        list_patterns = [r"\d+\.", r"•", r"\*", r"-\s", r"\[.+\]", r"\(.+\)"]
        
        for doc in documents:
            if any(re.search(pattern, doc) for pattern in list_patterns):
                list_docs.append(doc)
            else:
                other_docs.append(doc)
        
        if list_docs:
            context_parts.append(f"List items:\n{' '.join(list_docs[:3])}")
        
        if other_docs:
            context_parts.append(f"Additional information:\n{' '.join(other_docs[:2])}")
        
        integrated_context = "\n\n".join(context_parts)
        return self._trim_context(integrated_context)
    
    def _create_evaluative_context(self, query: str, documents: List[str], 
                                  chat_history: List[Dict[str, Any]], key_terms: List[str]) -> str:
        """Create a context optimized for evaluative queries."""
        context_parts = []
        
        # Look for evaluative language in documents
        eval_indicators = ["advantage", "disadvantage", "benefit", "drawback", "pro", "con", 
                         "better", "best", "worse", "worst", "good", "bad", "recommend"]
        
        eval_docs = []
        other_docs = []
        
        for doc in documents:
            if any(indicator in doc.lower() for indicator in eval_indicators):
                eval_docs.append(doc)
            else:
                other_docs.append(doc)
        
        if eval_docs:
            context_parts.append(f"Evaluation information:\n{' '.join(eval_docs[:3])}")
        
        if other_docs:
            context_parts.append(f"Additional context:\n{' '.join(other_docs[:2])}")
        
        integrated_context = "\n\n".join(context_parts)
        return self._trim_context(integrated_context)
    
    def _filter_relevant_history(self, chat_history: List[Dict[str, Any]], key_terms: List[str]) -> List[Dict[str, Any]]:
        """Filter chat history to find entries relevant to the current query."""
        if not chat_history or not key_terms:
            return []
        
        relevant_history = []
        
        # Check each history entry for relevant terms
        for entry in chat_history:
            # Get the text from both user and assistant
            entry_text = f"{entry.get('user', '')} {entry.get('assistant', '')}"
            
            # Check if any key term is in the entry
            if any(term.lower() in entry_text.lower() for term in key_terms):
                relevant_history.append(entry)
        
        return relevant_history[:3]  # Limit to the 3 most recent relevant entries
    
    def _filter_explanatory_history(self, chat_history: List[Dict[str, Any]], key_terms: List[str]) -> List[Dict[str, Any]]:
        """Filter for previous explanations in chat history."""
        explanation_queries = ["explain", "how", "what is", "describe"]
        relevant_history = self._filter_relevant_history(chat_history, key_terms)
        
        # Further filter for explanatory queries
        return [entry for entry in relevant_history 
                if any(eq in entry.get('user', '').lower() for eq in explanation_queries)]
    
    def _filter_definitional_history(self, chat_history: List[Dict[str, Any]], key_terms: List[str]) -> List[Dict[str, Any]]:
        """Filter for previous definitions in chat history."""
        definition_queries = ["what is", "define", "meaning of", "definition"]
        relevant_history = self._filter_relevant_history(chat_history, key_terms)
        
        # Further filter for definitional queries
        return [entry for entry in relevant_history 
                if any(dq in entry.get('user', '').lower() for dq in definition_queries)]
    
    def _trim_context(self, context: str) -> str:
        """Ensure the context doesn't exceed the maximum allowed length."""
        if len(context) <= self.max_context_length:
            return context
        
        # If it's too long, keep the beginning and end, trimming the middle
        keep_start = int(self.max_context_length * 0.6)  # Keep 60% from start
        keep_end = int(self.max_context_length * 0.4)    # Keep 40% from end
        
        start_text = context[:keep_start]
        end_text = context[-keep_end:]
        
        return f"{start_text}\n...[Content trimmed for length]...\n{end_text}"
    
    def _summarize_context(self, context: str, query_type: str) -> str:
        """Create a brief summary of the integrated context."""
        # Count the documents included
        doc_count = context.count("information:")
        
        # Create a simple summary based on query type
        type_summaries = {
            "explanatory": f"Prepared explanatory context with {doc_count} information sources",
            "definitional": f"Prepared definitional context with primary definition and {doc_count-1} supporting sources",
            "comparative": "Prepared comparative context with information for each entity",
            "causal": f"Prepared causal explanation context with {doc_count} information sources",
            "factual": f"Prepared factual context with {doc_count} relevant sources",
            "procedural": "Prepared step-by-step instructional context",
            "enumerative": "Prepared list-based context",
            "evaluative": "Prepared evaluative context with pros and cons",
            "general": f"Prepared general context with {doc_count} information sources"
        }
        
        return type_summaries.get(query_type, f"Prepared context with {doc_count} information sources")
