from typing import Dict, Any, List
from .base_agent import BaseAgent
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from collections import Counter


class QueryUnderstandingAgent(BaseAgent):
    """
    Agent responsible for analyzing and understanding user queries.
    
    This agent processes the raw user query to:
    1. Identify the main intent
    2. Extract key entities and concepts
    3. Determine the query type (factual, explanatory, etc.)
    4. Expand the query with related terms if needed
    """
    
    def __init__(self, name: str = "query_agent", use_spacy: bool = True):
        super().__init__(name)
        self.use_spacy = use_spacy
        
        # Initialize NLP resources as needed
        try:
            # Download NLTK resources if not present
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english'))
            
            # Load spaCy model if available
            if self.use_spacy:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except:
                    self.nlp = None
                    self.use_spacy = False
                    print("SpaCy model not available. Falling back to basic NLP.")
        except:
            self.stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "and", "or", "of", "is", "are"}
            self.use_spacy = False
            print("NLTK resources not available. Using basic stopwords.")
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the user query and enhance it for better retrieval.
        
        Args:
            input_data: A dictionary containing at least the 'query' key with the raw user query.
            
        Returns:
            A dictionary with enhanced query information, including:
            - analyzed_query: The processed query
            - query_type: The detected type of query
            - key_terms: Important terms extracted from the query
            - expanded_query: Optional expanded version of the query for better retrieval
            - entities: Named entities found in the query
        """
        raw_query = input_data.get("query", "")
        
        # Determine query type using NLP techniques
        query_type = self._determine_query_type(raw_query)
        
        # Extract key terms and entities
        key_terms = self._extract_key_terms(raw_query)
        entities = self._extract_entities(raw_query)
        
        # Expand the query based on key terms and context
        expanded_query = self._expand_query(raw_query, key_terms, query_type)
        
        # Combine all the analysis into a structured result
        return {
            "analyzed_query": raw_query,
            "query_type": query_type,
            "key_terms": key_terms,
            "expanded_query": expanded_query,
            "entities": entities,
            "intent": self._determine_intent(raw_query, query_type)
        }
    
    def _determine_query_type(self, query: str) -> str:
        """
        Determine the type of query using linguistic patterns and keywords.
        
        Args:
            query: The user's raw query
            
        Returns:
            A string representing the query type (explanatory, definitional, etc.)
        """
        query_lower = query.lower()
        
        # Order matters! Check the most specific patterns first, then move to more general ones
        
        # Check for comparative queries first (most prone to being incorrectly classified)
        if re.search(r'\bhow\b.+\bcompare\b|\bcompare\b|\bdifference\b|\bdistinguish\b|\bversus\b|\bvs\b|\bsimilar\b|\bdifferent\b', query_lower):
            return "comparative"
            
        # Then check procedural queries
        elif re.search(r'\bhow\b.+\bdo\b|\bhow\b.+\bcan\b|\bhow\b.+\bto\b', query_lower):
            return "procedural"
            
        # Then move to more general query types
        elif re.search(r'\bhow\b|\bexplain\b|\bdescribe\b|\belaborate\b', query_lower):
            return "explanatory"
        elif re.search(r'\bwhat\s+is\b|\bdefine\b|\bmeaning\s+of\b|\bdefinition\b', query_lower):
            return "definitional"
        elif re.search(r'\bwhy\b|\bcause\b|\breason\b', query_lower):
            return "causal"
        elif re.search(r'\bwhen\b|\bwhere\b|\bwho\b|\bwhich\b', query_lower):
            return "factual"
        elif re.search(r'\blist\b|\bname\b|\bgive\b.+\bexamples\b', query_lower):
            return "enumerative"
        elif re.search(r'\badvantages\b|\bbenefits\b|\bdrawbacks\b|\blimitations\b', query_lower):
            return "evaluative"
        else:
            return "general"
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """
        Extract important terms from the query using NLP techniques.
        
        Args:
            query: The user's raw query
            
        Returns:
            A list of key terms from the query
        """
        if self.use_spacy and self.nlp is not None:
            # Use spaCy for more advanced term extraction
            doc = self.nlp(query)
            
            # Extract nouns, proper nouns, and important verbs
            key_terms = []
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"] or (token.pos_ == "VERB" and token.dep_ == "ROOT"):
                    if not token.is_stop and len(token.text) > 2:
                        key_terms.append(token.lemma_)
            
            # If we didn't find enough terms, include adjectives
            if len(key_terms) < 2:
                for token in doc:
                    if token.pos_ == "ADJ" and not token.is_stop and len(token.text) > 2:
                        key_terms.append(token.lemma_)
                        
            return key_terms
        else:
            # Fallback to basic keyword extraction without using word_tokenize
            # Using simple split to avoid NLTK dependency
            words = query.lower().split()
            key_terms = [word for word in words if word not in self.stopwords and len(word) > 2]
            
            # Count frequency to find more important terms
            term_counts = Counter(key_terms)
            
            # Return unique terms sorted by frequency
            return [term for term, _ in term_counts.most_common()]
    
    def _extract_entities(self, query: str) -> List[Dict[str, str]]:
        """
        Extract named entities from the query.
        
        Args:
            query: The user's raw query
            
        Returns:
            A list of dictionaries containing entity text and type
        """
        entities = []
        
        if self.use_spacy and self.nlp is not None:
            doc = self.nlp(query)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "type": ent.label_
                })
        
        return entities
    
    def _expand_query(self, query: str, key_terms: List[str], query_type: str) -> str:
        """
        Expand the query with related terms to improve retrieval.
        
        Args:
            query: The original query
            key_terms: Key terms extracted from the query
            query_type: The type of query
            
        Returns:
            An expanded version of the query
        """
        # Start with the original query
        expanded = query
        
        # Add type-specific expansions
        if query_type == "definitional":
            expanded += f" definition meaning concept"
        elif query_type == "explanatory":
            expanded += f" explanation process steps method"
        elif query_type == "comparative":
            expanded += f" comparison differences similarities versus"
        elif query_type == "causal":
            expanded += f" cause reason why result effect"
        
        # Include key terms with their variants if available
        term_expansions = []
        if self.use_spacy and self.nlp is not None:
            for term in key_terms[:3]:  # Limit to top 3 terms to avoid dilution
                doc = self.nlp(term)
                if len(doc) > 0:
                    # Include lemma form if different from the original
                    if doc[0].lemma_ != term:
                        term_expansions.append(doc[0].lemma_)
        
        if term_expansions:
            expanded += " " + " ".join(term_expansions)
        
        return expanded
    
    def _determine_intent(self, query: str, query_type: str) -> str:
        """
        Determine the user's intent based on query analysis.
        
        Args:
            query: The original query
            query_type: The type of query
            
        Returns:
            A string describing the likely user intent
        """
        query_lower = query.lower()
        
        # Check for common intent patterns
        if "how to" in query_lower or "how do i" in query_lower:
            return "instruction_seeking"
        elif "what is" in query_lower or "define" in query_lower:
            return "knowledge_seeking"
        elif "why" in query_lower:
            return "understanding_seeking"
        elif "compare" in query_lower or "difference" in query_lower:
            return "comparison_seeking"
        elif any(word in query_lower for word in ["problem", "error", "issue", "bug", "fix"]):
            return "troubleshooting"
        elif any(word in query_lower for word in ["best", "recommend", "should", "better"]):
            return "recommendation_seeking"
        else:
            # Default mapping based on query type
            intent_map = {
                "procedural": "instruction_seeking",
                "explanatory": "understanding_seeking",
                "definitional": "knowledge_seeking",
                "comparative": "comparison_seeking",
                "causal": "understanding_seeking",
                "factual": "fact_seeking",
                "enumerative": "information_gathering",
                "evaluative": "assessment_seeking",
                "general": "information_seeking"
            }
            return intent_map.get(query_type, "information_seeking")
