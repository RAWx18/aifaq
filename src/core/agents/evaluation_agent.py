from typing import Dict, Any, List
from .base_agent import BaseAgent
import re


class EvaluationAgent(BaseAgent):
    """
    Agent responsible for evaluating the quality of responses.
    
    This agent:
    1. Assesses response relevance to the query
    2. Checks for factual accuracy against the context
    3. Evaluates response coherence and readability
    4. Provides feedback for improvement
    """
    
    def __init__(self, name: str = "evaluation_agent"):
        super().__init__(name)
        self.evaluation_metrics = [
            "relevance",
            "grounding",
            "completeness", 
            "coherence",
            "conciseness"
        ]
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of the generated response.
        
        Args:
            input_data: A dictionary containing:
                - response: The generated response
                - query: The original user query
                - integrated_context: The context used for generation
                - retrieved_documents: The documents retrieved for the query
                - query_type: The type of query being asked
            
        Returns:
            A dictionary containing:
                - evaluation_scores: Scores for each evaluation metric
                - evaluation_feedback: Detailed feedback about the response
                - quality_assessment: Overall assessment of response quality
                - improvement_suggestions: Suggestions for improving the response
        """
        # Get the necessary data
        response = input_data.get("response", "")
        query = input_data.get("query", "")
        context = input_data.get("integrated_context", "")
        documents = input_data.get("retrieved_documents", [])
        query_type = input_data.get("query_type", "general")
        
        # Evaluate different aspects of the response
        evaluation_scores = {
            "relevance": self._evaluate_relevance(response, query),
            "grounding": self._evaluate_grounding(response, documents),
            "completeness": self._evaluate_completeness(response, query, query_type),
            "coherence": self._evaluate_coherence(response),
            "conciseness": self._evaluate_conciseness(response)
        }
        
        # Calculate an overall score
        overall_score = sum(evaluation_scores.values()) / len(evaluation_scores)
        
        # Generate feedback based on the evaluation
        feedback = self._generate_feedback(evaluation_scores, query_type)
        
        # Provide an overall quality assessment
        quality_assessment = self._assess_overall_quality(evaluation_scores)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(evaluation_scores, query_type)
        
        return {
            "evaluation_scores": evaluation_scores,
            "overall_score": round(overall_score, 2),
            "evaluation_feedback": feedback,
            "quality_assessment": quality_assessment,
            "improvement_suggestions": improvement_suggestions
        }
    
    def _evaluate_relevance(self, response: str, query: str) -> float:
        """
        Evaluate how relevant the response is to the query.
        
        Args:
            response: The generated response
            query: The user's query
            
        Returns:
            A score between 0 and 1 indicating relevance
        """
        # Extract key terms from query
        query_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
        if not query_words:
            return 0.5  # Default if no significant query words
        
        # Check for presence of query terms in response
        response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))
        
        # Calculate term overlap
        if not response_words:
            return 0.0
            
        overlap = len(query_words.intersection(response_words))
        relevance_score = min(1.0, overlap / len(query_words))
        
        # Boost score if response directly addresses question words
        question_words = ["what", "who", "where", "when", "why", "how"]
        for word in question_words:
            if word in query.lower() and word not in response.lower():
                # The response should address the question type
                relevance_score = max(0.0, relevance_score - 0.1)
                
        return relevance_score
    
    def _evaluate_grounding(self, response: str, documents: List[str]) -> float:
        """
        Evaluate how well the response is grounded in the source documents.
        
        Args:
            response: The generated response
            documents: The source documents
            
        Returns:
            A score between 0 and 1 indicating grounding
        """
        if not documents:
            return 0.5  # Default when no documents are provided
            
        # Combine documents into a single context for matching
        combined_docs = " ".join(documents).lower()
        
        # Split response into sentences for analysis
        response_sentences = re.split(r'[.!?]', response)
        response_sentences = [s.strip() for s in response_sentences if s.strip()]
        
        if not response_sentences:
            return 0.5  # Default for empty response
            
        # Count sentences with good grounding
        grounded_sentences = 0
        
        for sentence in response_sentences:
            # Extract key terms (4+ letter words) from sentence
            sentence_terms = re.findall(r'\b\w{4,}\b', sentence.lower())
            if not sentence_terms:
                continue
                
            # Check term overlap with documents
            matching_terms = sum(1 for term in sentence_terms if term in combined_docs)
            
            # If more than 50% of terms are in docs, consider it grounded
            if matching_terms / len(sentence_terms) > 0.5:
                grounded_sentences += 1
        
        # Calculate grounding score
        grounding_score = grounded_sentences / len(response_sentences) if response_sentences else 0.5
        
        return grounding_score
    
    def _evaluate_completeness(self, response: str, query: str, query_type: str) -> float:
        """
        Evaluate the completeness of the response relative to the query.
        
        Args:
            response: The generated response
            query: The user's query
            query_type: The type of query
            
        Returns:
            A score between 0 and 1 indicating completeness
        """
        # Base completeness on response length appropriate to query type
        word_count = len(response.split())
        
        # Define minimum word counts for different query types
        min_words = {
            "factual": 20,
            "definitional": 40,
            "explanatory": 80,
            "comparative": 100,
            "procedural": 60,
            "causal": 70,
            "general": 50
        }
        
        # Define ideal word counts
        ideal_words = {
            "factual": 50,
            "definitional": 100,
            "explanatory": 200,
            "comparative": 250,
            "procedural": 150,
            "causal": 180,
            "general": 120
        }
        
        # Get thresholds for this query type
        min_threshold = min_words.get(query_type, 50)
        ideal_threshold = ideal_words.get(query_type, 120)
        
        # Score based on word count relative to thresholds
        if word_count < min_threshold:
            # Below minimum threshold
            completeness_score = word_count / min_threshold * 0.5
        elif word_count < ideal_threshold:
            # Between minimum and ideal
            completeness_score = 0.5 + ((word_count - min_threshold) / 
                                        (ideal_threshold - min_threshold) * 0.4)
        else:
            # At or above ideal threshold
            completeness_score = 0.9
            
        # Check for expected elements based on query type
        if query_type == "procedural":
            # Procedural responses should have numbered steps
            has_steps = bool(re.search(r'\d+\.|\d+\)|\bstep\b\s*\d+', response.lower()))
            completeness_score = completeness_score * 0.8 + (0.2 if has_steps else 0)
            
        elif query_type == "comparative":
            # Comparative responses should mention comparison terms
            comparison_terms = ["whereas", "compared to", "similarly", "unlike", "in contrast", "advantage", "disadvantage"]
            has_comparison = any(term in response.lower() for term in comparison_terms)
            completeness_score = completeness_score * 0.8 + (0.2 if has_comparison else 0)
            
        elif query_type == "definitional":
            # Definitional responses should start with a definition
            first_sentences = re.split(r'[.!?]', response)
            first_sentences = [s.strip() for s in first_sentences if s.strip()]
            first_sentence = first_sentences[0] if first_sentences else ""
            has_definition = bool(re.search(r'is\s+a|refers\s+to|defined\s+as', first_sentence.lower()))
            completeness_score = completeness_score * 0.8 + (0.2 if has_definition else 0)
            
        return min(1.0, completeness_score)
    
    def _evaluate_coherence(self, response: str) -> float:
        """
        Evaluate the logical flow and readability of the response.
        
        Args:
            response: The generated response
            
        Returns:
            A score between 0 and 1 indicating coherence
        """
        # Simple heuristics for coherence evaluation
        response_sentences = re.split(r'[.!?]', response)
        response_sentences = [s.strip() for s in response_sentences if s.strip()]
        
        if len(response_sentences) <= 1:
            return 0.5  # Default for very short responses
            
        # Check for transition words that indicate good structure
        transition_words = [
            "first", "second", "third", "finally", "additionally", "furthermore",
            "however", "therefore", "consequently", "in conclusion", "for example",
            "meanwhile", "nevertheless", "similarly", "in contrast", "specifically"
        ]
        
        # Count sentences with transition words
        transition_count = sum(1 for sentence in response_sentences 
                              if any(word in sentence.lower() for word in transition_words))
        
        # Calculate transition density (what % of sentences use transitions)
        transition_density = transition_count / (len(response_sentences) - 1)  # Exclude first sentence
        
        # Check for paragraph breaks (basic structure)
        has_paragraphs = '\n\n' in response or '\n \n' in response
        
        # Calculate coherence score
        coherence_score = 0.5  # Base score
        coherence_score += min(0.3, transition_density * 0.6)  # Up to 0.3 for transitions
        coherence_score += 0.2 if has_paragraphs else 0  # 0.2 for paragraph structure
        
        return coherence_score
    
    def _evaluate_conciseness(self, response: str) -> float:
        """
        Evaluate how concise and to-the-point the response is.
        
        Args:
            response: The generated response
            
        Returns:
            A score between 0 and 1 indicating conciseness
        """
        # Count words
        word_count = len(response.split())
        
        # Check for redundancy indicators
        redundant_phrases = [
            "as mentioned earlier", "as stated before", "as I said", 
            "to reiterate", "as previously mentioned"
        ]
        
        redundancy_count = sum(1 for phrase in redundant_phrases 
                              if phrase in response.lower())
        
        # Repeated sentences or near-duplicates (simplified check)
        response_sentences = re.split(r'[.!?]', response)
        response_sentences = [s.strip().lower() for s in response_sentences if s.strip()]
        
        # Check for similar sentences (crude approximation)
        similar_sentences = 0
        for i in range(len(response_sentences)):
            for j in range(i+1, len(response_sentences)):
                # If sentences share more than 70% of words, consider them similar
                words_i = set(response_sentences[i].split())
                words_j = set(response_sentences[j].split())
                if words_i and words_j:
                    overlap = len(words_i.intersection(words_j)) / min(len(words_i), len(words_j))
                    if overlap > 0.7:
                        similar_sentences += 1
        
        # Calculate base conciseness score based on word count
        # Ideal range: 50-250 words
        if word_count < 20:  # Too short
            conciseness_score = 0.3
        elif word_count <= 250:  # Ideal range
            conciseness_score = 0.9
        elif word_count <= 400:  # A bit verbose
            conciseness_score = 0.9 - ((word_count - 250) / 750)
        else:  # Too verbose
            conciseness_score = 0.5
            
        # Penalize for redundancy
        redundancy_penalty = redundancy_count * 0.1 + similar_sentences * 0.05
        conciseness_score = max(0.1, conciseness_score - redundancy_penalty)
        
        return min(1.0, conciseness_score)
    
    def _generate_feedback(self, scores: Dict[str, float], query_type: str) -> List[str]:
        """
        Generate feedback based on evaluation scores.
        
        Args:
            scores: Dictionary of metric scores
            query_type: The type of query
            
        Returns:
            List of feedback points
        """
        feedback = []
        
        # Add feedback for each metric below threshold
        threshold = 0.7
        
        if scores["relevance"] < threshold:
            feedback.append("The response could be more directly relevant to the query.")
            
        if scores["grounding"] < threshold:
            feedback.append("The response should be better grounded in the source documents.")
            
        if scores["completeness"] < threshold:
            if query_type == "explanatory":
                feedback.append("The explanation could be more comprehensive.")
            elif query_type == "procedural":
                feedback.append("The instructions could be more detailed.")
            elif query_type == "comparative":
                feedback.append("The comparison could cover more aspects.")
            else:
                feedback.append("The response could be more complete.")
                
        if scores["coherence"] < threshold:
            feedback.append("The response could have better logical flow and structure.")
            
        if scores["conciseness"] < threshold:
            feedback.append("The response could be more concise without losing essential information.")
            
        # If all scores are good, provide positive feedback
        if all(score >= threshold for score in scores.values()):
            feedback = ["The response is of high quality across all evaluation dimensions."]
            
        return feedback
    
    def _assess_overall_quality(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Provide an overall assessment of response quality.
        
        Args:
            scores: Dictionary of metric scores
            
        Returns:
            Dictionary with quality assessment
        """
        # Calculate average score
        avg_score = sum(scores.values()) / len(scores)
        
        # Determine quality level
        if avg_score >= 0.85:
            quality_level = "excellent"
        elif avg_score >= 0.75:
            quality_level = "good"
        elif avg_score >= 0.65:
            quality_level = "satisfactory"
        else:
            quality_level = "needs improvement"
            
        # Identify strongest and weakest areas
        strongest = max(scores.items(), key=lambda x: x[1])[0]
        weakest = min(scores.items(), key=lambda x: x[1])[0]
        
        return {
            "average_score": round(avg_score, 2),
            "quality_level": quality_level,
            "strongest_aspect": strongest,
            "weakest_aspect": weakest
        }
    
    def _generate_improvement_suggestions(self, scores: Dict[str, float], query_type: str) -> List[str]:
        """
        Generate concrete suggestions for improvement.
        
        Args:
            scores: Dictionary of metric scores
            query_type: The type of query
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Focus on the lowest scoring aspects
        lowest_metrics = sorted(scores.items(), key=lambda x: x[1])[:2]
        
        for metric, score in lowest_metrics:
            if score >= 0.7:  # Only suggest improvements for low scores
                continue
                
            if metric == "relevance":
                suggestions.append("Focus more directly on answering the specific question posed.")
                
            elif metric == "grounding":
                suggestions.append("Ensure all key statements are supported by the provided documents.")
                
            elif metric == "completeness":
                if query_type == "explanatory":
                    suggestions.append("Provide more comprehensive explanations with examples.")
                elif query_type == "procedural":
                    suggestions.append("Include more detailed, step-by-step instructions.")
                elif query_type == "comparative":
                    suggestions.append("Compare entities across more dimensions and characteristics.")
                else:
                    suggestions.append("Address more aspects of the query in the response.")
                    
            elif metric == "coherence":
                suggestions.append("Improve organization with clearer transitions between ideas.")
                
            elif metric == "conciseness":
                suggestions.append("Eliminate redundant statements and focus on essential information.")
                
        # Add query-type specific suggestions if overall quality is low
        if sum(scores.values()) / len(scores) < 0.7:
            type_suggestions = {
                "explanatory": "Structure explanations with an introduction, main points, and a conclusion.",
                "definitional": "Start with a clear definition before elaborating on details.",
                "comparative": "Use a parallel structure when comparing entities.",
                "procedural": "Number steps and consider potential challenges or variations.",
                "factual": "Focus on accuracy and provide specific details instead of generalizations."
            }
            
            if query_type in type_suggestions:
                suggestions.append(type_suggestions[query_type])
                
        return suggestions
