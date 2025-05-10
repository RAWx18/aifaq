from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
import torch
from transformers import TextIteratorStreamer
from threading import Thread
import re


class ResponseGenerationAgent(BaseAgent):
    """
    Agent responsible for generating the final response.
    
    This agent:
    1. Takes the integrated context from the Context Integration Agent
    2. Formulates appropriate prompts for the LLM
    3. Generates concise, accurate responses
    4. Ensures responses are grounded in the retrieved documents
    """
    
    def __init__(self, model, tokenizer, name: str = "response_agent", 
                 max_new_tokens: int = 512, temperature: float = 0.7):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response based on the integrated context.
        
        Args:
            input_data: A dictionary containing:
                - integrated_context: The context prepared by the ContextIntegrationAgent
                - query: The original user query
                - query_type: The type of query being asked
                - intent: The inferred user intent
                - entities: Named entities from the query
            
        Returns:
            A dictionary containing:
                - response: The generated response
                - response_metadata: Additional information about the response
        """
        # Get the necessary data
        context = input_data.get("integrated_context", "")
        query = input_data.get("query", "")
        query_type = input_data.get("query_type", "general")
        intent = input_data.get("intent", "information_seeking")
        entities = input_data.get("entities", [])
        
        # Create a prompt based on the query type and intent
        system_prompt = self._get_system_prompt(query_type, intent)
        
        # Build the full prompt with context
        full_prompt = self._build_prompt(system_prompt, context, query, entities)
        
        # Generate the response using the LLM
        response = self._generate_response(full_prompt)
        
        # Post-process the response to ensure quality
        processed_response = self._post_process_response(response, query_type)
        
        # Generate metadata about the response
        response_metadata = {
            "grounded": self._is_response_grounded(processed_response, context),
            "response_type": query_type,
            "prompt_tokens": len(self.tokenizer.encode(full_prompt)),
            "response_tokens": len(self.tokenizer.encode(processed_response))
        }
        
        return {
            "response": processed_response,
            "response_metadata": response_metadata
        }
    
    def _get_system_prompt(self, query_type: str, intent: str) -> str:
        """
        Get an appropriate system prompt based on query type and intent.
        
        Args:
            query_type: The type of query (explanatory, definitional, etc.)
            intent: The inferred user intent
            
        Returns:
            A system prompt tailored to the query type and intent
        """
        # Base prompts for different query types
        system_prompts = {
            "explanatory": "You are a helpful assistant providing clear explanations. Use the following context to explain the topic thoroughly and logically.",
            "definitional": "You are a precise assistant defining concepts. Use the following context to provide a clear, concise definition.",
            "comparative": "You are a balanced assistant comparing topics. Use the following context to highlight similarities and differences in a fair way.",
            "causal": "You are an insightful assistant explaining causes and effects. Use the following context to explain the relationships between events or concepts.",
            "factual": "You are a factual assistant providing accurate information. Use the following context to give precise, concise facts without speculation.",
            "procedural": "You are a helpful guide providing step-by-step instructions. Use the following context to explain how to perform a task clearly and accurately.",
            "enumerative": "You are a thorough assistant providing comprehensive lists. Use the following context to enumerate all relevant items clearly and concisely.",
            "evaluative": "You are a balanced reviewer evaluating options. Use the following context to assess advantages and disadvantages fairly.",
            "general": "You are a helpful assistant providing information. Use the following context to give a relevant, concise response."
        }
        
        # Get the base prompt for the query type
        base_prompt = system_prompts.get(query_type, system_prompts["general"])
        
        # Enhance the prompt based on intent
        intent_enhancements = {
            "instruction_seeking": " Focus on clear, actionable steps that are easy to follow.",
            "knowledge_seeking": " Prioritize accuracy and clarity in your educational response.",
            "understanding_seeking": " Ensure a thorough explanation that builds conceptual understanding.",
            "comparison_seeking": " Present a balanced view of all sides with clear distinctions.",
            "troubleshooting": " Focus on identifying potential solutions to the problem.",
            "recommendation_seeking": " Provide thoughtful recommendations with justifications.",
            "information_seeking": " Deliver comprehensive, well-organized information.",
            "assessment_seeking": " Offer a fair evaluation of pros and cons."
        }
        
        # Add the intent-specific enhancement if available
        enhancement = intent_enhancements.get(intent, "")
        enhanced_prompt = base_prompt + enhancement
        
        # Add universal guidelines
        universal_guidelines = (
            " Base your answer strictly on the provided context. "
            "If the context doesn't contain enough information to answer fully, "
            "acknowledge limitations rather than inventing information. "
            "Use a clear, concise, and helpful tone."
        )
        
        return enhanced_prompt + universal_guidelines
    
    def _build_prompt(self, system_prompt: str, context: str, query: str, entities: List[Dict[str, str]]) -> str:
        """
        Build the full prompt for the LLM.
        
        Args:
            system_prompt: The system instruction
            context: The retrieved document context
            query: The user's question
            entities: Named entities from the query
            
        Returns:
            A formatted prompt string
        """
        # Highlight entities in the query if any exist
        highlighted_query = query
        for entity in entities:
            highlighted_query = highlighted_query.replace(
                entity["text"], 
                f"**{entity['text']}**"
            )
        
        # Format for the model (adapt this to your specific model's preferred format)
        # This example uses a chat format with system, context, and user messages
        prompt = f"""<|system|>
{system_prompt}

Here is the relevant context information:
{context}
<|endoftext|>

<|user|>
{highlighted_query}
<|endoftext|>

<|assistant|>"""
        
        return prompt
    
    def _generate_response(self, prompt: str) -> str:
        """
        Generate a response using the language model.
        
        Args:
            prompt: The formatted prompt
            
        Returns:
            The generated response text
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": 0.9,
            "do_sample": self.temperature > 0,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else self.tokenizer.eos_token_id,
        }
        
        # Generate the response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **gen_kwargs
            )
            
        # Decode the generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract just the assistant's response 
        response = generated_text.split("<|assistant|>")[-1].strip()
        
        return response
    
    def _post_process_response(self, response, query_type: str) -> str:
        """
        Post-process the generated response to improve quality.
        
        Args:
            response: The raw generated response
            query_type: The type of query
            
        Returns:
            The processed response
        """
        # If response is None or not a string, return empty string
        if response is None or not isinstance(response, str):
            return ""
            
        # Remove any remaining special tokens or artifacts
        response = re.sub(r"<\|.*?\|>", "", response)
        
        # Clean up whitespace
        response = re.sub(r"\s+", " ", response).strip()
        
        # Remove redundant phrases often generated by LLMs
        redundant_phrases = [
            "Based on the provided context,",
            "According to the information provided,",
            "As mentioned in the context,",
            "From the context provided,"
        ]
        
        for phrase in redundant_phrases:
            if response.startswith(phrase):
                response = response[len(phrase):].strip()
        
        # Format the response based on query type
        if query_type == "procedural":
            # Ensure step numbering is consistent for procedural queries
            if not re.search(r"^\d+\.\s", response):
                steps = response.split(". ")
                if len(steps) > 2:
                    formatted_steps = []
                    for i, step in enumerate(steps[:-1]):  # Exclude the last element if it's not a complete step
                        formatted_steps.append(f"{i+1}. {step}")
                    response = "\n".join(formatted_steps)
        
        elif query_type == "comparative":
            # Enhance structure for comparative responses
            if "advantages" in response.lower() and "disadvantages" in response.lower():
                # It's already well-structured
                pass
            else:
                # Try to ensure a clear structure by adding headings if not present
                comparison_terms = []
                for line in response.split(". "):
                    for word in ["versus", "compared to", "whereas", "while", "unlike"]:
                        if word in line.lower():
                            parts = line.lower().split(word)
                            if len(parts) >= 2:
                                term1 = parts[0].strip()
                                term2 = parts[1].split(" ")[0].strip()
                                if term1 and term2:
                                    comparison_terms.extend([term1, term2])
                
                # If we found terms to compare, restructure the response
                if len(comparison_terms) >= 2:
                    term1, term2 = comparison_terms[:2]
                    if not re.search(f"{term1}:|{term2}:", response, re.IGNORECASE):
                        # Try to split the response into sections about each term
                        response = f"Comparison between {term1} and {term2}:\n\n" + response
        
        return response
    
    def _is_response_grounded(self, response: str, context: str) -> bool:
        """
        Check if the response is properly grounded in the provided context.
        
        Args:
            response: The generated response
            context: The context provided to the model
            
        Returns:
            True if the response appears to be grounded in the context
        """
        # A simple approach: check if key sentences from the response
        # have substantial overlap with the context
        response_sentences = re.split(r'[.!?]', response)
        context_lower = context.lower()
        
        grounded_sentences = 0
        total_sentences = len(response_sentences)
        
        for sentence in response_sentences:
            sentence = sentence.strip()
            if not sentence:
                total_sentences -= 1
                continue
                
            # Check for significant word overlap
            important_words = [word for word in sentence.lower().split() 
                              if len(word) > 4 and word.isalnum()]
            
            if important_words:
                overlapping = sum(1 for word in important_words if word in context_lower)
                # If more than 40% of important words are in the context, consider it grounded
                if overlapping / len(important_words) > 0.4:
                    grounded_sentences += 1
        
        # If more than 70% of sentences are grounded, consider the response grounded
        return (total_sentences == 0) or (grounded_sentences / total_sentences > 0.7)
