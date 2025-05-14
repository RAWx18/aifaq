"""
Guardrails for controlling LLM output in the AIFAQ system.
Enhanced with semantic understanding and more robust filtering.
"""
import re
import logging
from typing import Dict, List, Optional, Tuple, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GuardrailConfig:
    """Configuration for LLM output guardrails."""
    
    def __init__(self):
        # Primary topics to block
        self.blocked_topics: List[str] = []
        
        # Related terms to each blocked topic - used for semantic understanding
        self.topic_related_terms: Dict[str, List[str]] = {}
        
        # Patterns to filter out from responses
        self.filtered_patterns: List[str] = []
        
        # Maximum response length in characters
        self.max_response_length: int = 300
        
        # Whether to enable safety checks
        self.enable_safety_checks: bool = True
        
        # Custom response templates for specific queries
        self.custom_responses: Dict[str, str] = {}
        
        # Disclaimer to add to responses when discussing sensitive topics
        self.disclaimers: Dict[str, str] = {}
        
        # Additional semantic rules
        self.high_risk_combinations: List[List[str]] = []
    
    def load_from_file(self, filepath: str) -> None:
        """Load guardrail configuration from a YAML file."""
        import yaml
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
                
            if 'blocked_topics' in config:
                self.blocked_topics = config['blocked_topics']
            if 'topic_related_terms' in config:
                self.topic_related_terms = config['topic_related_terms']
            if 'filtered_patterns' in config:
                self.filtered_patterns = config['filtered_patterns']
            if 'max_response_length' in config:
                self.max_response_length = config['max_response_length']
            if 'enable_safety_checks' in config:
                self.enable_safety_checks = config['enable_safety_checks']
            if 'custom_responses' in config:
                self.custom_responses = config['custom_responses']
            if 'disclaimers' in config:
                self.disclaimers = config['disclaimers']
            if 'high_risk_combinations' in config:
                self.high_risk_combinations = config['high_risk_combinations']
        except Exception as e:
            logger.error(f"Error loading guardrails config: {e}")
            self._set_default_config()

    def _set_default_config(self):
        """Set default configuration values with enhanced semantic understanding."""
        # Primary topics to block
        self.blocked_topics = [
            "new_topic", 
            "cryptocurrency", 
            "hacking", 
            "illegal activities",
            "personal data extraction",
            "discrimination",
            "harmful content"
        ]
        
        # Related terms for semantic understanding
        self.topic_related_terms = {
            "cryptocurrency": [
                "bitcoin", "ethereum", "blockchain investment", "token sale", 
                "crypto trading", "ICO", "mining profitability", "altcoin", 
                "trading strategy", "coin market", "crypto exchange"
            ],
            "hacking": [
                "exploit", "vulnerability", "unauthorized access", "bypass security",
                "crack password", "penetration testing", "exploit code", "backdoor",
                "security breach", "break into", "infiltrate system", "circumvent protection"
            ],
            "illegal activities": [
                "black market", "smuggling", "theft", "fraud scheme", "money laundering",
                "illegal download", "counterfeit", "unauthorized copying", "piracy",
                "unlawful access", "stealing", "illicit"
            ],
            "personal data extraction": [
                "dox", "personal information", "home address", "private details",
                "contact information", "email address", "phone number", "track someone",
                "locate person", "find someone", "personal profile"
            ],
            "discrimination": [
                "racial bias", "gender discrimination", "stereotype", "prejudice",
                "sexist", "racist", "discriminatory", "bias against", "hate speech"
            ],
            "harmful content": [
                "self-harm", "suicide method", "dangerous chemical", "weapon making",
                "explosives", "harmful instructions", "dangerous experiment"
            ]
        }
        
        # More robust pattern filtering with better regex
        self.filtered_patterns = [
            r"(?i)password\s+(?:is|should\s+be|could\s+be)\s+[\w\d\s\W]{3,}",
            r"(?i)(?:your|the|a|my)\s+password\s+(?:is|should|could|might|will|would)\s+[\w\d\s\W]{3,}",
            r"(?i)(?:credit\s+card|card\s+number|cvv|security\s+code)\s*:?\s*\d{3,}",
            r"(?i)(?:private|secret)\s+key\s*:?\s*[\w\d/+]{5,}",
            r"(?i)access\s+(?:token|key|credential)\s*:?\s*[\w\d]{5,}",
            r"(?i)api\s+(?:key|token|secret)\s*:?\s*[\w\d]{5,}",
            r"(?i)(?:login|account)\s+credential\s*:?\s*[\w\d]{3,}"
        ]
        
        # Response length limit
        self.max_response_length = 300
        
        # Enhanced custom responses with better patterns
        self.custom_responses = {
            r"(?i).*\b(?:how|can|ways?|method|technique|steps?)\s+(?:to|for|of)?\s*(?:hack|exploit|break|bypass|compromise).*": 
                "I cannot provide information about hacking, exploiting vulnerabilities, or unauthorized system access. Instead, I recommend exploring secure development practices and ethical cybersecurity resources if you're interested in security topics.",
            
            r"(?i).*(?:what|recommend|suggest|tell\s+me|give\s+me).*\b(?:password|credentials).*": 
                "For security reasons, I cannot provide specific password recommendations or discuss credential management specifics. Please refer to your organization's security policies or consult trusted security frameworks like NIST for best practices on credential management.",
            
            r"(?i).*(?:how|ways?|method|steps?)\s+(?:to|for|of)?\s*(?:avoid|bypass|trick|fool|get\s+around|circumvent)\s+(?:detection|security|authentication|verification|validation).*":
                "I cannot provide information on bypassing security measures or authentication systems. This kind of activity could be illegal and unethical. Instead, I'd be happy to discuss proper security implementation or legitimate troubleshooting approaches.",
            
            r"(?i).*(?:how|can|ways?)\s+(?:to|for|of)?\s*(?:steal|obtain|get|extract)\s+(?:data|information|credentials|password|key).*":
                "I cannot provide guidance on extracting data or credentials without authorization. Such activities are potentially illegal and unethical. I can instead discuss proper data access methods, API usage, or legitimate data integration approaches."
        }
        
        # More comprehensive disclaimers
        self.disclaimers = {
            "security": "\n\nNote: This information is provided for educational purposes only. Always follow your organization's security policies, consult with security professionals, and ensure compliance with relevant regulations when implementing security measures.",
            
            "blockchain": "\n\nNote: The blockchain technology landscape evolves rapidly. This information may change over time, and you should verify it against current documentation for your specific platform version.",
            
            "technical": "\n\nNote: Implementation details may vary based on your specific environment, software versions, and organizational requirements. Always test in a non-production environment first."
        }
        
        # High-risk term combinations that should trigger blocking
        # If multiple terms from a combination appear in a query, it will be blocked
        self.high_risk_combinations = [
            ["hack", "tutorial", "step"],
            ["password", "crack", "tool"],
            ["bypass", "security", "how"],
            ["steal", "data", "method"],
            ["exploit", "vulnerability", "code"]
        ]


class GuardrailProcessor:
    """Processor for applying guardrails to LLM responses with enhanced semantic understanding."""
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
    
    def check_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a query should be processed normally or requires a custom response.
        Enhanced with semantic understanding of topic relationships.
        
        Args:
            query: The user query to check
            
        Returns:
            Tuple of (should_process, custom_response)
            - should_process: True if the query should be processed normally
            - custom_response: Custom response to return (if should_process is False)
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Check for custom responses first - most specific checks
        for pattern, response in self.config.custom_responses.items():
            if re.search(pattern, query, re.IGNORECASE):
                logger.info(f"Custom response triggered for pattern: {pattern}")
                return False, response
        
        # Check for blocked topics - direct matches
        for topic in self.config.blocked_topics:
            if topic.lower() in query_lower:
                logger.info(f"Query blocked due to topic: {topic}")
                return False, f"I'm sorry, but I cannot provide information about {topic}."
        
        # Check for related terms to blocked topics - semantic understanding
        for topic, related_terms in self.config.topic_related_terms.items():
            matched_terms = []
            for term in related_terms:
                if term.lower() in query_lower:
                    matched_terms.append(term)
            
            # If 2 or more related terms are found, consider it as discussing the blocked topic
            if len(matched_terms) >= 2:
                logger.info(f"Query blocked due to multiple related terms for topic {topic}: {matched_terms}")
                return False, f"I'm sorry, but I cannot provide information that appears to be related to {topic}."
        
        # Check for high-risk term combinations
        for combination in self.config.high_risk_combinations:
            if all(term in query_words for term in combination):
                logger.info(f"Query blocked due to high-risk term combination: {combination}")
                return False, "I'm sorry, but I cannot provide information on this topic as it appears to be requesting potentially harmful or unethical guidance."
        
        # If we've made it this far, the query is allowed
        return True, None
    
    def process_response(self, query: str, response: str) -> str:
        """
        Apply guardrails to an LLM-generated response with enhanced filtering.
        
        Args:
            query: The user query that generated the response
            response: The raw LLM response
            
        Returns:
            The processed response with guardrails applied
        """
        processed = response
        query_lower = query.lower()
        
        # Apply length limit
        if len(processed) > self.config.max_response_length:
            processed = processed[:self.config.max_response_length] + "... [Response truncated for brevity]"
            logger.info(f"Response truncated to {self.config.max_response_length} characters")
        
        # Apply pattern filters with word boundary checks for better matching
        for pattern in self.config.filtered_patterns:
            original_length = len(processed)
            processed = re.sub(pattern, "[FILTERED]", processed, flags=re.IGNORECASE)
            if len(processed) != original_length:
                logger.info(f"Pattern '{pattern}' filtered from response")
        
        # Add security disclaimer for security-related content
        if any(term in query_lower for term in ["security", "secure", "protection", "safety", "privacy", "firewall", "encrypt"]):
            if not processed.endswith('\n'):
                processed += '\n'
            processed += self.config.disclaimers.get("security", "")
            logger.info("Added security disclaimer to response")
        
        # Add blockchain disclaimer for blockchain-related content
        elif any(term in query_lower for term in ["blockchain", "hyperledger", "distributed ledger", "smart contract"]):
            if not processed.endswith('\n'):
                processed += '\n'
            processed += self.config.disclaimers.get("blockchain", "")
            logger.info("Added blockchain disclaimer to response")
        
        # Add technical disclaimer for implementation-related content
        elif any(term in query_lower for term in ["implement", "deploy", "install", "configure", "setup"]):
            if not processed.endswith('\n'):
                processed += '\n'
            processed += self.config.disclaimers.get("technical", "")
            logger.info("Added technical disclaimer to response")
        
        # Post-filtering check for sensitive information that might have slipped through
        sensitive_patterns = [
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card format
            r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b',  # SSN format
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP address format
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email format
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, processed):
                processed = re.sub(pattern, "[REDACTED]", processed)
                logger.info(f"Redacted potentially sensitive information matching pattern {pattern}")
        
        return processed


# Default instance for easy import
default_guardrails = GuardrailProcessor()