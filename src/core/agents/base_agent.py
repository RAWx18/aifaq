from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time
import logging
from .logger import get_agent_logger


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent RAG system.
    
    Each agent has a specific responsibility in the RAG pipeline and implements
    the process method to perform its task.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_agent_logger(name)
        
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and return the results.
        
        Args:
            input_data: A dictionary containing the input data for the agent.
            
        Returns:
            A dictionary containing the results of the agent's processing.
        """
        # This will be implemented by each specific agent
        pass
    
    async def execute_with_logging(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper around the process method that adds logging and timing.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Result from the agent's process method
        """
        self.logger.info(f"Starting processing in {self.name}")
        start_time = time.time()
        
        try:
            # Log key input data without sensitive information
            safe_input = {k: v for k, v in input_data.items() 
                         if k not in ["model", "tokenizer", "vectordb"]}
            self.logger.debug(f"Input data: {safe_input}")
            
            # Call the actual processing method
            result = await self.process(input_data)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log results without very large content
            safe_result = {k: (v if not isinstance(v, (str, list)) or (isinstance(v, str) and len(v) < 500) 
                              else f"[{type(v).__name__}: length={len(v)}]") 
                          for k, v in result.items()}
            
            self.logger.info(f"Completed processing in {self.name} (took {processing_time:.2f}s)")
            self.logger.debug(f"Result: {safe_result}")
            
            # Add processing time to the result
            result["processing_time"] = processing_time
            
            return result
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}", exc_info=True)
            # Return minimal result with error information
            return {
                "error": str(e),
                "agent": self.name,
                "processing_time": time.time() - start_time
            }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
