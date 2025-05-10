from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent
from .logger import get_agent_logger
import time


class AgentCoordinator:
    """
    Coordinates the activities of multiple agents in the RAG pipeline.
    
    The coordinator manages the flow of information between agents and ensures
    that each agent's output is properly passed to the next agent in the pipeline.
    """
    
    def __init__(self, agents: List[BaseAgent]):
        """
        Initialize the coordinator with a list of agents.
        
        Args:
            agents: A list of BaseAgent instances that will participate in the pipeline.
        """
        self.agents = agents
        self.agent_map = {agent.name: agent for agent in agents}
        self.logger = get_agent_logger("coordinator")
        
    async def run_pipeline(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete multi-agent pipeline.
        
        Args:
            initial_input: The initial input data for the pipeline, typically including the user query.
            
        Returns:
            The final result after all agents have processed the data.
        """
        self.logger.info("Starting multi-agent pipeline")
        start_time = time.time()
        
        current_data = initial_input
        processing_history = []
        
        try:
            for agent in self.agents:
                self.logger.info(f"Running agent: {agent.name}")
                agent_start_time = time.time()
                
                # Use the new execute_with_logging method
                agent_result = await agent.execute_with_logging(current_data)
                
                agent_duration = time.time() - agent_start_time
                
                processing_history.append({
                    "agent": agent.name,
                    "duration": agent_duration,
                    "error": agent_result.get("error", None)
                })
                
                # If there was an error, log it but continue with what we have
                if "error" in agent_result:
                    self.logger.error(f"Error in agent {agent.name}: {agent_result['error']}")
                
                # Merge the agent result with the current data
                current_data = {**current_data, **agent_result}
            
            # Add the processing history to the result for debugging and evaluation
            current_data["processing_history"] = processing_history
            
            total_duration = time.time() - start_time
            self.logger.info(f"Multi-agent pipeline completed in {total_duration:.2f}s")
            
            return current_data
            
        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {str(e)}", exc_info=True)
            return {
                "error": f"Pipeline error: {str(e)}",
                "processing_history": processing_history
            }
    
    async def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agent_map.get(name)
        return self.agent_map.get(name)
