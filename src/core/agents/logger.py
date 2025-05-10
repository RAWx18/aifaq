"""
Logger configuration for multi-agent system.

This module provides unified logging setup for all agents.
"""

import logging
import os
import sys
from typing import Optional

# Define log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# Default log format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logger(name: str, 
                level: str = "info", 
                log_file: Optional[str] = None,
                log_format: str = DEFAULT_FORMAT) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Name of the logger, typically __name__ or agent name
        level: Log level as string ("debug", "info", "warning", "error", "critical")
        log_file: Optional file path to write logs to
        log_format: Format string for log messages
        
    Returns:
        Configured logger instance
    """
    # Get the log level
    log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_agent_logger(agent_name: str, level: str = "info") -> logging.Logger:
    """
    Get a logger configured specifically for an agent.
    
    Args:
        agent_name: Name of the agent
        level: Log level
        
    Returns:
        Configured logger instance for the agent
    """
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{agent_name}.log")
    return setup_logger(
        name=f"agent.{agent_name}",
        level=level,
        log_file=log_file
    )
