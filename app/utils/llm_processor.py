import logging
import os
from typing import Dict, Any, Optional, List
import json
from app.models.model import UserRequest
from app.utils.graph_processor import GraphProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProcessor:
    """
    Processes user requests using LangGraph
    This class is maintained for backward compatibility
    """
    def __init__(self):
        # Initialize the GraphProcessor
        self.graph_processor = GraphProcessor()
        
    async def process_request(self, request: UserRequest) -> Optional[Dict[str, Any]]:
        """
        Process the user request using LangGraph
        
        Args:
            request: The user request object
            
        Returns:
            A dictionary containing the tool name and parameters, or None if no tool could be determined
        """
        try:
            # Process the request using the GraphProcessor
            result = await self.graph_processor.process_request(request)
            
            # If the status is input_required or error, return None
            if result.get("status") in ["input_required", "error"]:
                return None
            
            # Extract the tool_used and construct a tool_decision
            tool_name = result.get("tool_used")
            if not tool_name:
                return None
                
            # Get the parameters from the data if available
            parameters = {}
            data = result.get("data", {})
            if isinstance(data, dict):
                # Try to extract parameters from the data
                for key, value in data.items():
                    if key != "personas" and key != "email_personalization":
                        parameters[key] = value
            
            # Include user_id, model_id, and session_token from the request
            if request.user_id:
                parameters["user_id"] = request.user_id
            if request.model_id:
                parameters["model_id"] = request.model_id  
            if request.session_token:
                parameters["session_token"] = request.session_token
                
            # Construct a tool decision
            tool_decision = {
                "tool_name": tool_name,
                "parameters": parameters
            }
            
            return tool_decision
            
        except Exception as e:
            logger.error(f"Error processing request with LangGraph: {str(e)}", exc_info=True)
            return None 