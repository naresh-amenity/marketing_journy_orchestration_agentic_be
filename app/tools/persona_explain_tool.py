from typing import Dict, Any, List, Optional
import logging
from app.tools.base_tool import BaseTool
from app.models.model import ToolResponse
from app.utils.db import MongoDB, convert_objectid
from bson import ObjectId
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaExplainTool(BaseTool):
    """
    Tool for explaining persona summaries to users
    
    This tool retrieves and formats persona summaries from the database
    and returns them in a user-friendly format. It can either return a
    summary of a specific persona or a list of all available personas.
    """
    
    def get_name(self) -> str:
        return "persona_explain"
    
    def get_description(self) -> str:
        return "Retrieves and explains persona summaries from the database"
    
    def get_required_params(self) -> List[str]:
        return ["user_id", "model_id"]
    
    def get_optional_params(self) -> List[str]:
        return ["conversation_id", "persona_name"]
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResponse:
        """
        Execute the tool with the given parameters
        
        Args:
            parameters: The parameters for the tool
            
        Returns:
            The response from the tool with persona summaries
        """
        try:
            # Extract parameters
            user_id = parameters.get("user_id")  # required
            model_id = parameters.get("model_id")  # required
            conversation_id = parameters.get("conversation_id")  # optional
            persona_name = parameters.get("persona_name")  # optional
            
            # Get MongoDB instance
            mongodb = MongoDB()
            
            # Retrieve persona summaries
            persona_summaries = []
            if hasattr(mongodb, 'get_persona_summaries') and conversation_id:
                persona_summaries = await mongodb.get_persona_summaries(
                    user_id=user_id,
                    model_id=model_id,
                    conversation_id=conversation_id,
                    conversation_status=True
                )
            else:
                # Fallback to find_documents if get_persona_summaries doesn't exist
                persona_summaries = await mongodb.find_documents(
                    collection="persona_summaries",
                    query={"user_id": user_id, "model_id": model_id},
                    limit=0  # Get all summaries
                )
            
            # Convert ObjectId to string in all summaries
            persona_summaries = convert_objectid(persona_summaries)
            
            # Handle case when no personas are found
            if not persona_summaries or len(persona_summaries) == 0:
                return ToolResponse(
                    status="success",
                    message=f"I couldn't find any personas for model ID {model_id}. Would you like to create some?",
                    data={"personas": []}
                )
            
            # If a specific persona name is provided, return details for that persona only
            if persona_name:
                specific_persona = None
                for summary in persona_summaries:
                    if isinstance(summary, dict) and summary.get("persona_name") == persona_name:
                        specific_persona = summary
                        break
                
                if specific_persona:
                    # Format the persona details
                    persona_details = f"Here's the summary for persona '{specific_persona.get('persona_name')}':\n\n"
                    
                    # Add the summary text if available
                    if 'summary' in specific_persona and specific_persona['summary']:
                        persona_details += specific_persona['summary']
                    
                    return ToolResponse(
                        status="success",
                        message=persona_details,
                        data={
                            "persona": specific_persona,
                            "persona_name": persona_name,
                            "model_id": model_id
                        }
                    )
                else:
                    # Persona name provided but not found
                    persona_names = [summary.get("persona_name") for summary in persona_summaries 
                                    if isinstance(summary, dict) and "persona_name" in summary]
                    
                    message = f"I couldn't find a persona named '{persona_name}' for model ID {model_id}. Here are the available personas: "
                    message += ", ".join(persona_names)
                    
                    return ToolResponse(
                        status="success",
                        message=message,
                        data={
                            "personas": persona_names,
                            "model_id": model_id
                        }
                    )
            
            # Otherwise list all available personas with their summaries
            formatted_personas = []
            for summary in persona_summaries:
                if isinstance(summary, dict) and "persona_name" in summary:
                    persona_info = {
                        "name": summary.get("persona_name"),
                        "summary": summary.get("summary", "No summary available")
                    }
                    formatted_personas.append(persona_info)
            
            message = f"Here are the persona summaries for model ID {model_id}:\n\n"
            for persona in formatted_personas:
                message += f"Persona: {persona['name']}\n"
                message += f"Summary: {persona['summary']}\n\n"
            
            return ToolResponse(
                status="success",
                message=message,
                data={
                    "personas": formatted_personas,
                    "model_id": model_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error retrieving persona summaries: {str(e)}", exc_info=True)
            return ToolResponse(
                status="error",
                message=f"An error occurred retrieving persona summaries: {str(e)}"
            )
    
    def _get_base_price(self) -> float:
        """Get base price for this tool"""
        return 0.00
    
    def _get_pricing_unit(self) -> str:
        """Get pricing unit for this tool"""
        return "request"
    
    def _get_additional_fees(self) -> Dict[str, Any]:
        """Get additional fees for this tool"""
        return {} 