import json
from typing import Any, Dict, List, Optional, Union

from bson import ObjectId
from pydantic import BaseModel, Field


class ToolInput(BaseModel):
    """Model for tool input parameters"""

    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the tool")


class UserRequest(BaseModel):
    """Model for user request"""

    query: str = Field(..., description="User query or request")
    user_id: Optional[str] = Field(None, description="User ID")
    model_id: Optional[str] = Field(None, description="Model ID (optional)")
    session_token: Optional[str] = Field(None, description="Session token")
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID for stateful chat"
    )
    conversation_name: Optional[str] = Field(
        None, description="Name for the conversation"
    )
    is_conversation_end: Optional[bool] = Field(
        False, description="Flag indicating if this is the end of a conversation"
    )
    problem_statement: Optional[str] = Field(
        None, description="Problem statement of the user"
    )
    persona_name: Optional[str] = Field(
        None, description="persona name selected by user"
    )
    audience_id: Optional[str] = Field(None, description="audience id selected by user")
    budget: Optional[int] = Field(None, description="Budget for the journey")
    name: Optional[str] = Field(None, description="Name for the journey")
    date: Optional[str] = Field(
        None, description="Start date for the journey (YYYY-MM-DD format)"
    )
    target_id: Optional[str] = Field(None, description="Target ID for the journey")
    document_content: Optional[str] = Field(
        None, description="Base64-encoded document content for upload"
    )
    tool_input: Optional[ToolInput] = Field(
        None, description="Tool input parameters for direct tool calls"
    )
    has_audience_model: Optional[str] = Field(
        None, description="Whether the user has a model for the audience"
    )
    last_tool: Optional[str] = Field(None, description="Last tool used")


class ChatMessage(BaseModel):
    """Model for chat message"""

    role: str = Field(..., description="Role (user or assistant)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class ConversationState(BaseModel):
    """Model for conversation state"""

    conversation_id: str = Field(..., description="Conversation ID")
    user_id: str = Field(..., description="User ID")
    messages: List[ChatMessage] = Field(
        default_factory=list, description="Chat history"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Conversation context"
    )
    model_id: Optional[str] = Field(None, description="Model ID if available")
    active_persona_narratives: Optional[List[str]] = Field(
        None, description="List of active persona narratives"
    )
    conversation_name: Optional[str] = Field(
        None, description="Name of the conversation"
    )
    is_ended: Optional[bool] = Field(
        False, description="Whether the conversation has ended"
    )


class ToolResponse(BaseModel):
    """Model for tool response"""

    status: str = Field(..., description="Status of tool execution (success/error)")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    required_inputs: Optional[List[str]] = Field(
        None, description="List of required inputs if any"
    )


class ApiResponse(BaseModel):
    """Standard API response model"""

    status: str = Field(..., description="Status of the request (success/error)")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    required_inputs: Optional[List[str]] = Field(
        None, description="List of required inputs if any"
    )
    needed_input: Optional[List[str]] = Field(
        None, description="List of required inputs if any"
    )
    tool_used: Optional[str] = Field(None, description="Name of the tool that was used")
    pricing: Optional[Dict[str, Any]] = Field(None, description="Pricing information")
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID for stateful chat"
    )
    conversation_name: Optional[str] = Field(
        None, description="Name of the conversation"
    )
    is_conversation_end: Optional[bool] = Field(
        False, description="Whether this is the end of a conversation"
    )
    requesting_conversation_name: Optional[bool] = Field(
        False, description="Whether we're asking the user to name the conversation"
    )
    conversation_stage: Optional[str] = Field(
        "initial_greeting", description="Current stage of the conversation flow"
    )
    suggested_next_message: Optional[str] = Field(
        "", description="Suggested next message for guiding the user"
    )
    frontend_action: Optional[str] = Field(
        None,
        description="Action that frontend should take (e.g., show_persona_list, show_model_list)",
    )


# Specific models for different tools
class PersonaSummaryInput(BaseModel):
    """Input for persona summary tool"""

    user_id: str = Field(..., description="User ID")
    model_id: str = Field(..., description="Model ID")
    session_token: Optional[str] = Field(None, description="Session token")


class EmailPersonalizationInput(BaseModel):
    """Input for email personalization tool"""

    user_id: str = Field(..., description="User ID")
    model_id: str = Field(..., description="Model ID")
    session_token: Optional[str] = Field(None, description="Session token")
    problem_statement: Optional[str] = Field(None, description="Problem statement")


class DigitalAdPersonalizationInput(BaseModel):
    """Input for digital ad personalization tool"""

    user_id: str = Field(..., description="User ID")
    model_id: str = Field(..., description="Model ID")
    session_token: Optional[str] = Field(None, description="Session token")
    problem_statement: Optional[str] = Field(None, description="Problem statement")


class DirectMailPersonalizationInput(BaseModel):
    """Input for direct mail personalization tool"""

    user_id: str = Field(..., description="User ID")
    model_id: str = Field(..., description="Model ID")
    session_token: Optional[str] = Field(None, description="Session token")
    problem_statement: Optional[str] = Field(None, description="Problem statement")


class JourneyInput(BaseModel):
    """Input for journey-related tools"""

    user_id: str = Field(..., description="User ID")
    journey_id: str = Field(..., description="Journey ID")
    journey_name: Optional[str] = Field(None, description="Journey name")
    budget: Optional[int] = Field(None, description="Budget for the journey")
    date: Optional[str] = Field(
        None, description="Start date for the journey (YYYY-MM-DD format)"
    )
    target_id: Optional[str] = Field(None, description="Target ID for the journey")
    has_audience_model: Optional[str] = Field(
        None,
        description="Whether the user has a model for the audience ('yes' or 'no')",
    )
    model_id: Optional[str] = Field(
        None, description="Model ID (required only if has_audience_model is 'yes')"
    )


class HistoryConversationInput(BaseModel):
    """Input for history conversation tool"""

    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    model_id: Optional[str] = Field(None, description="Model ID")


# Custom JSON encoder to handle MongoDB ObjectId
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)
