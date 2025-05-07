import logging
import os
from typing import Dict, Any, List, TypedDict, Optional, Annotated, Union
import json
import operator
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, field_validator
from app.models.model import UserRequest, ToolResponse
from app.utils.tool_registry import ToolRegistry
from langchain_core.prompts import PromptTemplate
from app.utils.db import MongoDB

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("openai_key")
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key not found in environment variables")

# Get LangChain API key from environment
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

class GraphState(BaseModel):
    """State for the graph"""
    query: str
    context: Dict[str, Any]
    messages: List[Dict[str, Any]]
    tool_decision: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    missing_params: Optional[List[str]] = None
    validated: bool = False
    conversation_name: Optional[str] = None
    is_conversation_end: bool = False
    last_message: str
    conversation_stage: str = Field(default="initial_greeting")
    suggested_next_message: str = Field(default="How can I help you with personalization today?")
    needed_input: Optional[Dict[str, Any]] = None
    frontend_action: Optional[str] = None

class ToolSelection(BaseModel):
    """
    Model representing the tool selection and its parameters
    """
    tool_name: str = Field(default="")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    conversation_stage: str = Field(default="initial_greeting")
    suggested_next_message: str = Field(default="How can I help you with personalization today?")

tool_parm_dict = {
    "persona_summary": ["user_id", "model_id"],
    "email_personalization": ["user_id", "model_id", "problem_statement", "persona_name"],
    "directmail_personalization": ["user_id", "model_id", "problem_statement", "persona_name"],
    "digitalad_personalization": ["user_id", "model_id", "problem_statement", "persona_name"]
}


# System prompt for the LLM tool selector
TOOL_SELECTION_PROMPT = """
You are an AI assistant for a marketing personalization system that helps determine which tool to use for user requests. Your task is to analyze the user's input and decide which tool to call based on their query.

You will be given two input variables:
<last_message>
{last_message}
</last_message>

<user_query>
{user_query}
</user_query>

Analyze these inputs as follows:
1. If the last_message contains a question from the AI about creating a persona narrative (or any similar question) and the user_query contains an affirmative response, extract the relevant tool name based on that context.
2. If there's no relevant context in the last_message, focus on analyzing the user_query to determine the appropriate tool.
3. If the user requests multiple tools in a single message, prepare to ask a clarifying question.

Your analysis should include:
1. Carefully examining the user's request to understand their intent
2. Determining if they need a specific marketing tool or just general conversation
3. Extracting all necessary parameters mentioned in their request
4. Identifying any missing required parameters they need to provide

After your analysis, format your response as a valid JSON object with these exact keys:
- "tool_name": String with the name of the tool, or empty string "" if no specific tool is needed
- "parameters": Object containing parameter key-value pairs that were found in the query
- "required_parameters": Array of parameter names that are required but missing from the query
- "conversation_stage": String indicating the current stage of the conversation (e.g., "initial_greeting", "exploring_options", "collecting_model_id", "suggesting_personalization", "showing_results")
- "suggested_next_message": String with a suggested natural next message that guides the user (don't use this for direct response, it's just a suggestion for the conversation flow)

For general conversation, return an empty string "" for tool_name, but still include the parameters object.

Available tools and their parameters:

Persona Creation Tools:
- persona_summary
  Required: user_id, model_id
  Optional: session_token
  Use when: Users want to create or generate customer personas

Email Marketing Tools:
- email_personalization
  Required: user_id, model_id, problem_statement, persona_name
  Optional: session_token
  Use when: Users want personalized email content for existing personas

Direct Mail Tools:
- directmail_personalization
  Required: user_id, model_id, problem_statement, persona_name
  Optional: session_token
  Use when: Users want personalized direct mail content for existing personas

Digital Ad Tools:
- digitalad_personalization
  Required: user_id, model_id, problem_statement, persona_name
  Optional: session_token
  Use when: Users want personalized digital ad content for existing personas

Important guidelines:
- If the user wants personalized content but doesn't have personas, suggest using persona_summary first and set conversation_stage to "suggesting_persona_creation"
- If problem_statement is missing, identify it as a required parameter
- If user_id and model_id are provided in the user context, use those values
- Return parameter names exactly as specified above (don't rename or modify them)
- Only extract parameters that the user has explicitly mentioned

The conversation flow should follow this natural progression:
1. User greets -> respond with a helpful welcome and overview of capabilities
2. User mentions personalization -> ask if they have persona narratives or audience files
3. If user has persona narratives -> suggest selecting from available narratives (frontend will show list)
4. If user doesn't have persona narratives -> ask if they want to create one for a specific model_id
5. If user wants to create persona narrative -> ask for model_id if not provided
6. After persona creation -> ask for which channel they want personalization (email, direct mail, digital ad)
7. After channel selection -> ask for any required information and proceed with personalization

If the user requests multiple tools in a single message, prepare a clarifying question to determine which tool they need most urgently. Include this question in the JSON response under a "clarification_question" key.

Your final output should be a valid JSON object. Do not include any explanation or additional text outside of the JSON structure. Ensure that the JSON is properly formatted and can be parsed without errors.

Example valid responses:
{{"tool_name": "persona_summary", "parameters": {{"user_id": "123"}}, "conversation_stage": "collecting_model_id", "suggested_next_message": "Great! To create a persona narrative, I'll need your model ID. Do you have that information?"}}
{{"tool_name": "email_personalization", "parameters": {{"model_id": "456", "problem_statement": "promoting summer sale"}}, "conversation_stage": "collecting_persona_name", "suggested_next_message": "Perfect, now I need to know which persona you want to use for this email personalization."}}
{{"tool_name": "", "parameters": {{}}, "conversation_stage": "initial_greeting", "suggested_next_message": "How can I help you with personalization today?"}}
"""

class GraphNodes(Enum):
    """
    Nodes in the LangGraph for tool processing
    """
    TOOL_DECISION = "tool_decision"
    VALIDATE_TOOL = "validate_tool"
    EXECUTE_TOOL = "execute_tool"
    MISSING_PARAMS = "missing_params"
    ERROR = "error"


class AsyncToolExecutor:
    """
    Asynchronous executor for tools
    """
    def __init__(self):
        self.tool_registry = ToolRegistry()
    
    async def execute(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResponse:
        """
        Execute a tool with the given parameters
        
        Args:
            tool_name: The name of the tool to execute
            parameters: The parameters for the tool
            
        Returns:
            The response from the tool execution
        """
        if not self.tool_registry.has_tool(tool_name):
            return ToolResponse(
                status="error",
                message=f"Tool '{tool_name}' not found",
                data=None
            )
        
        tool = self.tool_registry.get_tool(tool_name)
        return await tool.execute(parameters)


class GraphProcessor:
    """
    Class for processing requests using LangGraph
    """
    
    def __init__(self):
        """Initialize the GraphProcessor"""
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0,
            api_key=OPENAI_API_KEY
        )
        self.graph = self._create_graph()
        self.db = MongoDB()
        
    def _create_graph(self) -> StateGraph:
        """
        Create the LangGraph for processing requests
        
        Returns:
            A StateGraph instance
        """
        # Create the graph
        graph = StateGraph(GraphState)
        
        # Add nodes
        graph.add_node("analyze_request", self._analyze_request)
        graph.add_node("extract_parameters", self._extract_parameters)
        graph.add_node("validate_parameters", self._validate_parameters)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_node("format_response", self._format_response)
        
        # Add edges
        graph.add_edge("analyze_request", "extract_parameters")
        graph.add_edge("extract_parameters", "validate_parameters")
        graph.add_edge("validate_parameters", "execute_tool")
        graph.add_edge("execute_tool", "format_response")
        graph.add_edge("format_response", END)
        
        # Set entry point
        graph.set_entry_point("analyze_request")
        
        return graph.compile()

    async def _analyze_request(self, state: GraphState) -> GraphState:
        """
        Analyze the request to determine the type and required parameters
        
        Args:
            state: The current state
            
        Returns:
            Updated state
        """
        parser = JsonOutputParser(pydantic_object=ToolSelection)

        prompt = PromptTemplate(
            template=TOOL_SELECTION_PROMPT,
            input_variables=["last_message", "user_query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.llm | parser
        
        # try:
        if True:
            # Run the chain
            result = await chain.ainvoke({"last_message": state.last_message, "user_query": state.query})
            try:
                result["required_parameters"] = tool_parm_dict[result["tool_name"]]
            except:
                result["required_parameters"] = []
            data_dict = {}
            if state.context["model_id"]:
                data_dict["model_id"] = state.context["model_id"]
            if state.context["user_id"]:
                data_dict["user_id"] = state.context["user_id"]

            # Store conversation stage and suggested message in the state
            if "conversation_stage" in result:
                state.conversation_stage = result["conversation_stage"]
            
            # Store the suggested next message for possible use
            if "suggested_next_message" in result:
                state.suggested_next_message = result["suggested_next_message"]

            # Check if user is responding about having persona narratives or not
            user_query_lower = state.query.lower()
            has_affirmative = any(word in user_query_lower for word in ["yes", "yeah", "yep", "sure", "i do", "have"])
            has_negative = any(word in user_query_lower for word in ["no", "nope", "don't", "do not", "haven't", "have not"])
            wants_recreate = any(phrase in user_query_lower for phrase in ["recreate", "create new", "make new", "new ones", "replace"])
            
            # Check for conversation stage and user responses
            last_stage = getattr(state, "conversation_stage", "")
            
            # If user wants to recreate personas when they already exist
            if (wants_recreate and 
                ("correcting_has_personas" in last_stage or 
                 "show_persona_list_with_recreate_option" in str(state.last_message).lower() or
                 ("recreate" in state.last_message.lower() and has_affirmative))):
                
                # User wants to recreate personas
                if not result.get("tool_name"):
                    result["tool_name"] = "persona_summary"
                result["conversation_stage"] = "recreating_personas"
                result["suggested_next_message"] = "I'll create new persona narratives for you. This will replace your existing ones for this model ID."
                result["frontend_action"] = "recreate_personas"
                
                # Add needed_input to indicate frontend should recreate personas
                if not "needed_input" in result:
                    result["needed_input"] = {}
                result["needed_input"]["recreate_personas"] = True
                result["needed_input"]["user_id"] = state.context.get("user_id")
                result["needed_input"]["model_id"] = state.context.get("model_id")
                
                # Add parameter to tool execution to force recreation
                if not "parameters" in result:
                    result["parameters"] = {}
                result["parameters"]["recreate_personas"] = True
                
                return state
            
            # Check for explicit statements about recreating personas
            if "recreate" in user_query_lower and "persona" in user_query_lower:
                user_id = state.context.get("user_id")
                model_id = state.context.get("model_id")
                
                # Check if model_id is missing, ask for it first
                if not model_id:
                    if not result.get("tool_name"):
                        result["tool_name"] = ""
                    result["conversation_stage"] = "collecting_model_id"
                    result["suggested_next_message"] = "I'll help you recreate your persona narratives. Please select your model ID from the dropdown."
                    result["frontend_action"] = "show_model_list"
                    
                    # Add needed_input to indicate frontend should show model ID selection
                    if not "needed_input" in result:
                        result["needed_input"] = {}
                    result["needed_input"]["show_model_list"] = True
                    result["needed_input"]["user_id"] = user_id
                    return state
                
                # Ready to recreate personas
                if not result.get("tool_name"):
                    result["tool_name"] = "persona_summary"
                result["conversation_stage"] = "recreating_personas"
                result["suggested_next_message"] = "I'll recreate persona narratives for your model now. This will replace any existing ones."
                
                # Add parameters for tool execution
                if not "parameters" in result:
                    result["parameters"] = {}
                result["parameters"]["user_id"] = user_id
                result["parameters"]["model_id"] = model_id  
                result["parameters"]["recreate_personas"] = True
                
                return state
            
            # If last message was asking about having persona narratives
            elif "persona narrative" in state.last_message.lower() or "audience file" in state.last_message.lower():
                if has_affirmative:
                    # User indicates they have persona narratives - check if they really exist and show list
                    user_id = state.context.get("user_id")
                    model_id = state.context.get("model_id")
                    
                    # Check if model_id is missing, ask for it first
                    if not model_id:
                        if not result.get("tool_name"):
                            result["tool_name"] = ""
                        result["conversation_stage"] = "collecting_model_id"
                        result["suggested_next_message"] = "Great! To check your existing persona narratives, I'll need your model ID. Please select it from the dropdown."
                        result["frontend_action"] = "show_model_list"
                        
                        # Add needed_input to indicate frontend should show model ID selection
                        if not "needed_input" in result:
                            result["needed_input"] = {}
                        result["needed_input"]["show_model_list"] = True
                        result["needed_input"]["user_id"] = user_id
                        return state
                    
                    # Model ID is available, check if persona narratives exist for this user and model
                    persona_exists = False
                    try:
                        persona_summaries = await self.db.get_persona_summaries(
                            user_id=str(user_id), 
                            model_id=str(model_id), 
                            conversation_id=str(state.context.get("conversation_id")), 
                            conversation_status=True
                        )
                        persona_exists = persona_summaries and len(persona_summaries) > 0
                    except Exception as e:
                        logger.error(f"Error checking persona existence: {str(e)}")
                    
                    if persona_exists:
                        # Personas exist, show the list and ask which channel for personalization
                        if not result.get("tool_name"):
                            result["tool_name"] = ""
                        result["conversation_stage"] = "selecting_personalization_channel"
                        result["suggested_next_message"] = "Great! I found your existing persona narratives. For which channel would you like to create personalization - email, direct mail, or digital ads?"
                        result["frontend_action"] = "show_persona_list"
                        
                        # Add needed_input to indicate frontend should show persona list
                        if not "needed_input" in result:
                            result["needed_input"] = {}
                        result["needed_input"]["show_persona_list"] = True
                        result["needed_input"]["user_id"] = user_id
                        result["needed_input"]["model_id"] = model_id
                    else:
                        # User says they have personas but none found
                        if not result.get("tool_name"):
                            result["tool_name"] = "persona_summary"
                        result["conversation_stage"] = "correcting_no_personas"
                        result["suggested_next_message"] = "I've checked but don't actually see any existing persona narratives for this model ID. We'll need to create some first. Is that okay?"
                        result["frontend_action"] = "show_model_list"
                        
                        # Add needed_input to indicate frontend should show model ID selection
                        if not "needed_input" in result:
                            result["needed_input"] = {}
                        result["needed_input"]["show_model_list"] = True
                        result["needed_input"]["user_id"] = user_id
                    
                elif has_negative:
                    # User indicates they don't have persona narratives - create new ones
                    user_id = state.context.get("user_id")
                    model_id = state.context.get("model_id")
                    
                    # Check if model_id is missing, ask for it first
                    if not model_id:
                        if not result.get("tool_name"):
                            result["tool_name"] = ""
                        result["conversation_stage"] = "collecting_model_id"
                        result["suggested_next_message"] = "No problem, I'll help you create new persona narratives. Please select your model ID from the dropdown."
                        result["frontend_action"] = "show_model_list"
                        
                        # Add needed_input to indicate frontend should show model ID selection
                        if not "needed_input" in result:
                            result["needed_input"] = {}
                        result["needed_input"]["show_model_list"] = True
                        result["needed_input"]["user_id"] = user_id
                        return state
                    
                    # Ready to create personas
                    if not result.get("tool_name"):
                        result["tool_name"] = "persona_summary"
                    result["conversation_stage"] = "creating_personas"
                    result["suggested_next_message"] = "I'll create new persona narratives for your model now."
                    
                    # Add parameters for tool execution
                    if not "parameters" in result:
                        result["parameters"] = {}
                    result["parameters"]["user_id"] = user_id
                    result["parameters"]["model_id"] = model_id

            # Check if user is asking for personalization (email, direct mail, etc.)
            personalization_requested = any(word in user_query_lower for word in ["personalization", "personalize", "email personalization", "direct mail", "digital ad"])
            email_requested = "email" in user_query_lower
            direct_mail_requested = any(phrase in user_query_lower for phrase in ["direct mail", "directmail", "physical mail"])
            digital_ad_requested = any(phrase in user_query_lower for phrase in ["digital ad", "digitad", "ad", "ads", "advertisement"])
            
            if personalization_requested and not "has_checked_personas" in state.context:
                # User is requesting personalization, need to check for personas first
                user_id = state.context.get("user_id")
                model_id = state.context.get("model_id")
                
                # Set conversation stage to checking personas
                if not result.get("tool_name"):
                    result["tool_name"] = ""
                result["conversation_stage"] = "checking_personas"
                result["suggested_next_message"] = "Before we create personalized content, I need to check if you have any existing persona narratives. Do you have persona narratives ready for this personalization?"
                
                # Mark in context that we've started the persona check
                state.context["has_checked_personas"] = True
                
                # Store the requested personalization type for later
                if email_requested:
                    state.context["requested_personalization"] = "email"
                elif direct_mail_requested:
                    state.context["requested_personalization"] = "direct_mail"  
                elif digital_ad_requested:
                    state.context["requested_personalization"] = "digital_ad"
                else:
                    state.context["requested_personalization"] = "generic"
                    
                return state
            
            # Check if we're in the checking_personas stage and the user has responded
            if "has_checked_personas" in state.context and "checking_personas" in last_stage and not "has_persona_response" in state.context:
                # User is responding to the question about having personas
                user_id = state.context.get("user_id")
                model_id = state.context.get("model_id")
                
                # Record that the user has responded to the persona question
                state.context["has_persona_response"] = True
                
                if has_affirmative:
                    # User says they have personas - let's verify
                    try:
                        persona_summaries = await self.db.get_persona_summaries(
                            user_id=str(user_id), 
                            model_id=str(model_id), 
                            conversation_id=str(state.context.get("conversation_id")), 
                            conversation_status=True
                        )
                        
                        personas_exist = persona_summaries and len(persona_summaries) > 0
                        state.context["personas_exist"] = personas_exist
                        
                        if personas_exist:
                            # They exist, let's move to selection
                            if not result.get("tool_name"):
                                result["tool_name"] = ""
                            result["conversation_stage"] = "selecting_persona"
                            result["suggested_next_message"] = "Great! Here are your existing persona narratives. Which one would you like to use for personalization?"
                            result["frontend_action"] = "show_persona_list"
                            
                            # Add needed_input for frontend
                            if not "needed_input" in result:
                                result["needed_input"] = {}
                            result["needed_input"]["show_persona_list"] = True
                            result["needed_input"]["user_id"] = user_id
                            result["needed_input"]["model_id"] = model_id
                        else:
                            # They don't exist, user was mistaken
                            if not result.get("tool_name"):
                                result["tool_name"] = "persona_summary"
                            result["conversation_stage"] = "correcting_no_personas"
                            result["suggested_next_message"] = "I don't actually see any existing persona narratives for this model ID. Let's create some personas first, and then we can do the personalization you requested."
                            result["frontend_action"] = "show_model_list"
                            
                            # Add needed_input for model selection
                            if not "needed_input" in result:
                                result["needed_input"] = {}
                            result["needed_input"]["show_model_list"] = True
                            result["needed_input"]["user_id"] = user_id
                    except Exception as e:
                        # Error checking personas, assume they don't exist
                        logger.error(f"Error checking personas: {str(e)}")
                        state.context["personas_exist"] = False
                        
                        # Suggest creating personas
                        if not result.get("tool_name"):
                            result["tool_name"] = "persona_summary"
                        result["conversation_stage"] = "suggesting_persona_creation"
                        result["suggested_next_message"] = "I'm having trouble checking for existing personas. Let's create new ones to be safe."
                        result["frontend_action"] = "show_model_list"
                        
                        # Add needed_input for model selection
                        if not "needed_input" in result:
                            result["needed_input"] = {}
                        result["needed_input"]["show_model_list"] = True
                        result["needed_input"]["user_id"] = user_id
                
                elif has_negative:
                    # User says they don't have personas - let's verify anyway
                    try:
                        persona_summaries = await self.db.get_persona_summaries(
                            user_id=str(user_id), 
                            model_id=str(model_id), 
                            conversation_id=str(state.context.get("conversation_id")), 
                            conversation_status=True
                        )
                        
                        personas_exist = persona_summaries and len(persona_summaries) > 0
                        state.context["personas_exist"] = personas_exist
                        
                        if personas_exist:
                            # They exist despite user saying no
                            if not result.get("tool_name"):
                                result["tool_name"] = ""
                            result["conversation_stage"] = "correcting_has_personas"
                            result["suggested_next_message"] = "Actually, I found existing persona narratives for this model ID. Would you like to use these existing personas or create new ones?"
                            result["frontend_action"] = "show_persona_list_with_recreate_option"
                            
                            # Add needed_input for frontend
                            if not "needed_input" in result:
                                result["needed_input"] = {}
                            result["needed_input"]["show_persona_list_with_recreate_option"] = True
                            result["needed_input"]["user_id"] = user_id
                            result["needed_input"]["model_id"] = model_id
                        else:
                            # They don't exist as user said
                            if not result.get("tool_name"):
                                result["tool_name"] = "persona_summary"
                            result["conversation_stage"] = "suggesting_persona_creation"
                            result["suggested_next_message"] = "Let's create persona narratives first, and then we can do the personalization you requested."
                            result["frontend_action"] = "show_model_list"
                            
                            # Add needed_input for model selection
                            if not "needed_input" in result:
                                result["needed_input"] = {}
                            result["needed_input"]["show_model_list"] = True
                            result["needed_input"]["user_id"] = user_id
                    except Exception as e:
                        # Error checking personas, assume they don't exist
                        logger.error(f"Error checking personas: {str(e)}")
                        state.context["personas_exist"] = False
                        
                        # Suggest creating personas
                        if not result.get("tool_name"):
                            result["tool_name"] = "persona_summary"
                        result["conversation_stage"] = "suggesting_persona_creation"
                        result["suggested_next_message"] = "I'm having trouble checking for existing personas. Let's create new ones to be safe."
                        result["frontend_action"] = "show_model_list"
                        
                        # Add needed_input for model selection
                        if not "needed_input" in result:
                            result["needed_input"] = {}
                        result["needed_input"]["show_model_list"] = True
                        result["needed_input"]["user_id"] = user_id
                
                return state
            
            # Check if we're continuing a personalization flow after persona check
            if "has_checked_personas" in state.context and "has_persona_response" in state.context and "personas_exist" in state.context:
                # We've already checked for personas and got a response
                user_id = state.context.get("user_id")
                model_id = state.context.get("model_id")
                personalization_type = state.context.get("requested_personalization", "email")
                
                # If personas exist, set up the appropriate personalization tool
                if state.context["personas_exist"]:
                    if personalization_type == "email":
                        # Set up email personalization
                        if not result.get("tool_name"):
                            result["tool_name"] = "email_personalization"
                        result["conversation_stage"] = "email_personalization"
                        result["suggested_next_message"] = "Now let's create your email personalization. I'll need a problem statement and which persona to use."
                    elif personalization_type == "direct_mail":
                        # Set up direct mail personalization
                        if not result.get("tool_name"):
                            result["tool_name"] = "directmail_personalization"
                        result["conversation_stage"] = "directmail_personalization"
                        result["suggested_next_message"] = "Now let's create your direct mail personalization. I'll need a problem statement and which persona to use."
                    elif personalization_type == "digital_ad":
                        # Set up digital ad personalization
                        if not result.get("tool_name"):
                            result["tool_name"] = "digitalad_personalization"
                        result["conversation_stage"] = "digitalad_personalization"
                        result["suggested_next_message"] = "Now let's create your digital ad personalization. I'll need a problem statement and which persona to use."
                else:
                    # Personas don't exist, suggest creating them first
                    if not result.get("tool_name"):
                        result["tool_name"] = "persona_summary"
                    result["conversation_stage"] = "suggesting_persona_creation"
                    result["suggested_next_message"] = "Before we can create personalized content, we need to build persona narratives first. Let's create some personas for your model."

            # Check for personalization channel selection after confirming personas exist
            if "selecting_personalization_channel" in last_stage:
                user_id = state.context.get("user_id") 
                model_id = state.context.get("model_id")
                
                # Check which channel the user selected
                if "email" in user_query_lower:
                    # Set up email personalization
                    if not result.get("tool_name"):
                        result["tool_name"] = "email_personalization"
                    result["conversation_stage"] = "email_personalization"
                    result["suggested_next_message"] = "Great! Let's create your email personalization. I'll need a problem statement and which persona to use."
                    
                    # Add needed_input for problem statement and persona selection
                    if not "needed_input" in result:
                        result["needed_input"] = {}
                    result["needed_input"]["get_problem_statement"] = True
                    result["needed_input"]["select_persona"] = True
                    result["needed_input"]["user_id"] = user_id
                    result["needed_input"]["model_id"] = model_id
                    
                elif "direct mail" in user_query_lower or "mail" in user_query_lower:
                    # Set up direct mail personalization
                    if not result.get("tool_name"):
                        result["tool_name"] = "directmail_personalization"
                    result["conversation_stage"] = "directmail_personalization"
                    result["suggested_next_message"] = "Great! Let's create your direct mail personalization. I'll need a problem statement and which persona to use."
                    
                    # Add needed_input for problem statement and persona selection
                    if not "needed_input" in result:
                        result["needed_input"] = {}
                    result["needed_input"]["get_problem_statement"] = True
                    result["needed_input"]["select_persona"] = True
                    result["needed_input"]["user_id"] = user_id
                    result["needed_input"]["model_id"] = model_id
                    
                elif "digital" in user_query_lower or "ad" in user_query_lower or "ads" in user_query_lower:
                    # Set up digital ad personalization
                    if not result.get("tool_name"):
                        result["tool_name"] = "digitalad_personalization"
                    result["conversation_stage"] = "digitalad_personalization"
                    result["suggested_next_message"] = "Great! Let's create your digital ad personalization. I'll need a problem statement and which persona to use."
                    
                    # Add needed_input for problem statement and persona selection
                    if not "needed_input" in result:
                        result["needed_input"] = {}
                    result["needed_input"]["get_problem_statement"] = True
                    result["needed_input"]["select_persona"] = True
                    result["needed_input"]["user_id"] = user_id
                    result["needed_input"]["model_id"] = model_id
                else:
                    # User didn't specify a valid channel
                    if not result.get("tool_name"):
                        result["tool_name"] = ""
                    result["conversation_stage"] = "selecting_personalization_channel"
                    result["suggested_next_message"] = "I need to know which channel you want to personalize for. Please choose from email, direct mail, or digital ads."
                
                return state

            # Check for user saying they have personas but don't want to recreate
            has_persona_no_recreate = ("already have" in user_query_lower and "persona" in user_query_lower and 
                                      any(phrase in user_query_lower for phrase in ["don't recreate", "do not recreate", 
                                                                                  "don't want to recreate", "no need to recreate"]))
            
            if has_persona_no_recreate:
                user_id = state.context.get("user_id")
                model_id = state.context.get("model_id")
                
                # Check if model_id is missing, ask for it first
                if not model_id:
                    if not result.get("tool_name"):
                        result["tool_name"] = ""
                    result["conversation_stage"] = "collecting_model_id_no_recreate"
                    result["suggested_next_message"] = "Great! I'll check your existing persona narratives. Please select your model ID from the dropdown."
                    result["frontend_action"] = "show_model_list"
                    
                    # Add needed_input to indicate frontend should show model ID selection
                    if not "needed_input" in result:
                        result["needed_input"] = {}
                    result["needed_input"]["show_model_list"] = True
                    result["needed_input"]["user_id"] = user_id
                    return state
                
                # If model_id is available, check if persona narratives exist
                try:
                    persona_summaries = await self.db.get_persona_summaries(
                        user_id=str(user_id), 
                        model_id=str(model_id), 
                        conversation_id=str(state.context.get("conversation_id")), 
                        conversation_status=True
                    )
                    
                    persona_exists = persona_summaries and len(persona_summaries) > 0
                    
                    if persona_exists:
                        # Personas exist, proceed to personalization channel selection
                        if not result.get("tool_name"):
                            result["tool_name"] = ""
                        result["conversation_stage"] = "selecting_personalization_channel"
                        result["suggested_next_message"] = "Perfect! I found your existing persona narratives. For which channel would you like to create personalization - email, direct mail, or digital ads?"
                        result["frontend_action"] = "show_persona_list"
                        
                        # Add needed_input for frontend
                        if not "needed_input" in result:
                            result["needed_input"] = {}
                        result["needed_input"]["show_persona_list"] = True
                        result["needed_input"]["user_id"] = user_id
                        result["needed_input"]["model_id"] = model_id
                    else:
                        # No personas found despite user claiming they exist
                        if not result.get("tool_name"):
                            result["tool_name"] = ""
                        result["conversation_stage"] = "correcting_no_personas"
                        result["suggested_next_message"] = "I've checked but don't actually see any existing persona narratives for this model ID. Would you like me to create some for you?"
                        result["frontend_action"] = "show_model_list"
                        
                        # Add needed_input for frontend
                        if not "needed_input" in result:
                            result["needed_input"] = {}
                        result["needed_input"]["show_model_list"] = True
                        result["needed_input"]["user_id"] = user_id
                except Exception as e:
                    logger.error(f"Error checking personas: {str(e)}")
                    # Handle error gracefully
                    if not result.get("tool_name"):
                        result["tool_name"] = ""
                    result["conversation_stage"] = "error_checking_personas"
                    result["suggested_next_message"] = "I'm having trouble checking your existing personas. Would you like to try creating new ones?"
                
                return state

            # Handle model_id selection from UI in collecting_model_id_no_recreate stage
            elif "collecting_model_id_no_recreate" in last_stage and model_id:
                # User provided model_id through UI, check for personas
                try:
                    persona_summaries = await self.db.get_persona_summaries(
                        user_id=str(user_id), 
                        model_id=str(model_id), 
                        conversation_id=str(state.context.get("conversation_id")), 
                        conversation_status=True
                    )
                    
                    persona_exists = persona_summaries and len(persona_summaries) > 0
                    
                    if persona_exists:
                        # Personas exist, proceed to personalization channel selection
                        if not result.get("tool_name"):
                            result["tool_name"] = ""
                        result["conversation_stage"] = "selecting_personalization_channel"
                        result["suggested_next_message"] = "Perfect! I found your existing persona narratives. For which channel would you like to create personalization - email, direct mail, or digital ads?"
                        result["frontend_action"] = "show_persona_list"
                        
                        # Add needed_input for frontend
                        if not "needed_input" in result:
                            result["needed_input"] = {}
                        result["needed_input"]["show_persona_list"] = True
                        result["needed_input"]["user_id"] = user_id
                        result["needed_input"]["model_id"] = model_id
                    else:
                        # No personas found despite user claiming they exist
                        if not result.get("tool_name"):
                            result["tool_name"] = ""
                        result["conversation_stage"] = "correcting_no_personas"
                        result["suggested_next_message"] = "I've checked but don't actually see any existing persona narratives for this model ID. Would you like me to create some for you?"
                        result["frontend_action"] = "show_model_list"
                        
                        # Add needed_input for frontend
                        if not "needed_input" in result:
                            result["needed_input"] = {}
                        result["needed_input"]["show_model_list"] = True
                        result["needed_input"]["user_id"] = user_id
                except Exception as e:
                    logger.error(f"Error checking personas: {str(e)}")
                    # Handle error gracefully
                    if not result.get("tool_name"):
                        result["tool_name"] = ""
                    result["conversation_stage"] = "error_checking_personas"
                    result["suggested_next_message"] = "I'm having trouble checking your existing personas. Would you like to try creating new ones?"
                
                return state

            await self.db.update_query_context(state.context['user_id'], state.context['conversation_id'], data_dict)
            # Log the result for debugging
            logger.info(f"Tool selection result: {result}")
            
            # Check if this is a general conversation (empty tool_name)
            if not result.get("tool_name"):
                logger.info("General conversation detected (empty tool_name)")
                state.error = "General conversation detected"
                # Store any needed_input for frontend actions
                if "needed_input" in result:
                    state.needed_input = result["needed_input"]
                if "frontend_action" in result:
                    state.frontend_action = result["frontend_action"]
                return state
                
            # Otherwise, set the tool decision
            state.tool_decision = result
            
            # Store any needed_input for frontend actions
            if "needed_input" in result:
                state.needed_input = result["needed_input"]
            if "frontend_action" in result:
                state.frontend_action = result["frontend_action"]
            
            # Log the detected tool and parameters
            logger.info(f"Detected tool: {result.get('tool_name')}")
            logger.info(f"Conversation stage: {result.get('conversation_stage', 'not specified')}")
            
            # When a user says they have personas, make sure we ask for model_id if it's not provided
            user_query_lower = state.query.lower()
            has_personas_phrases = ["have persona", "have narratives", "already have", "got persona", "got narratives"]
            user_indicates_has_personas = any(phrase in user_query_lower for phrase in has_personas_phrases)
            
            if user_indicates_has_personas and not state.context.get("model_id"):
                if not result.get("tool_name"):
                    result["tool_name"] = ""
                result["conversation_stage"] = "collecting_model_id_no_recreate"
                result["suggested_next_message"] = "Great! To check your existing persona narratives, I'll need your model ID. Please select it from the dropdown."
                result["frontend_action"] = "show_model_list"
                
                # Add needed_input to indicate frontend should show model ID selection
                if not "needed_input" in result:
                    result["needed_input"] = {}
                result["needed_input"]["show_model_list"] = True
                result["needed_input"]["user_id"] = state.context.get("user_id")
                
                # Also add model_id to missing params to ensure it's requested
                if not "required_parameters" in result:
                    result["required_parameters"] = []
                if "model_id" not in result["required_parameters"]:
                    result["required_parameters"].append("model_id")
                
                # Store any needed_input for frontend actions
                state.needed_input = result["needed_input"]
                if "frontend_action" in result:
                    state.frontend_action = result["frontend_action"]
                
                return state
            
        # except Exception as e:
        #     logger.error(f"Error in analyze_request: {str(e)}")
        #     state.error = str(e)
        
        return state
        
    async def _extract_parameters(self, state: GraphState) -> GraphState:
        """
        Extract parameters from the request
        
        Args:
            state: The current state
            
        Returns:
            Updated state
        """
        if state.error:
            return state
            
        # Get the analysis
        analysis = state.tool_decision or {}
        # Extract parameters
        parameters = {}

        # Check if tool_decision exists and has tool_name before accessing it
        if state.tool_decision is None or "tool_name" not in state.tool_decision:
            logger.error("Tool decision is None or missing tool_name")
            state.error = "Invalid tool decision"
            return state
            
        tool_name = state.tool_decision.get("tool_name", "")
        if not tool_name:
            logger.error("Empty tool_name in tool_decision")
            state.error = "Invalid tool name"
            return state
            
        # Make sure the tool_name exists in tool_parm_dict
        if tool_name not in tool_parm_dict:
            logger.error(f"Unknown tool: {tool_name}")
            state.error = f"Unknown tool: {tool_name}"
            return state
        
        # Get required parameters for this tool
        required_params = tool_parm_dict[tool_name]
        
        # First, check if any parameters are already in the conversation context
        conversation_context = {}
        try:
            # Get conversation context data
            conversation = await self.db.get_conversation(state.context.get('conversation_id'))
            if conversation:
                conversation_context = conversation.get('context', {})
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
        
        # Check if user explicitly requested to change any parameter
        # Look for phrases like "use different model_id" or "change model ID"
        user_query_lower = state.query.lower()
        request_new_model = any(phrase in user_query_lower for phrase in [
            "new model", "different model", "change model", "another model", 
            "new model id", "different model id", "change model id", "another model id"
        ])
        request_new_conversation = any(phrase in user_query_lower for phrase in [
            "new conversation", "different conversation", "change conversation", 
            "another conversation", "new chat", "start over"
        ])
        
        # Build the parameters dictionary from context and analysis
        missing_params = []
        for param in required_params:
            # Check if user requested to change this parameter specifically
            param_change_requested = False
            if param == "model_id" and request_new_model:
                param_change_requested = True
            elif param == "conversation_id" and request_new_conversation:
                param_change_requested = True
            
            # Priority 1: Use direct values from UI selections in state.context (request.model_id, etc.)
            # These should always take precedence as they are freshly provided from the UI
            if param in state.context and state.context[param] and state.context[param] != "":
                parameters[param] = state.context[param]
                logger.info(f"Using {param} from UI selection/request context: {parameters[param]}")
            
            # Priority 2: Try to get from tool_decision parameters if provided
            elif "parameters" in state.tool_decision and param in state.tool_decision["parameters"] and state.tool_decision["parameters"][param] != "":
                parameters[param] = state.tool_decision["parameters"][param]
                logger.info(f"Using {param} from tool_decision parameters: {parameters[param]}")
            
            # Priority 3: Try to get from conversation context if not explicitly requesting a change
            elif not param_change_requested and param in conversation_context and conversation_context[param] and conversation_context[param] != "":
                parameters[param] = conversation_context[param]
                logger.info(f"Using {param} from conversation context: {parameters[param]}")
            
            # If parameter not found anywhere, add to missing list
            else:
                # Parameter is missing, add to missing list
                missing_params.append(param)
                logger.info(f"Parameter {param} is missing, will prompt user")
        
        # Get detailed data about missing parameters
        data = {"current_context": {}, "missing": {}}
        if missing_params:
            try:
                data = await self.db.get_missing_context_fields(
                    state.context['user_id'], 
                    state.context['conversation_id'], 
                    missing_params
                )
                
                if data is None:
                    data = {"current_context": {}, "missing": {}}
                
                missing_params = list(data.get("missing", {}).keys())
                if not missing_params:
                    missing_params = []
            except Exception as e:
                logger.error(f"Error in get_missing_context_fields: {str(e)}")
                # Continue with existing missing_params list rather than failing
                logger.info(f"Continuing with missing_params: {missing_params}")
        
        # Update parameters with any context parameters already available
        if "current_context" in data:
            for k, v in data["current_context"].items():
                if k not in parameters and v and v != '':
                    parameters[k] = v
                    logger.info(f"Using {k} from current_context: {v}")
        
        # Set the extracted parameters and mark any missing parameters
        state.tool_decision["parameters"] = parameters
        state.missing_params = missing_params
        
        # Check if we have all required parameters
        if len(missing_params) == 0:
            state.validated = True
        
        return state
        
    async def _validate_parameters(self, state: GraphState) -> GraphState:
        """
        Validate that all required parameters are present
        
        Args:
            state: The current state
            
        Returns:
            Updated state
        """
        if state.error:
            return state
            
        # Get the analysis and parameters
        analysis = state.tool_decision or {}
        tool_name = analysis.get("tool_name", "")
        parameters = analysis.get("parameters", {})
        if not tool_name:
            state.error = "No tool specified"
            return state
            
        # Get the tool to check its required parameters
        tool_registry = ToolRegistry()
        if not tool_registry.has_tool(tool_name):
            state.error = f"Tool {tool_name} not found"
            return state
            
        tool = tool_registry.get_tool(tool_name)
        required_params = tool.get_required_params()
        
        # Check for missing parameters
        missing_params = []
        
        for param in required_params:
            if param not in parameters or not parameters[param]:
                missing_params.append(param)
                
        # Update the state
        if missing_params:
            state.missing_params = missing_params
        else:
            state.validated = True
            
        return state
        
    async def _execute_tool(self, state: GraphState) -> GraphState:
        """
        Execute the appropriate tool
        
        Args:
            state: The current state
            
        Returns:
            Updated state
        """
        
        if state.error or not state.validated:
            return state
        
        # Get the tool name and parameters
        analysis = state.tool_decision or {}
        tool_name = analysis.get("tool_name", "")
        parameters = analysis.get("parameters", {})
        
        # If tool_name is empty, it's a general conversation
        if not tool_name:
            logger.info("General conversation detected during tool execution")
            state.error = "General conversation detected"
            return state
        
        # try:
        if True:
            # Execute the tool
            tool_registry = ToolRegistry()
            if tool_registry.has_tool(tool_name):
                tool = tool_registry.get_tool(tool_name)

                # Pass parameters directly instead of unpacking
                result = await tool.execute(parameters)

                print("^^^^^^^^^^^^^^^^^", result)
                
                # Convert ToolResponse to dictionary for further processing
                result_dict = {
                    "status": result.status,
                    "message": result.message,
                    "data": result.data if result.data is not None else {},
                    "required_inputs": result.required_inputs
                }
                
                # Format the result for better readability if it's a successful result
                if result.status == "success" and result.data is not None:
                    data = result.data
                    
                    # Format persona data if this is a persona tool
                    if tool_name == "persona_summary" and "personas" in data:
                        personas = data.get("personas", {})
                        formatted_message = "### Persona Narratives\n\n"
                        
                        for persona_name, summary in personas.items():
                            formatted_message += f"#### {persona_name}\n"
                            formatted_message += f"{summary}\n\n"
                            
                        result_dict["message"] = formatted_message
                    
                    # Format email personalization data
                    elif tool_name == "email_personalization" and "personalization" in data:
                        personalization = data.get("personalization", [])
                        formatted_message = "### Email Personalization\n\n"
                        
                        for item in personalization:
                            persona_name = item.get("persona_name", "")
                            incentive = item.get("incentive", "")
                            call_to_action = item.get("call_to_action", "")
                            content = item.get("content", "")
                            
                            formatted_message += f"#### {persona_name}\n"
                            formatted_message += f"- **Incentive**\n\t- {incentive}\n"
                            formatted_message += f"- **Call to Action**\n\t- {call_to_action}\n"
                            formatted_message += f"- **Personalized Content**\n\t- {content}\n\n"
                            
                        result_dict["message"] = formatted_message
                
                state.tool_result = result_dict
            else:
                state.error = f"Tool {tool_name} not found"
        # except Exception as e:
        #     logger.error(f"Error executing tool: {str(e)}")
        #     state.error = str(e)
            
        return state
        
    def _format_response(self, state: GraphState) -> Dict[str, Any]:
        """
        Format the response from the graph
        
        Args:
            state: The current state
            
        Returns:
            The formatted response
        """
        # Add model_id to missing_params when it's clearly needed based on the message or stage
        conversation_stage = getattr(state, "conversation_stage", "")
        suggested_message = getattr(state, "suggested_next_message", "")
        
        # Check if we're asking for model_id but it's not in missing_params
        model_id_keywords = [
            "provide the model id", 
            "select your model id", 
            "choose a model id",
            "select the model",
            "which model"
        ]
        
        model_id_stages = [
            "collecting_model_id", 
            "collecting_model_id_no_recreate", 
            "exploring_options"
        ]
        
        is_asking_for_model_id = (
            any(keyword.lower() in suggested_message.lower() for keyword in model_id_keywords) or
            any(stage in conversation_stage for stage in model_id_stages) or
            "show_model_list" in (getattr(state, "frontend_action", "") or "")
        )
        
        # Make sure model_id is included in missing_params when needed
        if is_asking_for_model_id:
            missing_params = getattr(state, "missing_params", []) or []
            if "model_id" not in missing_params:
                if not missing_params:
                    missing_params = ["model_id"]
                else:
                    missing_params.append("model_id")
                state.missing_params = missing_params
                
                # Also update needed_input if it exists
                needed_input = getattr(state, "needed_input", {}) or {}
                if needed_input is None:
                    needed_input = {}
                
                needed_input["show_model_list"] = True
                user_id = state.context.get("user_id") if hasattr(state, "context") else None
                if user_id:
                    needed_input["user_id"] = user_id
                
                state.needed_input = needed_input
                
                # Force status to input_required when asking for model_id
                error_state = getattr(state, "error", None)
                if error_state is None and not getattr(state, "tool_result", None):
                    return {
                        "status": "input_required",
                        "message": suggested_message,
                        "data": None,
                        "tool_used": state.tool_decision.get("tool_name") if hasattr(state, "tool_decision") and state.tool_decision else None,
                        "required_inputs": missing_params,
                        "needed_input": needed_input,
                        "frontend_action": "show_model_list",
                        "conversation_id": state.context.get("conversation_id") if hasattr(state, "context") else None,
                        "conversation_name": getattr(state, "conversation_name", None),
                        "is_conversation_end": getattr(state, "is_conversation_end", False),
                        "last_message": getattr(state, "last_message", ""),
                        "conversation_stage": conversation_stage,
                        "suggested_next_message": suggested_message
                    }
        
        # Check if error exists and is "General conversation detected"
        error = getattr(state, "error", None)
        if error and error == "General conversation detected":
            logger.info("Delegating to chat processor for general conversation")
            return {
                "status": "chat",
                "message": "",
                "data": None,
                "tool_used": None,
                "required_inputs": None,
                "needed_input": getattr(state, "needed_input", None),
                "frontend_action": getattr(state, "frontend_action", None),
                "conversation_id": state.context.get("conversation_id") if hasattr(state, "context") else None,
                "conversation_name": getattr(state, "conversation_name", None),
                "is_conversation_end": getattr(state, "is_conversation_end", False),
                "last_message": getattr(state, "last_message", ""),
                "conversation_stage": getattr(state, "conversation_stage", "initial_greeting"),
                "suggested_next_message": getattr(state, "suggested_next_message", "How can I help you with personalization today?")
            }
            
        # If there was a tool execution error
        if error and error != "General conversation detected":
            logger.error(f"Tool execution error: {error}")
            return {
                "status": "error",
                "message": f"An error occurred: {error}",
                "data": None,
                "tool_used": state.tool_decision.get("tool_name") if hasattr(state, "tool_decision") and state.tool_decision else None,
                "required_inputs": None,
                "needed_input": getattr(state, "needed_input", None),
                "frontend_action": getattr(state, "frontend_action", None),
                "conversation_id": state.context.get("conversation_id") if hasattr(state, "context") else None,
                "conversation_name": getattr(state, "conversation_name", None),
                "is_conversation_end": getattr(state, "is_conversation_end", False),
                "last_message": getattr(state, "last_message", ""),
                "conversation_stage": getattr(state, "conversation_stage", "initial_greeting"),
                "suggested_next_message": getattr(state, "suggested_next_message", "How can I help you with personalization today?")
            }
        
        # If there are missing parameters
        missing_params = getattr(state, "missing_params", None)
        if missing_params and len(missing_params) > 0:
            logger.info(f"Missing parameters: {missing_params}")
            return {
                "status": "input_required",
                "message": f"Additional information needed: {', '.join(missing_params)}",
                "data": None,
                "tool_used": state.tool_decision.get("tool_name") if hasattr(state, "tool_decision") and state.tool_decision else None,
                "required_inputs": missing_params,
                "needed_input": getattr(state, "needed_input", None),
                "frontend_action": getattr(state, "frontend_action", None),
                "conversation_id": state.context.get("conversation_id") if hasattr(state, "context") else None,
                "conversation_name": getattr(state, "conversation_name", None),
                "is_conversation_end": getattr(state, "is_conversation_end", False),
                "last_message": getattr(state, "last_message", ""),
                "conversation_stage": getattr(state, "conversation_stage", "initial_greeting"),
                "suggested_next_message": getattr(state, "suggested_next_message", "How can I help you with personalization today?")
            }
        
        # If there was a successful tool execution
        tool_result = getattr(state, "tool_result", None)
        if tool_result:
            tool_decision = getattr(state, "tool_decision", {})
            tool_name = tool_decision.get("tool_name") if tool_decision else None
            logger.info(f"Successful tool execution: {tool_name}")
            
            # If tool_result contains needed_input, use it. Otherwise, use state.needed_input
            result_needed_input = tool_result.get("needed_input", getattr(state, "needed_input", None))
            
            return {
                "status": tool_result.get("status", "success"),
                "message": tool_result.get("message", ""),
                "data": tool_result.get("data"),
                "tool_used": tool_name,
                "required_inputs": None,
                "needed_input": result_needed_input,
                "frontend_action": tool_result.get("frontend_action", getattr(state, "frontend_action", None)),
                "conversation_id": state.context.get("conversation_id") if hasattr(state, "context") else None,
                "conversation_name": getattr(state, "conversation_name", None),
                "is_conversation_end": getattr(state, "is_conversation_end", False),
                "last_message": getattr(state, "last_message", ""),
                "conversation_stage": getattr(state, "conversation_stage", "initial_greeting"),
                "suggested_next_message": getattr(state, "suggested_next_message", "How can I help you with personalization today?")
            }
        
        # Default response
        logger.info("Returning default response")
        return {
            "status": "error",
            "message": "No response could be generated",
            "data": None,
            "tool_used": None,
            "required_inputs": None,
            "needed_input": getattr(state, "needed_input", None),
            "frontend_action": getattr(state, "frontend_action", None),
            "conversation_id": state.context.get("conversation_id") if hasattr(state, "context") else None,
            "conversation_name": getattr(state, "conversation_name", None),
            "is_conversation_end": getattr(state, "is_conversation_end", False),
            "last_message": getattr(state, "last_message", ""),
            "conversation_stage": getattr(state, "conversation_stage", "initial_greeting"),
            "suggested_next_message": getattr(state, "suggested_next_message", "How can I help you with personalization today?")
        }
        
    async def process_request(self, request: UserRequest, conversation_last_message) -> Dict[str, Any]:
        """
        Process a user request through the graph
        
        Args:
            request: The user request
            conversation_last_message: The last message from the conversation
            
        Returns:
            The processing result
        """
        # Create the initial state
        state = GraphState(
            query=request.query,
            context={
                "user_id": request.user_id,
                "model_id": request.model_id,
                "session_token": request.session_token,
                "conversation_id": request.conversation_id,
                "conversation_name": request.conversation_name
            },
            messages=[],
            conversation_name=request.conversation_name,
            is_conversation_end=request.is_conversation_end,
            last_message=conversation_last_message,
            conversation_stage="initial_greeting",
            suggested_next_message="How can I help you with personalization today?"
        )
        
        # Run the graph
        try:
            result = await self.graph.ainvoke(state)
            
            # Convert result to GraphState if needed
            if not isinstance(result, GraphState):
                result_dict = dict(result)
                result = GraphState(
                    query=result_dict.get("query", request.query),
                    context=result_dict.get("context", {}),
                    messages=result_dict.get("messages", []),
                    tool_decision=result_dict.get("tool_decision"),
                    tool_result=result_dict.get("tool_result"),
                    error=result_dict.get("error"),
                    missing_params=result_dict.get("missing_params"),
                    validated=result_dict.get("validated", False),
                    conversation_name=result_dict.get("conversation_name", request.conversation_name),
                    is_conversation_end=result_dict.get("is_conversation_end", request.is_conversation_end),
                    last_message=result_dict.get("last_message", conversation_last_message),
                    conversation_stage=result_dict.get("conversation_stage", "initial_greeting"),
                    suggested_next_message=result_dict.get("suggested_next_message", "How can I help you with personalization today?"),
                    needed_input=result_dict.get("needed_input"),
                    frontend_action=result_dict.get("frontend_action")
                )
            
            # Format the response
            response = self._format_response(result)
            
            # Include the conversation stage and suggested message for frontend
            if not response.get("conversation_stage"):
                response["conversation_stage"] = result.conversation_stage
                
            if not response.get("suggested_next_message"):
                response["suggested_next_message"] = result.suggested_next_message
            
            return response
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"An error occurred while processing your request: {str(e)}",
                "data": None,
                "conversation_id": request.conversation_id,
                "conversation_name": request.conversation_name,
                "is_conversation_end": request.is_conversation_end,
                "conversation_stage": "error",
                "suggested_next_message": "I'm sorry, but there was an error processing your request. Would you like to try again?"
            } 