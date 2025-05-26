import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

import requests
from bson import ObjectId
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from pydantic import BaseModel, Field

from app.agent.chat_handler import ChatHandler
from app.models.model import ApiResponse, ToolResponse, UserRequest
from app.utils.db import MongoDB, convert_objectid
from app.utils.langfush_utils import config_llm_callback, get_prompt_config
from app.utils.tool_registry import ToolRegistry

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API keys
OPENAI_API_KEY = os.getenv("openai_key")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
ANTHROPIC_API_KEY = os.getenv("anthropic_key")
LANGFUSE_PUBLIC_KEY = os.getenv("langfuse_public_key")
LANGFUSE_SECRET_KEY = os.getenv("langfuse_secret_key")
LANGFUSE_HOST = os.getenv("langfuse_host")

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST
)

langfuse_handler = CallbackHandler(
    public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST
)
# Tool parameter mapping
TOOL_PARAM_DICT = {
    "persona_summary": ["user_id", "personalization_data_type", "model_id"],
    "email_personalization": [
        "user_id",
        "model_id",
        "personalization_data_type",
        "problem_statement",
        "persona_name",
        "audience_id",
    ],
    "directmail_personalization": [
        "user_id",
        "model_id",
        "personalization_data_type",
        "problem_statement",
        "persona_name",
        "audience_id",
    ],
    "digitalad_personalization": [
        "user_id",
        "model_id",
        "personalization_data_type",
        "problem_statement",
        "persona_name",
        "audience_id",
    ],
    # "show_persona_summary": ["user_id", "model_id", "persona_name"],
    "persona_explain": ["user_id", "model_id"],
    "journy_tool": [
        "user_problem_statement",
        "budget",
        "target_id",
        "session_token",
        "name",
        "date",
        "persona_files",
        "model_id",
    ],
}

TOOL_SELECTION_PROMPT, config_tool_selection = get_prompt_config(
    prompt_tag="tool selection prompt", label="latest"
)
extraction_prompt, config_journy_action = get_prompt_config(
    prompt_tag="extraction prompt", label="stage_v1"
)

# ===========================class for pydantic model================================


class DataTypeSelection(BaseModel):
    """Model representing the persona data type selection results"""

    persona_creation_intent: bool = Field(default=False)
    data_type_specified: str = Field(default="")
    should_ask_data_type: bool = Field(default=False)
    recommended_next_message: str = Field(default="")


class ToolSelection(BaseModel):
    """Model representing tool selection decision"""

    tool_name: str = Field(default="")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_parameters: List[str] = Field(default_factory=list)
    conversation_stage: str = Field(default="initial_greeting")
    suggested_next_message: str = Field(
        default="How can I help you with personalization today?"
    )


class JournyAction(BaseModel):
    """Model representing tool selection decision"""

    action_name: str = Field(default="name of the action from user input")


class ProblemStatementAnalysis(BaseModel):
    valid: bool = Field(
        description="Indicates whether the problem statement is complete and valid."
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of fields that are missing or incomplete.",
    )
    suggestions: Dict[str, str] = Field(
        default_factory=lambda: {
            "User Background": "",
            "Campaign Goal": "",
            "Incentive Offered": "",
            "Call to Action": "",
        },
        description="Suggested or completed text for missing or weak parts of the problem statement.",
    )


class ProblemStatementAnalysisJourny(BaseModel):
    valid: bool = Field(
        description="Indicates whether the problem statement is complete and valid."
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of fields that are missing or incomplete.",
    )
    suggestions: Dict[str, str] = Field(
        default_factory=lambda: {
            "Objective": "",
            "Audience Setup": "",
            "Channel Flow Steps": "",
            "Nurture Journey Builder": "",
            "Compliance & Approvals": "",
            "KPI Cards": "",
            "Data & Settings Panel": "",
        },
        description="Suggested or completed text for missing or weak parts of the problem statement.",
    )


# ===========================END: class for pydantic model================================


class MainAgent:
    """
    Main agent that coordinates all tools and handles requests
    """

    def __init__(self, mongodb: Optional[MongoDB] = None):
        """Initialize the main agent with tool registry and database connection"""
        self.tool_registry = ToolRegistry()
        self.mongodb = mongodb
        self.chat_handler = ChatHandler()

        # ===================tool selection========================================================
        # gpt-4o, gpt-4o-mini, claude-3-5-sonnet-latest, claude-3-5-haiku-latest, claude-3-7-sonnet-latest
        self.llm_tool_selection = ChatOpenAI(
            temperature=config_tool_selection["temperature"],
            model=config_tool_selection["model"],
            api_key=OPENAI_API_KEY,
        )
        # Create parser for tool selection output
        self.parser = JsonOutputParser(pydantic_object=ToolSelection)
        self.tool_selection_prompt = PromptTemplate(
            template=TOOL_SELECTION_PROMPT,
            input_variables=["query", "user_history"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        # Create tool selection chain
        self.tool_selection_chain = (
            self.tool_selection_prompt | self.llm_tool_selection | self.parser
        )
        # =================end tool selection======================================================

        # ================================journy action selection===================================
        self.llm_journy_action = ChatAnthropic(
            temperature=config_journy_action["temperature"],
            model=config_journy_action["model"],
            api_key=ANTHROPIC_API_KEY,
        )
        self.parser_journy = JsonOutputParser(pydantic_object=JournyAction)
        self.journy_chain_prompt = PromptTemplate(
            template=extraction_prompt,
            input_variables=["query"],
            partial_variables={
                "format_instructions": self.parser_journy.get_format_instructions()
            },
        )
        self.journy_chain = (
            self.journy_chain_prompt | self.llm_journy_action | self.parser_journy
        )
        # ===============================end journy action selection================================

    def get_context_msg(self, query, conversation_history: List[Dict[str, Any]] = None):
        context_messages = []
        last_6_messages = []

        if conversation_history and len(conversation_history) > 0:
            # Get last 6 messages for better context understanding
            last_6_messages = (
                conversation_history[-50:]
                if len(conversation_history) >= 50
                else conversation_history
            )

            # Process the messages to build context
            for i, message in enumerate(last_6_messages):
                role = message.get("role", "")
                content = message.get("content", "").lower()
                problem_statement_flag = message.get("problem_statement", False)
                if not problem_statement_flag:
                    context_messages.append(f"{role}: {content}")

            context = "\n".join(context_messages) if context_messages else ""
            match = query.lower().find("problem statement:")
            print("not in the match", query)
            if match != -1:
                print("here in the problem")
                query = ""
            if context:
                context += f"\n\nUser query: {query}"
            else:
                context = f"User query: {query}"
            return context

    async def select_tool(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]] = None,
        conversation_id: str = None,
        user: str = None,
    ) -> ToolSelection:
        """
        Select the appropriate tool based on the user query and conversation history

        Args:
            query: The user query
            conversation_history: Optional conversation history
            conversation_id: Optional conversation id
            user: Optional user id

        Returns:
            ToolSelection object with the selected tool and parameters
        """
        try:
            # Get the last assistant message if conversation history exists
            last_assistant_message = ""

            # Check if the last message suggests personalization after persona creation
            persona_creation_completed = False
            wants_persona_narrative = False
            model_id_requested = False
            model_id_provided = None

            # Extract context from last 6 messages if available
            context_messages = []
            last_6_messages = []

            if conversation_history and len(conversation_history) > 0:
                # Get last 6 messages for better context understanding
                last_6_messages = (
                    conversation_history[-50:]
                    if len(conversation_history) >= 50
                    else conversation_history
                )
                # Process the messages to build context
                for i, message in enumerate(last_6_messages):
                    role = message.get("role", "")
                    content = message.get("content", "").lower()
                    problem_statement_flag = message.get("problem_statement", False)
                    if not problem_statement_flag:
                        context_messages.append(f"{role}: {content}")

                    # Check for persona narrative creation intent
                    if role == "user" and any(
                        phrase in content
                        for phrase in [
                            "persona narrative",
                            "create persona",
                            "generate persona",
                            "persona creation",
                            "let's do persona",
                            "make a persona",
                        ]
                    ):
                        wants_persona_narrative = True
                        logger.info("User wants to create a persona narrative")

                    # Check for model_id request
                    if role == "assistant" and any(
                        phrase in content
                        for phrase in [
                            "model id",
                            "provide your model id",
                            "what is your model id",
                            "which model id",
                            "need to know which model",
                        ]
                    ):
                        model_id_requested = True
                        logger.info("Assistant requested model_id")

                    # Check if user provided model_id after being asked
                    if (
                        role == "user"
                        and model_id_requested
                        and i > 0
                        and last_6_messages[i - 1].get("role") == "assistant"
                    ):
                        # Try to extract model_id using regex
                        model_id_match = re.search(
                            r"model[\s_]?id[:\s]*([^\s,\.]+)|model[:\s]*([^\s,\.]+)",
                            content,
                        )
                        if model_id_match:
                            model_id_provided = model_id_match.group(
                                1
                            ) or model_id_match.group(2)
                            logger.info(f"User provided model_id: {model_id_provided}")
                        elif content.strip() and not (
                            "personalization" in content
                            or "journey" in content
                            or "audience" in content
                            or "summary" in content
                        ):
                            # If content is not empty and doesn't contain other tool keywords,
                            # assume the content itself is the model_id
                            model_id_provided = content.strip()
                            logger.info(
                                f"Assuming user content is model_id: {model_id_provided}"
                            )

                    # Check for last assistant message
                    if role == "assistant":
                        last_assistant_message = content
                        # Check if this message indicates successful persona creation
                        if (
                            "successfully created the persona narrative" in content
                            and "would you like to create personalized content"
                            in content
                        ):
                            persona_creation_completed = True

                # Look for parameter values mentioned in previous messages
                parameter_values = self._extract_parameters_from_history(
                    last_6_messages
                )

            # If the query is empty but user wanted persona narrative and provided model_id,
            # we should create the persona narrative
            if (
                (not query or query.strip() == "")
                and wants_persona_narrative
                and model_id_requested
                and model_id_provided
            ):
                logger.info(
                    "Empty query with persona narrative intent and model_id provided. Creating persona narrative."
                )
                return ToolSelection(
                    tool_name="persona_summary",
                    parameters={"model_id": model_id_provided},
                    required_parameters=[],
                    conversation_stage="creating_persona_narrative",
                    suggested_next_message="Creating your persona narrative...",
                )

            # Extract personalization_data_type from the query if mentioned
            personalization_data_type = None
            if any(
                phrase in query.lower()
                for phrase in [
                    "persona data",
                    "use persona",
                    "with persona",
                    "using persona",
                ]
            ):
                personalization_data_type = "persona data"
            elif any(
                phrase in query.lower()
                for phrase in [
                    "audience data",
                    "use audience",
                    "with audience",
                    "using audience",
                ]
            ):
                personalization_data_type = "audience data"

            # Combine context_messages with user query for better tool selection
            context = "\n".join(context_messages) if context_messages else ""
            print("??????????????????????????", query)
            print("?????????????????????????? context", context)
            match = query.lower().find("problem statement:")
            if match != -1:
                print("here in the problem")
                query = ""
            print("??????????????????????????", query)
            print("^^^^^^^^^^^^^^^^^ context", context)
            # Check if this is a general personalization request
            is_general_personalization = False
            if any(
                phrase in query.lower()
                for phrase in [
                    "personalization",
                    "personalized content",
                    "personalize",
                    "create personalized",
                    "make personalized",
                    "do personalization",
                ]
            ) and not any(
                specific_type in query.lower()
                for specific_type in [
                    "email",
                    "mail",
                    "direct mail",
                    "directmail",
                    "digital ad",
                    "digitalad",
                    "ad",
                ]
            ):
                is_general_personalization = True
                logger.info(
                    "Detected general personalization request without specific type"
                )

            # If this is a general personalization request, ask for specific type
            if is_general_personalization:
                return ToolSelection(
                    tool_name="",  # No specific tool yet
                    parameters={},
                    required_parameters=[],
                    conversation_stage="selecting_personalization_type",
                    suggested_next_message="What type of personalization would you like to create? Email, Direct Mail, or Digital Ad?",
                )

            config = config_llm_callback(
                run_name="agent_tool_selection",
                tag="tool_selection",
                conversation_id=conversation_id,
                user_id=user,
            )
            # Run the tool selection chain
            selection_dict = self.tool_selection_chain.invoke(
                {"query": query, "user_history": context}, config=config
            )
            logger.info(f"Tool selection result: {selection_dict}")

            # Create a ToolSelection object from the dictionary
            # Make sure to handle case where any expected field might be missing
            tool_selection = ToolSelection(
                tool_name=selection_dict.get("tool_name", ""),
                parameters=selection_dict.get("parameters", {}),
                required_parameters=selection_dict.get("required_parameters", []),
                conversation_stage=selection_dict.get(
                    "conversation_stage", "initial_greeting"
                ),
                suggested_next_message=selection_dict.get(
                    "suggested_next_message",
                    "How can I help you with personalization today?",
                ),
            )

            # If model_id was provided earlier but not detected by the LLM, add it
            if model_id_provided and "model_id" not in tool_selection.parameters:
                tool_selection.parameters["model_id"] = model_id_provided
                logger.info(
                    f"Added previously provided model_id={model_id_provided} to parameters"
                )

            # Fill any empty parameters from conversation history
            if conversation_history and len(conversation_history) > 0:
                tool_selection.parameters = self._fill_empty_parameters(
                    tool_selection.parameters,
                    parameter_values,
                    tool_selection.tool_name,
                )

            # Special handling for empty query with model_id context
            if (
                (not query or query.strip() == "")
                and model_id_provided
                and model_id_requested
            ):
                # If user wanted to create a persona narrative and then provided model_id,
                # ensure we're using the persona_summary tool
                if wants_persona_narrative:
                    tool_selection.tool_name = "persona_summary"
                    if "model_id" not in tool_selection.parameters:
                        tool_selection.parameters["model_id"] = model_id_provided
                    logger.info(
                        "Forcing persona_summary tool for empty query after model_id was provided"
                    )

            # Explicitly add personalization_data_type if present in the query but not detected by LLM
            if (
                personalization_data_type
                and "personalization" in tool_selection.tool_name
            ):
                # Add personalization_data_type to parameters
                tool_selection.parameters["personalization_data_type"] = (
                    personalization_data_type
                )
                logger.info(
                    f"Added personalization_data_type={personalization_data_type} from query"
                )

                # Remove personalization_data_type from required_parameters if present
                if "personalization_data_type" in tool_selection.required_parameters:
                    tool_selection.required_parameters.remove(
                        "personalization_data_type"
                    )

                # Adjust required parameters based on personalization_data_type
                if personalization_data_type == "persona data":
                    # For persona data: model_id is required, audience_id is optional
                    if (
                        "model_id" not in tool_selection.parameters
                        and "model_id" not in tool_selection.required_parameters
                    ):
                        tool_selection.required_parameters.append("model_id")

                    # Remove audience_id from required parameters if present - it's optional for persona data
                    if "audience_id" in tool_selection.required_parameters:
                        tool_selection.required_parameters.remove("audience_id")

                elif personalization_data_type == "audience data":
                    # For audience data: audience_id is required, model_id and persona_name are optional
                    if (
                        "audience_id" not in tool_selection.parameters
                        and "audience_id" not in tool_selection.required_parameters
                    ):
                        tool_selection.required_parameters.append("audience_id")

                    # Remove model_id and persona_name from required parameters if present - they're optional for audience data
                    if "model_id" in tool_selection.required_parameters:
                        tool_selection.required_parameters.remove("model_id")
                    if "persona_name" in tool_selection.required_parameters:
                        tool_selection.required_parameters.remove("persona_name")

            # If user just completed persona creation and now wants to create personalization content
            if persona_creation_completed and query.lower().strip() in [
                "yes",
                "yes please",
                "sure",
                "let's do it",
                "ok",
                "okay",
                "let's create personalized content",
            ]:
                # Extract the personalization type from the query or default to email
                personalization_type = "email_personalization"
                if "email" in query.lower():
                    personalization_type = "email_personalization"
                elif "direct mail" in query.lower() or "directmail" in query.lower():
                    personalization_type = "directmail_personalization"
                elif "digital ad" in query.lower() or "digitalad" in query.lower():
                    personalization_type = "digitalad_personalization"

                # Override the tool selection to use the appropriate personalization tool
                tool_selection.tool_name = personalization_type
                # Since this follows persona creation, we know we're using persona data
                tool_selection.parameters["personalization_data_type"] = "persona data"
                logger.info(
                    f"Overriding tool selection to {personalization_type} based on post-persona creation context"
                )

            return tool_selection
        except Exception as e:
            logger.error(f"Error in tool selection: {str(e)}", exc_info=True)
            # Return default tool selection (no tool)
            return ToolSelection()

    def _extract_parameters_from_history(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract potential parameter values from conversation history

        Args:
            messages: List of conversation messages

        Returns:
            Dictionary of parameter names and their values
        """
        parameter_values = {}

        param_patterns = {
            "personalization_data_type": [
                r"using (persona|audience) data",
                r"with (persona|audience) data",
                r"(persona|audience) data",
            ],
        }

        for message in messages:
            content = message.get("content", "")

            # Process each parameter pattern
            for param_name, patterns in param_patterns.items():
                for pattern in patterns:
                    matches = re.search(pattern, content, re.IGNORECASE)
                    if matches:
                        value = matches.group(1).strip()
                        # Handle special case for personalization_data_type
                        if param_name == "personalization_data_type":
                            value = f"{value} data"
                        parameter_values[param_name] = value
                        break

            # todo: solve this accept dait in utc format only
            # Special case for journey parameters
            if "budget" not in parameter_values:
                budget_match = re.search(r"(\$|USD|budget:?\s*)(\d[\d,]*)", content)
                if budget_match:
                    budget_str = budget_match.group(2).replace(",", "")
                    try:
                        parameter_values["budget"] = int(budget_str)
                    except ValueError:
                        pass

            # Extract date if in YYYY-MM-DD format
            if "date" not in parameter_values:
                date_match = re.search(r"(\d{4}-\d{2}-\d{2})", content)
                if date_match:
                    parameter_values["date"] = date_match.group(1)

        return parameter_values

    def _fill_empty_parameters(
        self,
        current_params: Dict[str, Any],
        history_params: Dict[str, Any],
        tool_name: str,
    ) -> Dict[str, Any]:
        """
        Fill empty parameters in current_params with values from history_params

        Args:
            current_params: Current parameters
            history_params: Parameters extracted from history
            tool_name: Name of the tool being used

        Returns:
            Updated parameters dictionary
        """
        # Copy the current parameters to avoid modifying the original
        updated_params = current_params.copy()

        # Which parameters to consider for this tool
        relevant_params = TOOL_PARAM_DICT.get(tool_name, [])

        for param in relevant_params:
            # If parameter is empty or missing in current_params but exists in history_params
            if (
                param not in updated_params or not updated_params[param]
            ) and param in history_params:
                updated_params[param] = history_params[param]
                logger.info(
                    f"Filled empty parameter {param}={history_params[param]} from conversation history"
                )

        return updated_params

    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> ToolResponse:
        """
        Execute a tool with the given parameters

        Args:
            tool_name: The name of the tool to execute
            parameters: The parameters for the tool

        Returns:
            ToolResponse object with the result
        """
        if not self.tool_registry.has_tool(tool_name):
            return ToolResponse(
                status="error", message=f"Tool '{tool_name}' not found", data=None
            )

        tool = self.tool_registry.get_tool(tool_name)
        logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
        return await tool.execute(parameters)

    async def check_missing_parameters(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> List[str]:
        """
        Check for missing parameters and try to retrieve them from conversation context

        Args:
            tool_name: The name of the tool
            parameters: The current parameters
            user_id: The user ID
            conversation_id: Optional conversation ID

        Returns:
            List of missing parameter names
        """
        # If tool not registered, return empty list
        if not self.tool_registry.has_tool(tool_name):
            return []

        # Get tool and required parameters
        tool = self.tool_registry.get_tool(tool_name)
        required_params = tool.get_required_params()
        missing_params = []

        if tool_name in ["persona_summary"]:
            print(
                "personalization_data_type gggggggggg66666666666666666666666",
                parameters.get("personalization_data_type", ""),
            )
            personalization_data_type = parameters.get(
                "personalization_data_type", ""
            ).lower()
            if personalization_data_type == "audience data":
                if "model_id" in required_params:
                    required_params.remove("model_id")
            if personalization_data_type == "persona data":
                if "audience_id" in required_params:
                    required_params.remove("audience_id")
        # For personalization tools, adjust required parameters based on personalization_data_type
        personalization_tools = [
            "email_personalization",
            "directmail_personalization",
            "digitalad_personalization",
        ]
        if tool_name in personalization_tools:
            personalization_data_type = parameters.get(
                "personalization_data_type", ""
            ).lower()

            if personalization_data_type == "audience data":
                # For audience data, model_id and persona_name are optional
                if "model_id" in required_params:
                    required_params.remove("model_id")
                if "persona_name" in required_params:
                    required_params.remove("persona_name")
                logger.info(
                    f"Using audience data - model_id and persona_name are optional"
                )

            elif personalization_data_type == "persona data":
                # For persona data, audience_id is optional
                if "audience_id" in required_params:
                    required_params.remove("audience_id")
                logger.info(f"Using persona data - audience_id is optional")

        # For journey tool operations that need journey_id or mongo_id, first check if we have an action
        if tool_name == "journy_tool" and "action" in parameters:
            action = parameters.get("action", "")

            # For actions that need journey_id or mongo_id, try to fetch them from context
            if action in [
                "get_journey_status",
                "get_journey_report",
                "check_journey_report",
            ]:
                # Check if we're missing either journey_id or mongo_id based on the action
                journey_id_needed = (
                    action in ["get_journey_report", "check_journey_report"]
                    and "journey_id" not in parameters
                )
                mongo_id_needed = (
                    action == "get_journey_status" and "mongo_id" not in parameters
                )

                # If we need either ID and have conversation context, try to retrieve from context
                if (
                    (journey_id_needed or mongo_id_needed)
                    and conversation_id
                    and user_id
                    and self.mongodb
                ):
                    try:
                        # Check if we have access to the context methods
                        if hasattr(self.mongodb, "get_conversation"):
                            # Get conversation to access the context
                            conversation = await self.mongodb.get_conversation(
                                conversation_id=conversation_id
                            )

                            if conversation and "context" in conversation:
                                context = conversation["context"]

                                # Check if we can retrieve the needed IDs
                                if journey_id_needed and "journey_id" in context:
                                    parameters["journey_id"] = context["journey_id"]
                                    logger.info(
                                        f"Retrieved journey_id={context['journey_id']} from context"
                                    )

                                if mongo_id_needed and "mongo_id" in context:
                                    parameters["mongo_id"] = context["mongo_id"]
                                    logger.info(
                                        f"Retrieved mongo_id={context['mongo_id']} from context"
                                    )
                    except Exception as e:
                        logger.error(
                            f"Error retrieving journey IDs from context: {str(e)}"
                        )

        # Check each required parameter
        for param in required_params:
            if (
                param not in parameters
                or not parameters[param]
                or (
                    param == "persona_name"
                    and isinstance(parameters[param], str)
                    and parameters[param].strip() == ""
                )
            ):
                # Try to retrieve from database if mongodb is available
                if self.mongodb and user_id and conversation_id:
                    try:
                        # Check if the method exists in the MongoDB class
                        if hasattr(self.mongodb, "get_context_value"):
                            context_value = await self.mongodb.get_context_value(
                                user_id, conversation_id, param
                            )
                            if context_value:
                                parameters[param] = context_value
                            else:
                                missing_params.append(param)
                        # If get_context_value doesn't exist, try to get from conversation history
                        elif hasattr(self.mongodb, "get_conversation_history"):
                            # Fallback: Try to extract from conversation history
                            logger.info(
                                f"get_context_value method not found, trying to find {param} in conversation history"
                            )
                            history = await self.mongodb.get_conversation_history(
                                conversation_id=conversation_id, limit=0
                            )
                            # Simple extraction logic - this can be improved
                            param_value = None
                            for message in reversed(history):
                                content = message.get("content", "")
                                # Very basic extraction - looking for param=value pattern
                                if (
                                    f"{param}=" in content
                                    or f"{param}:" in content
                                    or f"{param} =" in content
                                    or f"{param} :" in content
                                ):
                                    # Attempt to extract value
                                    try:
                                        parts = content.split(f"{param}=")
                                        if len(parts) > 1:
                                            param_value = parts[1].split()[0].strip()
                                        else:
                                            parts = content.split(f"{param}:")
                                            if len(parts) > 1:
                                                param_value = (
                                                    parts[1].split()[0].strip()
                                                )
                                            else:
                                                parts = content.split(f"{param} =")
                                                if len(parts) > 1:
                                                    param_value = (
                                                        parts[1].split()[0].strip()
                                                    )
                                                else:
                                                    parts = content.split(f"{param} :")
                                                    if len(parts) > 1:
                                                        param_value = (
                                                            parts[1].split()[0].strip()
                                                        )
                                    except:
                                        pass

                                if param_value:
                                    parameters[param] = param_value
                                    break

                            if not param_value:
                                missing_params.append(param)
                        else:
                            # If neither method exists, add to missing params
                            missing_params.append(param)
                    except Exception as e:
                        logger.error(
                            f"Error retrieving context value for {param}: {str(e)}"
                        )
                        missing_params.append(param)
                else:
                    missing_params.append(param)

        return missing_params

    def fetch_model_list(self, session_token, user_id, model_id):
        """
        Fetch the list of models from the API

        Args:
            session_token: The session token for authentication
            user_id: The user ID
            model_id: The model ID to fetch

        Returns:
            The API response as a dictionary or list
        """
        url = "https://staging-api.boostt.ai/api/cdm/model/get"

        payload = json.dumps(
            {"modelID": model_id, "userID": user_id, "session_token": session_token}
        )

        headers = {"Content-Type": "application/json"}
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            response.raise_for_status()  # Raises an error for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []

    def get_model_name_by_id(self, data, model_id):
        """
        Get the model name from the model ID

        Args:
            data: The list of models
            model_id: The model ID to find

        Returns:
            The model name if found, otherwise None
        """
        for item in data:
            if item.get("modelID") == model_id:
                return item.get("name")
        return None

    def replace_model_ids_with_names(self, text, session_token, model_id, user_id):
        """
        Replace model IDs in text with their corresponding model names.
        Uses regex to find potential model IDs and replaces them with model names.

        Args:
            text (str): The text containing model IDs
            session_token (str): Session token for API authentication
            user_id (str): User ID for API authentication

        Returns:
            str: Text with model IDs replaced by model names
        """
        try:
            if not text:
                return text

            response = self.fetch_model_list(
                session_token=session_token, user_id=user_id, model_id=model_id
            )

            try:
                model_name = response["name"]
            except:
                model_name = ""

            if model_name:
                text = text.replace(model_id, f"{model_name}")
        except:
            return text
        return text

    def analize_problem_statment(
        self, problem_statment, tool_name, conversation_id, user
    ):
        """
        Analyze the problem statement and return the appropriate action
        """
        prompt = langfuse.get_prompt(
            "problem statment analysis personalization", label="stage_v1"
        )
        config_problem_analysis_personalization = prompt.config
        PROMPT_PROBLEM_STATEMENT_ANALYSIS_PERSONALIZATION = prompt.compile()

        prompt = langfuse.get_prompt(
            "problem statment analysis journey", label="stage_v1"
        )
        config_problem_analysis_journy = prompt.config
        PROMPT_PROBLEM_STATEMENT_ANALYSIS_JOURNEY = prompt.compile()

        print("tool_name", tool_name)
        if tool_name == "journy_tool":
            parser = JsonOutputParser(pydantic_object=ProblemStatementAnalysisJourny)
            tamplate = PROMPT_PROBLEM_STATEMENT_ANALYSIS_JOURNEY
            tag = "problem_statment_analysis_journey"
            model = config_problem_analysis_journy["model"]
            temperature = config_problem_analysis_journy["temperature"]
        else:
            parser = JsonOutputParser(pydantic_object=ProblemStatementAnalysis)
            tamplate = PROMPT_PROBLEM_STATEMENT_ANALYSIS_PERSONALIZATION
            tag = "problem_statment_analysis_personalization"
            model = config_problem_analysis_personalization["model"]
            temperature = config_problem_analysis_personalization["temperature"]

        llm = ChatOpenAI(model=model, temperature=temperature, api_key=OPENAI_API_KEY)
        prompt = PromptTemplate(
            template=tamplate,
            input_variables=["problem_statment"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser

        config = config_llm_callback(
            run_name="problem_statment_analysis",
            tag=tag,
            conversation_id=conversation_id,
            user_id=user,
        )
        output = chain.invoke({"problem_statment": problem_statment}, config=config)
        return output

    async def process_request(self, request: UserRequest) -> ApiResponse:
        """
        Process a user request to determine the appropriate action and response

        Args:
            request: The user request

        Returns:
            ApiResponse object with the result
        """
        try:
            # Get or create conversation context
            conversation_id = request.conversation_id
            conversation_history = []
            problem_statement_flag = False
            if (
                request.problem_statement
                or (request.model_id!="" and request.model_id!=None)
                or (request.audience_id!="" and request.audience_id!=None)
                or  (request.target_id!="" and request.target_id!=None)
            ):
                problem_statement_flag = True

            print("&&&&&&&&&&&&&&&&&&&&& request.query", request.query)

            # Get conversation history if available
            if request.user_id and conversation_id and self.mongodb:
                conversation = await self.mongodb.get_conversation(
                    conversation_id=conversation_id
                )
                if conversation and conversation.get("user_id") == request.user_id:
                    # Get complete conversation history
                    # todo: check conversation history cominig as expacted or not
                    conversation_history = await self.mongodb.get_conversation_history(
                        conversation_id=conversation_id, limit=0
                    )
                    print(
                        "conversation_history, conversation_history",
                        conversation_history,
                    )
                    # Add user message to conversation
                    await self.mongodb.add_message_to_conversation(
                        conversation_id=conversation_id,
                        role="user",
                        content=request.query,
                        problem_statement=problem_statement_flag,
                    )
                else:
                    # Create new conversation if ID not found or doesn't belong to user
                    conversation = await self.mongodb.create_conversation(
                        user_id=request.user_id,
                        initial_message=request.query,
                        conversation_name=request.conversation_name,
                    )
                    conversation_id = conversation["conversation_id"]
            elif request.user_id and self.mongodb:
                # Create new conversation
                conversation = await self.mongodb.create_conversation(
                    user_id=request.user_id,
                    initial_message=request.query,
                    conversation_name=request.conversation_name,
                )
                conversation_id = conversation["conversation_id"]
            print("### request.query, conversation_history, conversation_id, request.user_id",request.query, conversation_history, conversation_id, request.user_id)
            tool_selection = await self.select_tool(
                request.query, conversation_history, conversation_id, request.user_id
            )
            parameters = tool_selection.parameters.copy()

            print("hereh hr hrewhf sdvn s vdddddddddddddddddddddddddddd")
            # Initialize parameters dictionary with values from request

            print(tool_selection, "^^^^^^^^^^^^^^^^^^^^^", parameters)

            if not request.conversation_id:
                conversation_id_ = conversation_id
            else:
                conversation_id_ = request.conversation_id

            exsting_parm = await self.mongodb.get_parameters_from_context(
                request.user_id, conversation_id_
            )

            # Add parameters from the request object
            if request.user_id:
                parameters["user_id"] = request.user_id
            if exsting_parm.get("user_id"):
                parameters["user_id"] = exsting_parm.get("user_id")

            if request.model_id:
                parameters["model_id"] = request.model_id
            if exsting_parm.get("model_id"):
                parameters["model_id"] = exsting_parm.get("model_id")

            if request.session_token:
                parameters["session_token"] = request.session_token
            if exsting_parm.get("session_token"):
                parameters["session_token"] = exsting_parm.get("session_token")

            if request.problem_statement:
                if request.problem_statement:
                    analize_problem_statment_out = self.analize_problem_statment(
                        request.problem_statement,
                        request.last_tool,
                        conversation_id,
                        request.user_id,
                    )
                    if analize_problem_statment_out["valid"]:
                        parameters["problem_statement"] = request.problem_statement
                        print(
                            "PROBLEM STATEMENT SET FROM REQUEST:",
                            request.problem_statement,
                        )
                    else:
                        print(
                            "INVALID PROBLEM STATEMENT:",
                            analize_problem_statment_out["missing_fields"],
                        )
                        print(
                            "SUGGESTED IMPROVEMENTS:",
                            analize_problem_statment_out["suggestions"],
                        )
                        return ApiResponse(
                            status="problem_statment_invalid",
                            message="Please provide a valid problem statement",
                            data={
                                "missing_fields": analize_problem_statment_out[
                                    "missing_fields"
                                ],
                                "suggestions": analize_problem_statment_out[
                                    "suggestions"
                                ],
                            },
                            required_inputs=["problem_statement"],
                            needed_input=["problem_statement"],
                            tool_used=tool_selection.tool_name,
                            conversation_id=conversation_id,
                            conversation_name=request.conversation_name,
                            conversation_stage="",
                            suggested_next_message="",
                        )

                print("PROBLEM STATEMENT SET FROM REQUEST:", request.problem_statement)
            else:
                print("NO PROBLEM STATEMENT IN REQUEST")
            if exsting_parm.get("problem_statement"):
                parameters["problem_statement"] = exsting_parm.get("problem_statement")
                print(
                    "PROBLEM STATEMENT SET FROM CONTEXT:",
                    exsting_parm.get("problem_statement"),
                )

            if request.audience_id:
                parameters["audience_id"] = request.audience_id
                print("AUDIENCE ID SET FROM REQUEST:", request.audience_id)
            else:
                print("NO AUDIENCE ID IN REQUEST")
            if exsting_parm.get("audience_id"):
                parameters["audience_id"] = exsting_parm.get("audience_id")
                print("AUDIENCE ID SET FROM CONTEXT:", exsting_parm.get("audience_id"))

            if request.persona_name:
                parameters["persona_name"] = request.persona_name
                print("PERSONA NAME SET FROM REQUEST:", request.persona_name)
            else:
                print("NO PERSONA NAME IN REQUEST")
            if exsting_parm.get("persona_name"):
                parameters["persona_name"] = exsting_parm.get("persona_name")
                print(
                    "PERSONA NAME SET FROM CONTEXT:", exsting_parm.get("persona_name")
                )

            if request.budget:
                parameters["budget"] = request.budget
                print("BUDGET SET FROM REQUEST:", request.budget)
            else:
                print("NO BUDGET IN REQUEST")
            if exsting_parm.get("budget"):
                parameters["budget"] = exsting_parm.get("budget")
                print("BUDGET SET FROM CONTEXT:", exsting_parm.get("budget"))

            if request.name:
                parameters["name"] = request.name
                print("NAME SET FROM REQUEST:", request.name)
            else:
                print("NO NAME IN REQUEST")
            if exsting_parm.get("name"):
                parameters["name"] = exsting_parm.get("name")
                print("NAME SET FROM CONTEXT:", exsting_parm.get("name"))

            if request.date:
                parameters["date"] = request.date
                print("DATE SET FROM REQUEST:", request.date)
            else:
                print("NO DATE IN REQUEST")
            if exsting_parm.get("date"):
                parameters["date"] = exsting_parm.get("date")
                print("DATE SET FROM CONTEXT:", exsting_parm.get("date"))

            if request.target_id:
                parameters["target_id"] = request.target_id
                print("TARGET ID SET FROM REQUEST:", request.target_id)
            else:
                print("NO TARGET ID IN REQUEST")
            if exsting_parm.get("target_id"):
                parameters["target_id"] = exsting_parm.get("target_id")
                print("TARGET ID SET FROM CONTEXT:", exsting_parm.get("target_id"))

            if request.has_audience_model:
                parameters["has_audience_model"] = request.has_audience_model
                print("has_audience_model FROM REQUEST:", request.has_audience_model)
            else:
                print("NO has_audience_model IN REQUEST")
            if exsting_parm.get("has_audience_model"):
                parameters["has_audience_model"] = exsting_parm.get(
                    "has_audience_model"
                )
                print(
                    "has_audience_model FROM CONTEXT:",
                    exsting_parm.get("has_audience_model"),
                )

            print("PARAMETERS:", parameters)
            parameters_copy = parameters.copy()
            if "action" in parameters_copy:
                parameters_copy.pop("action")
            await self.mongodb.save_parameters_to_context(
                request.user_id, conversation_id_, parameters_copy
            )
            print("PARAMETERS SAVED TO CONTEXT:", parameters_copy)
            print("PARAMETERS1:", parameters)

            # Handle document_content if provided in the request
            if hasattr(request, "document_content") and request.document_content:
                parameters["document_content"] = request.document_content
                print("DOCUMENT CONTENT ADDED FROM REQUEST")

            # Override tool selection if tool_input is provided directly in the request
            if hasattr(request, "tool_input") and request.tool_input:
                tool_selection.tool_name = request.tool_input.tool_name

                # Merge parameters from tool_input with existing parameters
                for key, value in request.tool_input.parameters.items():
                    parameters[key] = value

                print(f"TOOL INPUT OVERRIDE: {tool_selection.tool_name}")
                print(f"TOOL PARAMETERS FROM REQUEST: {parameters}")

            # Extract any field from the user's query - assume query might contain parameter values
            # This is especially important for the journey tool's conversational flow
            query = request.query.lower() if request.query else ""

            # Special handling for journy_tool
            if tool_selection.tool_name == "journy_tool":

                # Check if the user just provided a problem_statement parameter
                # This is likely in response to a request for the problem_statement
                if (
                    "problem_statement" in request.__dict__
                    and request.problem_statement
                ):
                    parameters["user_problem_statement"] = request.problem_statement
                    logger.info(
                        f"Using problem_statement from request for user_problem_statement: {request.problem_statement[:30]}..."
                    )

                # If action is missing, try to extract it from the query
                if "action" not in parameters:
                    get_context_msg = self.get_context_msg(
                        request.query, conversation_history
                    )
                    config = config_llm_callback(
                        run_name="journey_action_detection",
                        tag="journey_action_detection",
                        conversation_id=conversation_id,
                        user_id=request.user_id,
                    )
                    action = await self.journy_chain.ainvoke(
                        {"query": get_context_msg}, config=config
                    )
                    parameters["action"] = action

                    # If action still can't be determined, ask explicitly
                    if "action" not in parameters:
                        message = "I'd be happy to help with your marketing journey. What would you like to do? You can create a new journey, check status of an existing journey, get a journey report, or update a journey report with persona files."

                        # Process the message to replace model IDs with names if session token is available
                        if request.session_token and request.user_id and request.model_id:
                            message = self.replace_model_ids_with_names(
                                message,
                                request.session_token,
                                request.model_id,
                                request.user_id,
                            )

                        # Add assistant response to conversation
                        if conversation_id and request.user_id and self.mongodb:
                            await self.mongodb.add_message_to_conversation(
                                conversation_id=conversation_id,
                                role="assistant",
                                content=message,
                            )

                        return ApiResponse(
                            status="input_required",
                            message=message,
                            required_inputs=["action"],
                            needed_input=["action"],
                            tool_used=tool_selection.tool_name,
                            conversation_id=conversation_id,
                            conversation_name=request.conversation_name,
                            conversation_stage="selecting_journey_action",
                            suggested_next_message="I'd like to create a new journey.",
                        )

            # IMPORTANT: Always add conversation_id to parameters if available
            if conversation_id:
                parameters["conversation_id"] = conversation_id
                logger.info(
                    f"Added conversation_id={conversation_id} to tool parameters"
                )

            # For personalization tools, handle the multi-step flow
            personalization_tools = [
                "email_personalization",
                "directmail_personalization",
                "digitalad_personalization",
            ]
            if tool_selection.tool_name in personalization_tools:
                # First, check if we know the personalization data type (audience or persona)
                personalization_data_type = parameters.get("personalization_data_type")

                # If we don't have the personalization data type, try to detect it from the query
                if not personalization_data_type:
                    # Try to detect from current query
                    if request.query:
                        if any(
                            phrase in request.query.lower()
                            for phrase in [
                                "persona data",
                                "use persona",
                                "with persona",
                            ]
                        ):
                            personalization_data_type = "persona data"
                            parameters["personalization_data_type"] = "persona data"
                            logger.info(
                                f"Detected personalization_data_type='persona data' from query"
                            )
                        elif any(
                            phrase in request.query.lower()
                            for phrase in [
                                "audience data",
                                "use audience",
                                "with audience",
                            ]
                        ):
                            personalization_data_type = "audience data"
                            parameters["personalization_data_type"] = "audience data"
                            logger.info(
                                f"Detected personalization_data_type='audience data' from query"
                            )

                    # If not found in query, try to retrieve from conversation context
                    if not personalization_data_type:
                        data_type = None
                        if conversation_id and request.user_id and self.mongodb:
                            try:
                                if hasattr(self.mongodb, "get_context_value"):
                                    data_type = await self.mongodb.get_context_value(
                                        request.user_id,
                                        conversation_id,
                                        "personalization_data_type",
                                    )
                                elif hasattr(self.mongodb, "get_conversation"):
                                    conversation = await self.mongodb.get_conversation(
                                        conversation_id=conversation_id
                                    )
                                    if (
                                        conversation
                                        and "context" in conversation
                                        and "personalization_data_type"
                                        in conversation["context"]
                                    ):
                                        data_type = conversation["context"][
                                            "personalization_data_type"
                                        ]
                            except Exception as e:
                                logger.error(
                                    f"Error retrieving personalization_data_type from context: {str(e)}"
                                )

                        if data_type:
                            parameters["personalization_data_type"] = data_type
                            personalization_data_type = data_type
                            logger.info(
                                f"Retrieved personalization_data_type={data_type} from conversation context"
                            )

                # If we still don't have the personalization data type, ask user to mention it in their query
                if not personalization_data_type:
                    logger.info(
                        f"Personalization data type not specified for {tool_selection.tool_name}. Asking user to mention it in query."
                    )

                    tool_display_name = tool_selection.tool_name.replace("_", " ")
                    message = f"For {tool_display_name}, Do you want to go with audience data or persona data?"

                    # Process the message to replace model IDs with names if session token is available
                    if request.session_token and request.user_id:
                        message = self.replace_model_ids_with_names(
                            message,
                            request.session_token,
                            request.model_id,
                            request.user_id,
                        )

                    # Add assistant response to conversation
                    if conversation_id and request.user_id and self.mongodb:
                        await self.mongodb.add_message_to_conversation(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=message,
                        )

                    return ApiResponse(
                        status="success",
                        message=message,
                        tool_used=tool_selection.tool_name,
                        conversation_id=conversation_id,
                        conversation_name=request.conversation_name,
                        conversation_stage="selecting_personalization_type",
                        suggested_next_message="I want to create personalized content using persona data.",
                    )

                # Store the personalization_data_type in conversation context for future use
                if (
                    conversation_id
                    and request.user_id
                    and self.mongodb
                    and personalization_data_type
                ):
                    try:
                        if hasattr(self.mongodb, "update_conversation_context"):
                            await self.mongodb.update_conversation_context(
                                conversation_id=conversation_id,
                                context_updates={
                                    "personalization_data_type": personalization_data_type
                                },
                            )
                        elif hasattr(self.mongodb, "update_query_context"):
                            await self.mongodb.update_query_context(
                                user_id=request.user_id,
                                conversation_id=conversation_id,
                                new_params={
                                    "personalization_data_type": personalization_data_type
                                },
                                expected_keys=None,
                            )
                    except Exception as e:
                        logger.error(
                            f"Error storing personalization_data_type in context: {str(e)}"
                        )

                # Based on the personalization data type, set the required parameters
                if personalization_data_type.lower() == "audience data":
                    # For audience data, we need audience_id (model_id is optional)

                    # Remove model_id from required parameters if present - it's optional for audience data
                    if "model_id" in tool_selection.required_parameters:
                        tool_selection.required_parameters.remove("model_id")

                    # Check if we have audience_id
                    if not parameters.get("audience_id"):
                        # Try to retrieve audience_id from conversation context
                        audience_id = None
                        if conversation_id and request.user_id and self.mongodb:
                            try:
                                if hasattr(self.mongodb, "get_context_value"):
                                    audience_id = await self.mongodb.get_context_value(
                                        request.user_id, conversation_id, "audience_id"
                                    )
                                elif hasattr(self.mongodb, "get_conversation"):
                                    conversation = await self.mongodb.get_conversation(
                                        conversation_id=conversation_id
                                    )
                                    if (
                                        conversation
                                        and "context" in conversation
                                        and "audience_id" in conversation["context"]
                                    ):
                                        audience_id = conversation["context"][
                                            "audience_id"
                                        ]
                            except Exception as e:
                                logger.error(
                                    f"Error retrieving audience_id from context: {str(e)}"
                                )

                        if audience_id:
                            parameters["audience_id"] = audience_id
                            logger.info(
                                f"Retrieved audience_id={audience_id} from conversation context"
                            )
                        else:
                            logger.info(
                                "Audience ID not provided for persona creation. Requesting audience_id first."
                            )

                            message = "To create persona narratives using audience data, I'll need to know which audience ID you want to use. Please provide your audience ID."

                            # Add assistant response to conversation
                            if conversation_id and request.user_id and self.mongodb:
                                await self.mongodb.add_message_to_conversation(
                                    conversation_id=conversation_id,
                                    role="assistant",
                                    content=message,
                                )

                            return ApiResponse(
                                status="input_required",
                                message=message,
                                required_inputs=["audience_id"],
                                tool_used=tool_selection.tool_name,
                                conversation_id=conversation_id,
                                conversation_name=request.conversation_name,
                                conversation_stage="collecting_audience_id",
                                suggested_next_message="My audience ID is ABC123.",
                            )
                elif (
                    personalization_data_type.lower() == "model data"
                    or personalization_data_type.lower() == "persona data"
                ):
                    # For model data, we need model_id (audience_id is not needed)

                    # Remove audience_id from required parameters if present
                    if "audience_id" in tool_selection.required_parameters:
                        tool_selection.required_parameters.remove("audience_id")

                    # Check if we have model_id
                    if not parameters.get("model_id"):
                        # Try to retrieve model_id from conversation context
                        model_id = None
                        if conversation_id and request.user_id and self.mongodb:
                            try:
                                if hasattr(self.mongodb, "get_context_value"):
                                    model_id = await self.mongodb.get_context_value(
                                        request.user_id, conversation_id, "model_id"
                                    )
                                elif hasattr(self.mongodb, "get_conversation"):
                                    conversation = await self.mongodb.get_conversation(
                                        conversation_id=conversation_id
                                    )
                                    if (
                                        conversation
                                        and "context" in conversation
                                        and "model_id" in conversation["context"]
                                    ):
                                        model_id = conversation["context"]["model_id"]
                            except Exception as e:
                                logger.error(
                                    f"Error retrieving model_id from context: {str(e)}"
                                )

                        if model_id:
                            parameters["model_id"] = model_id
                            logger.info(
                                f"Retrieved model_id={model_id} from conversation context"
                            )
                        else:
                            logger.info(
                                "Model ID not provided for persona creation. Requesting model_id first."
                            )

                            message = "To create persona narratives using model data, I'll need to know which model ID you want to use. Please provide your model ID."

                            # Process the message to replace model IDs with names if session token is available
                            if request.session_token and request.user_id:
                                message = self.replace_model_ids_with_names(
                                    message,
                                    request.session_token,
                                    request.model_id,
                                    request.user_id,
                                )

                            # Add assistant response to conversation
                            if conversation_id and request.user_id and self.mongodb:
                                await self.mongodb.add_message_to_conversation(
                                    conversation_id=conversation_id,
                                    role="assistant",
                                    content=message,
                                )

                            return ApiResponse(
                                status="input_required",
                                message=message,
                                required_inputs=["model_id"],
                                tool_used=tool_selection.tool_name,
                                conversation_id=conversation_id,
                                conversation_name=request.conversation_name,
                                conversation_stage="collecting_model_id",
                                suggested_next_message="My model ID is xyz123.",
                            )

            # For standard tools (not personalization) or if all ecksch pass
            if tool_selection.tool_name:
                # Check for missing parameters (only essential ones after our personalization checks)
                missing_params = await self.check_missing_parameters(
                    tool_selection.tool_name,
                    parameters,
                    request.user_id,
                    conversation_id,
                )

                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
                print(parameters.get("personalization_data_type", ""))

                if not parameters.get("personalization_data_type", ""):
                    print("kkkkkkkkkkkkkkkkkkkkkkkkkkk here")
                    message = "Would you like to generate a persona narrative based on audience data or persona-specific data?"
                    if conversation_id and request.user_id and self.mongodb:
                        await self.mongodb.add_message_to_conversation(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=message,
                        )
                    return ApiResponse(
                        status="input_required",
                        message=message,
                        required_inputs=[],
                        tool_used=tool_selection.tool_name,
                        conversation_id=conversation_id,
                        conversation_name=request.conversation_name,
                        conversation_stage="",
                        suggested_next_message="",
                    )
                print("#########################missing_params", missing_params)
                # If there are missing parameters, return input_required response
                if missing_params:
                    message = f"Please provide the following information: {', '.join(missing_params)}"

                    # Add assistant response to conversation
                    if conversation_id and request.user_id and self.mongodb:
                        await self.mongodb.add_message_to_conversation(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=message,
                        )

                    return ApiResponse(
                        status="input_required",
                        message=message,
                        required_inputs=missing_params,
                        needed_input=missing_params,
                        tool_used=tool_selection.tool_name,
                        conversation_id=conversation_id,
                        conversation_name=request.conversation_name,
                        conversation_stage=tool_selection.conversation_stage,
                        suggested_next_message=tool_selection.suggested_next_message,
                    )

                # Execute the tool
                print("#########################", tool_selection.tool_name)
                tool_result = await self.execute_tool(
                    tool_selection.tool_name, parameters
                )

                # Special handling for persona_summary tool
                if (
                    tool_selection.tool_name == "persona_summary"
                    and tool_result.status == "success"
                ):
                    # Extract persona name from the result if available
                    persona_name = ""
                    model_id = parameters.get("model_id", "")

                    if (
                        tool_result.data
                        and isinstance(tool_result.data, dict)
                        and "persona_name" in tool_result.data
                    ):
                        persona_name = tool_result.data["persona_name"]
                    elif (
                        tool_result.data
                        and isinstance(tool_result.data, dict)
                        and "output_persona_json" in tool_result.data
                    ):
                        persona_json = tool_result.data["output_persona_json"]
                        if (
                            isinstance(persona_json, dict)
                            and "persona_name" in persona_json
                        ):
                            persona_name = persona_json["persona_name"]

                    # Store the persona_name in conversation context for future use
                    if (
                        conversation_id
                        and request.user_id
                        and self.mongodb
                        and persona_name
                    ):
                        # Update conversation context with persona info
                        if hasattr(self.mongodb, "update_conversation_context"):
                            await self.mongodb.update_conversation_context(
                                conversation_id=conversation_id,
                                context_updates={
                                    "persona_name": persona_name,
                                    "model_id": model_id,
                                },
                            )
                        elif hasattr(self.mongodb, "update_query_context"):
                            await self.mongodb.update_query_context(
                                user_id=request.user_id,
                                conversation_id=conversation_id,
                                new_params={
                                    "persona_name": persona_name,
                                    "model_id": model_id,
                                },
                                expected_keys=None,
                            )

                        # Ensure the persona summary is associated with this conversation_id
                        # This is important for maintaining the connection between the persona and the conversation
                        try:
                            # Check if we have the save_persona_summary method
                            if hasattr(self.mongodb, "save_persona_summary"):
                                # If the persona was already saved but without conversation_id, update it
                                # This handles the case where the tool already stored the persona but didn't include conversation_id
                                if tool_result.data and isinstance(
                                    tool_result.data, dict
                                ):
                                    output_data = tool_result.data.get(
                                        "output_persona_json", {}
                                    )
                                    if isinstance(output_data, dict):
                                        logger.info(
                                            f"Updating persona summary with conversation_id: {conversation_id}"
                                        )
                                        # Save/update the persona with the conversation_id
                                        await self.mongodb.save_persona_summary(
                                            user_id=request.user_id,
                                            model_id=model_id,
                                            output_persona_json=output_data,
                                            conversation_id=conversation_id,
                                            is_summary=True,
                                        )
                        except Exception as e:
                            logger.error(
                                f"Error updating persona summary with conversation_id: {str(e)}",
                                exc_info=True,
                            )

                    # Set up appropriate conversation stage for next step (personalization)
                    return ApiResponse(
                        status="success",
                        message=tool_result.message,
                        data=convert_objectid(tool_result.data),
                        tool_used="persona_summary",
                        conversation_id=conversation_id,
                        conversation_name=request.conversation_name,
                        conversation_stage="completed_persona_creation",
                        suggested_next_message="",
                    )

                # Special handling for journey_tool after creation
                if (
                    tool_selection.tool_name == "journy_tool"
                    and tool_result.status == "success"
                    and parameters.get("action") == "create_journey"
                ):
                    # Extract journey IDs from the result
                    journey_id = None
                    mongo_id = None

                    if tool_result.data and isinstance(tool_result.data, dict):
                        journey_id = tool_result.data.get("journey_id")
                        mongo_id = tool_result.data.get("journey_process_id")

                    # Store the journey IDs in conversation context for future use
                    if (
                        conversation_id
                        and request.user_id
                        and self.mongodb
                        and (journey_id or mongo_id)
                    ):
                        # Log the journey IDs we're storing
                        logger.info(
                            f"Storing journey IDs in context: journey_id={journey_id}, mongo_id={mongo_id}"
                        )

                        context_updates = {}
                        if journey_id:
                            context_updates["journey_id"] = journey_id
                        if mongo_id:
                            context_updates["mongo_id"] = mongo_id

                        # Update conversation context with journey info
                        if hasattr(self.mongodb, "update_conversation_context"):
                            await self.mongodb.update_conversation_context(
                                conversation_id=conversation_id,
                                context_updates=context_updates,
                            )
                        elif hasattr(self.mongodb, "update_query_context"):
                            await self.mongodb.update_query_context(
                                user_id=request.user_id,
                                conversation_id=conversation_id,
                                new_params=context_updates,
                                expected_keys=None,
                            )

                # If tool_result has required_inputs, transfer them to needed_input as well
                needed_input = None
                if (
                    hasattr(tool_result, "required_inputs")
                    and tool_result.required_inputs
                ):
                    needed_input = tool_result.required_inputs
                # If tool_result already has needed_input, use that
                elif hasattr(tool_result, "needed_input") and tool_result.needed_input:
                    needed_input = tool_result.needed_input

                # Regular response for other tools
                # Process the response message to replace model IDs with names if session token is available
                processed_message = tool_result.message
                if request.session_token and request.user_id:
                    processed_message = self.replace_model_ids_with_names(
                        tool_result.message,
                        request.session_token,
                        request.model_id,
                        request.user_id,
                    )

                # Add assistant response to conversation
                if conversation_id and request.user_id and self.mongodb:
                    await self.mongodb.add_message_to_conversation(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=processed_message,
                    )

                # Return result with needed_input field if present
                return ApiResponse(
                    status=tool_result.status,
                    message=processed_message,
                    data=convert_objectid(tool_result.data),
                    required_inputs=tool_result.required_inputs,
                    needed_input=needed_input,
                    tool_used=tool_selection.tool_name,
                    conversation_id=conversation_id,
                    conversation_name=request.conversation_name,
                    conversation_stage=tool_selection.conversation_stage,
                    suggested_next_message=tool_selection.suggested_next_message,
                )
            else:
                # No specific tool identified, use chat handler for general conversation
                context = {
                    "user_id": request.user_id,
                    "model_id": request.model_id,
                    "conversation_id": conversation_id,
                }

                general_response = await self.chat_handler.handle_message(
                    query=request.query,
                    conversation_history=conversation_history,
                    context=context,
                )

                # Process the response message to replace model IDs with names if session token is available
                processed_response = general_response
                if request.session_token and request.user_id:
                    processed_response = self.replace_model_ids_with_names(
                        general_response,
                        request.session_token,
                        request.model_id,
                        request.user_id,
                    )

                # Add assistant response to conversation
                if conversation_id and request.user_id and self.mongodb:
                    await self.mongodb.add_message_to_conversation(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=processed_response,
                    )

                return ApiResponse(
                    status="success",
                    message=processed_response,
                    tool_used=None,
                    conversation_id=conversation_id,
                    conversation_name=request.conversation_name,
                    conversation_stage=tool_selection.conversation_stage,
                    suggested_next_message=tool_selection.suggested_next_message,
                )

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return ApiResponse(
                status="error",
                message=f"An error occurred while processing your request: {str(e)}",
                conversation_id=request.conversation_id,
            )
