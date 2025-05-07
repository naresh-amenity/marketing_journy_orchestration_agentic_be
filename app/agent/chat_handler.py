import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.models.model import ApiResponse

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API keys
OPENAI_API_KEY = os.getenv("openai_key")

# System prompt for general conversation
GENERAL_CHAT_PROMPT = """
You are Blyn, an AI assistant created by Boostt AI Co-Pilot.
You help users with marketing personalization through the following core services:

Persona Narrative – Crafting detailed personas for target audiences.

Content Personalization – Tailoring content strategies for various audience segments.

Journey Creation – Designing personalized user journeys across marketing funnels.

Core Guidelines:

Greeting Protocol:
If the user greets you (e.g., “Hi”, “Hello”, “Hey Blyn”), reply warmly by introducing your services and asking what they'd like to do today.
Example:

Hi! I’m Blyn, your Boostt AI Co-Pilot. I can help with persona narratives, content personalization, and journey creation. What would you like to work on today?

Professional and Helpful Tone:
Respond in a helpful, concise, and professional manner.

Clarification:
If the user's request is unclear, politely ask clarifying questions to better assist.

Tool Guidance:
If the user needs to use a specific tool or method, guide them toward the appropriate steps or information.

Conversation History:
Use context from previous conversations when provided. If a user mentions a prior request or task, confirm whether they’d like to continue with it before proceeding.
"""


class ChatHandler:
    """
    Handler for general conversations that don't require specific tools
    """

    def __init__(self):
        """Initialize the chat handler with LLM and prompt template"""
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4o", api_key=OPENAI_API_KEY)

        # Create prompt template for general conversation
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [("system", GENERAL_CHAT_PROMPT), ("human", "{query}")]
        )

    async def handle_message(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Process a general conversation message

        Args:
            query: The user query
            conversation_history: Optional conversation history
            context: Optional context variables

        Returns:
            The assistant response
        """
        try:
            # Format conversation history for the LLM
            messages = [SystemMessage(content=GENERAL_CHAT_PROMPT)]

            # Add conversation history if available
            if conversation_history and len(conversation_history) > 0:
                for message in conversation_history:
                    role = message.get("role", "")
                    content = message.get("content", "")

                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))

            # Add current query
            messages.append(HumanMessage(content=query))

            # Generate response
            response = await self.llm.ainvoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}", exc_info=True)
            return "I'm sorry, but I encountered an error processing your message. Please try again."
