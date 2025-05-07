import logging
import os
import json
import re
from typing import Dict, Any, List, Optional, Union
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from app.utils.tool_registry import ToolRegistry
from app.utils.graph_processor import GraphProcessor
from app.models.model import UserRequest

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("openai_key")
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key not found in environment variables")

# Define the system prompt for the chat
CHAT_SYSTEM_PROMPT = """
You are a helpful AI assistant for a marketing personalization system. You help users with creating persona narratives and personalizing marketing content.

Your primary capabilities include:
1. Creating persona narratives based on user data
2. Generating personalized email content
3. Creating digital ad content
4. Developing direct mail content
5. Retrieving conversation history
6. Managing marketing journeys

IMPORTANT RULES:
- If a user wants to create email, digital ad, or direct mail personalization but NO persona narratives exist, FIRST guide them to create a persona narrative.
- When a user asks to create a persona narrative, ask for their model_id if not provided. A model_id is required for creating personas.
- Be friendly, helpful, and conversational in your tone.
- If you don't have enough information to call a specific tool, ask for the missing details in a natural way.
- When retrieving information from previous conversations, summarize it clearly.

When creating a marketing asset, the general workflow is:
1. First, ensure persona narratives exist
2. Then, use those narratives to personalize marketing content

Remember that your goal is to help the user complete their marketing tasks in a natural conversational way.
"""

class ChatProcessor:
    """
    Class for processing chat interactions
    """
    
    def __init__(self, mongodb):
        """
        Initialize the chat processor
        
        Args:
            mongodb: MongoDB instance for storing conversations
        """
        self.mongodb = mongodb
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            api_key=OPENAI_API_KEY
        )
        
    async def process_message(
        self, 
        message: str, 
        user_id: str = None, 
        model_id: str = None, 
        session_token: str = None,
        conversation_id: str = None,
        conversation_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message
        
        Args:
            message: The user's message
            user_id: User identifier
            model_id: Model identifier
            session_token: Session token
            conversation_id: Conversation identifier
            conversation_history: List of previous messages
            
        Returns:
            Dict with response
        """
        try:
            prompt_blyn = """
            You are Blyn, a friendly and helpful AI marketing expert specializing in personalized marketing campaigns for email, direct mail, and digital ads. Your tone is warm, conversational, and engaging - like a helpful colleague rather than a robotic assistant.

            When responding to users, follow these core principles:
            
            1. Be naturally conversational and guide the user through a coherent flow
            2. Avoid asking directly for parameters - instead, weave questions into a natural dialogue
            3. Provide context and explanation before asking for information
            4. Use casual, friendly language while remaining professional
            5. Acknowledge what the user has already shared
            
            CONVERSATION FLOW GUIDANCE:
            
            When a user first engages with you, welcome them warmly and briefly mention what you can help with.
            
            If a user mentions personalization:
            - Ask if they already have persona narratives or audience files
            - If they do, suggest they select from their available options (the frontend will show the list)
            - If they don't, gently guide them toward creating a persona narrative first
            
            When guiding users to create a persona narrative:
            - Explain why a persona narrative is important
            - Ask if they have a specific model ID they want to use
            - Explain that this will help create effective personas
            
            If you discover the user already has persona narratives when they said they don't:
            - Politely inform them that you found existing personas
            - Ask if they'd like to use the existing ones or create new ones
            - If they want new ones, confirm that will replace the existing ones
            
            If the user says they have personas but none are found:
            - Gently correct them and offer to create personas
            - Guide them through the persona creation process
            
            After creating persona narratives, ask which marketing channel they want to personalize for:
            - Email
            - Direct mail
            - Digital ad
            
            For each channel, guide them through providing the necessary information while maintaining a natural conversation flow. If certain information is missing, don't just ask for it directly - explain why it's needed and how it contributes to better personalization.

            FORMATTING STYLE:
            - Keep your responses concise and easy to read
            - Use bullet points for lists and options
            - Insert line breaks between paragraphs
            - Use emphasis sparingly for important points
            - Ask only one question at a time to avoid overwhelming the user
            
            Remember, your goal is to make the conversation feel natural and human-like, guiding users through the process without making it feel like they're simply filling out a form or answering a robotic questionnaire.
            """
            # Format the conversation history for the LLM
            messages = []

            # Add system message with formatting instructions
            # messages.append(("system", """You are a helpful assistant for a marketing personalization system. 
            # You help users with creating persona narratives and personalizing marketing content.

            # FORMATTING GUIDELINES:
            # - Always use bullet points (- ) for lists of items
            # - Use numbered lists (1., 2., etc.) for sequential steps
            # - Break your response into clear sections with line breaks between paragraphs
            # - Use bold for important concepts or terms
            # - Use headings for section titles
            # - Format code or technical terms appropriately
            # - Use indentation for highlighting important information

            # Your primary capabilities include:
            # 1. Creating persona narratives based on user data
            # 2. Generating personalized email content
            # 3. Creating digital ad content
            # 4. Developing direct mail content
            # 5. Retrieving conversation history
            # 6. Managing marketing journeys

            # IMPORTANT RULES:
            # - If a user wants to create email, digital ad, or direct mail personalization but NO persona narratives exist, FIRST guide them to create a persona narrative.
            # - When a user asks to create a persona narrative, ask for their model_id if not provided. A model_id is required for creating personas.
            # - Be friendly, helpful, and conversational in your tone.
            # - If you don't have enough information to call a specific tool, ask for the missing details in a natural way.
            # - When retrieving information from previous conversations, summarize it clearly.

            # Make your responses visually organized and easy to read. Use formatting to highlight key points.
            # """))

            messages.append(("system", prompt_blyn))
            
            # First check if this is a new session
            is_new_session = False
            if not conversation_history or len(conversation_history) == 0:
                is_new_session = True
                
            # If it's a new session and we have persona summaries, add them to context
            # if is_new_session and model_id:
            #     try:
            #         # Get persona summaries if available
            #         persona_summaries = await self._get_persona_summaries(user_id, model_id)
            #         if persona_summaries:
            #             persona_summary_str = ""
            #             for idx, summary in enumerate(persona_summaries):
            #                 persona_name = summary.get("persona_name", f"persona_{idx+1}")
            #                 data = summary.get("data", "")
            #                 persona_summary_str += f"{persona_name}: {data}\n"
                        
            #             if persona_summary_str:
            #                 # Add persona context as a system message
            #                 messages.append(("system", f"Here are the persona narratives for this conversation:\n{persona_summary_str}"))
            #     except Exception as e:
            #         logger.error(f"Error getting persona summaries: {str(e)}")
            
            # Process conversation history - handle large histories efficiently
            if conversation_history:
                logger.info(f"Processing {len(conversation_history)} messages from conversation history")
                
                # # If history is very large, use a summary approach
                # if len(conversation_history) > 20:
                #     logger.info("Large conversation history detected, using summary approach")
                    
                #     # Include the first system messages (context setting)
                #     early_messages = []
                #     for i, msg in enumerate(conversation_history[:5]):
                #         if msg.get("role") in ["system", "assistant", "user"]:
                #             role = "ai" if msg.get("role") == "assistant" else msg.get("role")
                #             role = "human" if role == "user" else role
                #             early_messages.append((role, msg.get("content", "")))
                    
                #     # Create a summary of the middle part
                #     if len(conversation_history) > 10:
                #         middle_index = len(conversation_history) // 2
                #         middle_range = conversation_history[5:middle_index-3]
                        
                #         if middle_range:
                #             # Add a summary message of the middle context
                #             summary_points = []
                #             for msg in middle_range:
                #                 if msg.get("role") == "user":
                #                     summary_points.append(f"User asked: {msg.get('content', '')[:100]}...")
                #                 elif msg.get("role") == "assistant":
                #                     summary_points.append(f"Assistant responded about: {msg.get('content', '')[:100]}...")
                            
                #             if summary_points:
                #                 messages.append(("system", f"Summary of previous conversation:\n" + "\n".join(summary_points)))
                    
                #     # Always include the most recent context (last 10 messages)
                #     recent_messages = conversation_history[-10:]
                #     for msg in recent_messages:
                #         role = msg.get("role", "")
                #         content = msg.get("content", "")
                #         if role == "user":
                #             messages.append(("human", content))
                #         elif role == "assistant":
                #             messages.append(("ai", content))
                
                # For smaller histories, include all messages
                # else:
                for msg in conversation_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        messages.append(("human", content))
                    elif role == "assistant":
                        messages.append(("ai", content))
            
            # Add the current message
            messages.append(("human", message))
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages(messages)
            
            # Create the chain
            chain = prompt | self.llm
            
            # Run the chain
            response = await chain.ainvoke({})
            
            # Extract the content
            response_content = response.content
            
            return {
                "status": "success",
                "message": response_content,
                "data": {
                    "user_id": user_id,
                    "conversation_id": conversation_id
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "status": "error",
                "message": f"An error occurred while processing your message: {str(e)}",
                "data": None
            }
    
    async def _get_persona_summaries(self, user_id: str, model_id: str) -> List[Dict[str, Any]]:
        """
        Get persona summaries for the user and model
        
        Args:
            user_id: User identifier
            model_id: Model identifier
            
        Returns:
            List of persona summaries
        """
        try:
            # Query the database for persona summaries without any limit
            logger.info(f"Retrieving all persona summaries for user_id={user_id}, model_id={model_id}")
            summaries = await self.mongodb.find_documents(
                collection="persona_summaries",
                query={
                    "user_id": user_id,
                    "model_id": model_id,
                    "is_summary": True
                },
                limit=0  # Get all summaries
            )
            
            # Also try the persona_summaries collection or get_persona_summaries method
            # This is to handle different DB schema possibilities
            if not summaries:
                try:
                    logger.info("Trying to get persona summaries from get_persona_summaries method")
                    summaries = await self.mongodb.get_persona_summaries(
                        user_id=user_id,
                        model_id=model_id
                    )
                except Exception as e:
                    logger.warning(f"Could not get persona summaries from get_persona_summaries: {str(e)}")
                    
            logger.info(f"Retrieved {len(summaries)} persona summaries")
            return summaries
        except Exception as e:
            logger.error(f"Error getting persona summaries: {str(e)}")
            return []
    
    async def _analyze_intent(self, 
                             query: str,
                             chat_history: List[Union[HumanMessage, AIMessage, SystemMessage]],
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the user's intent to determine if a tool call is needed
        
        Args:
            query: The user's message
            chat_history: The conversation history
            context: Additional context
            
        Returns:
            A dictionary with the intent analysis
        """
        # Create the prompt for intent analysis
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items() if v])
        
        # Add the current query to chat history
        full_history = chat_history + [HumanMessage(content=query)]
        
        # System message with instructions for intent analysis
        intent_system_prompt = f"""
        You are an AI assistant for a marketing personalization system.
        
        Context:
        {context_str}
        
        Analyze the user's message and determine:
        1. If a tool call is required
        2. Which tool should be called
        3. What parameters are needed for the tool
        4. If model_id is needed for the operation
        5. If persona narratives are needed for a personalization task
        
        Available tools:
        - persona_summary: Creates persona summaries (requires model_id)
        - email_personalization: Creates email content (requires persona narratives)
        - directmail_personalization: Creates direct mail content (requires persona narratives)
        - digitalad_personalization: Creates digital ad content (requires persona narratives)
        - persona_narrative: Creates detailed persona narratives (requires model_id)
        - history_conversation: Retrieves conversation history
        
        Return your analysis in JSON format:
        {{
            "tool_call_required": true|false,
            "tool_name": "tool_name" if tool_call_required,
            "parameters": {{}} if tool_call_required,
            "model_id_needed": true|false,
            "persona_narrative_needed": true|false,
            "response": "conversational response if no tool call is needed"
        }}
        """
        
        # Create a separate conversation for intent analysis
        intent_messages = [
            SystemMessage(content=intent_system_prompt),
            HumanMessage(content=f"User message: {query}")
        ]
        
        # Use LLM to analyze intent
        intent_result = await self.llm.ainvoke(intent_messages)
        intent_content = intent_result.content
        
        # Extract the JSON from the response
        try:
            # The LLM might wrap the JSON in markdown code blocks, so we need to extract it
            json_match = re.search(r'```json\s*(.*?)\s*```', intent_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'(\{.*\})', intent_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = intent_content
                    
            # Parse the JSON
            intent_analysis = json.loads(json_str)
            
            # Ensure we have all required fields
            if "tool_call_required" not in intent_analysis:
                intent_analysis["tool_call_required"] = False
            if "response" not in intent_analysis:
                intent_analysis["response"] = "I'm not sure how to respond to that."
                
            return intent_analysis
                
        except Exception as e:
            logger.error(f"Error parsing intent analysis: {str(e)}", exc_info=True)
            return {
                "tool_call_required": False,
                "response": "I'm having trouble understanding your request. Could you please rephrase it?"
            } 