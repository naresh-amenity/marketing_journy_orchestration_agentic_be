import glob
import logging
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from app.agent.main_agent import MainAgent
from app.models.model import (
    ApiResponse,
    ChatMessage,
    ConversationState,
    ToolInput,
    UserRequest,
)
from app.utils.db import MongoDB

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process")
async def process_request(request: UserRequest, req: Request):
    """
    Process a user request and return the appropriate response

    This endpoint handles both general chat and tool-specific requests using a single MainAgent:
    - Analyzes if the request needs a specific tool
    - If a tool is needed, executes the tool and returns its result
    - Otherwise, processes as a general chat response
    - Maintains conversation history throughout
    - Handles conversation ending when is_conversation_end flag is set by frontend

    Args:
        request: The user request
        req: The FastAPI request object

    Returns:
        ApiResponse containing the processing result
    """
    try:
        # Get MongoDB instance
        mongodb = req.app.mongodb

        # Log if this is a conversation ending request
        if request.is_conversation_end:
            logger.info(
                f"Frontend requested to end conversation: {request.conversation_id} with name: {request.conversation_name or 'General Conversation'}"
            )

            # Update conversation as ended
            if request.user_id and request.conversation_id:
                await mongodb.update_conversation(
                    conversation_id=request.conversation_id,
                    user_id=request.user_id,
                    updates={"is_ended": True},
                    conversation_name=request.conversation_name,
                )

                return ApiResponse(
                    status="success",
                    message="Conversation ended successfully",
                    conversation_id=request.conversation_id,
                    is_conversation_end=True,
                )

        # Create MainAgent instance with MongoDB
        main_agent = MainAgent(mongodb)

        # Process the request using the MainAgent
        response = await main_agent.process_request(request)

        return response

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return ApiResponse(
            status="error",
            message=f"An error occurred: {str(e)}",
            conversation_id=request.conversation_id,
        )


@router.get("/conversations/{user_id}", response_model=list)
async def get_user_conversations(user_id: str, req: Request, limit: int = 10):
    """
    Get a list of conversations for a user

    Args:
        user_id: The user ID
        req: The FastAPI request object
        limit: The maximum number of conversations to return

    Returns:
        A list of conversation objects
    """
    try:
        # Get MongoDB instance
        mongodb = req.app.mongodb

        # Get conversations
        conversations = await mongodb.get_user_conversations(
            user_id=user_id, limit=limit
        )

        return conversations
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.get("/conversation/{conversation_id}", response_model=dict)
async def get_conversation(
    conversation_id: str, user_id: str, req: Request, include_messages: bool = True
):
    """
    Get a conversation by ID

    Args:
        conversation_id: The ID of the conversation
        user_id: The ID of the user (for security check)
        include_messages: Whether to include the full message history

    Returns:
        The conversation record
    """
    try:
        # Access MongoDB instance
        mongodb = req.app.mongodb

        # Get the conversation
        conversation = await mongodb.get_conversation(conversation_id=conversation_id)
        print("conversation111111111", conversation)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        print("conversation22222222")
        # Security check - ensure the conversation belongs to the requesting user
        if conversation.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to access this conversation",
            )
        print("conversation33333333")
        # If include_messages is true, get the complete message history
        if include_messages:
            # Get all messages for this conversation
            messages = await mongodb.get_conversation_history(
                conversation_id=conversation_id, limit=0  # Get all messages
            )
            conversation["full_message_history"] = messages

        if conversation and "_id" in conversation:
            conversation["_id"] = str(conversation["_id"])
        return conversation

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error getting conversation: {str(e)}"
        )


@router.get("/history/{user_id}", response_model=list)
async def get_user_history(
    user_id: str, req: Request, limit: int = 0, conversation_id: str = None
):
    """
    Get the complete history of tool executions for a user

    Args:
        user_id: The ID of the user
        limit: Maximum number of records to return (0 means no limit)
        conversation_id: Optional conversation ID to filter by

    Returns:
        A list of tool execution records
    """
    try:
        # Access MongoDB instance
        mongodb = req.app.mongodb

        # Get the user's complete history
        history = await mongodb.get_tool_execution_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=limit,  # 0 will get all history
        )

        logger.info(f"Retrieved {len(history)} history records for user {user_id}")

        return history

    except Exception as e:
        logger.error(f"Error getting user history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error getting user history: {str(e)}"
        )


@router.patch("/conversation/{conversation_id}/name")
async def update_conversation_name(
    conversation_id: str,
    user_id: str,
    req: Request,
    conversation_name: str,
    mark_as_ended: bool = False,
    category: Optional[str] = None,
):
    """
    Update the name of a conversation

    Args:
        conversation_id: The ID of the conversation
        user_id: The ID of the user (for security check)
        conversation_name: The new name for the conversation
        mark_as_ended: Whether to mark the conversation as ended
        category: Optional category for the conversation to enable sorting

    Returns:
        Success message
    """
    try:
        # Access MongoDB instance
        mongodb = req.app.mongodb

        # Get the conversation for security check
        conversation = await mongodb.get_conversation(conversation_id=conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Security check - ensure the conversation belongs to the requesting user
        if conversation.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to update this conversation",
            )

        # If marking as ended
        if mark_as_ended:
            success = await mongodb.mark_conversation_ended(
                conversation_id=conversation_id,
                conversation_name=conversation_name,
                category=category,
            )
            if success:
                return {
                    "status": "success",
                    "message": f"Conversation marked as ended with name: {conversation_name}",
                }
            else:
                raise HTTPException(
                    status_code=500, detail="Failed to update conversation"
                )
        else:
            # Just update the name
            success = await mongodb.update_conversation_name(
                conversation_id=conversation_id,
                conversation_name=conversation_name,
                category=category,
            )
            if success:
                message = f"Conversation name updated to: {conversation_name}"
                if category:
                    message += f" with category: {category}"
                return {"status": "success", "message": message}
            else:
                raise HTTPException(
                    status_code=500, detail="Failed to update conversation name"
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation name: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error updating conversation name: {str(e)}"
        )


@router.post("/conversation/{conversation_id}/end")
async def end_conversation(
    conversation_id: str,
    user_id: str,
    req: Request,
    conversation_name: Optional[str] = None,
    category: Optional[str] = None,
):
    """
    Explicitly end a conversation

    Args:
        conversation_id: The ID of the conversation
        user_id: The ID of the user (for security check)
        conversation_name: Optional name for the conversation
        category: Optional category for the conversation to enable sorting

    Returns:
        Success message
    """
    try:
        # Access MongoDB instance
        mongodb = req.app.mongodb

        # Get the conversation for security check
        conversation = await mongodb.get_conversation(conversation_id=conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Security check - ensure the conversation belongs to the requesting user
        if conversation.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to update this conversation",
            )

        # Mark the conversation as ended
        success = await mongodb.mark_conversation_ended(
            conversation_id=conversation_id,
            conversation_name=conversation_name,
            category=category,
        )

        if success:
            message = f"Conversation marked as ended with name: {conversation_name or 'General Conversation'}"
            if category:
                message += f" and category: {category}"
            return {"status": "success", "message": message}
        else:
            raise HTTPException(status_code=500, detail="Failed to end conversation")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error ending conversation: {str(e)}"
        )


@router.get("/persona_name/{user_id}", response_model=list)
async def get_user_history(user_id: str, req: Request, model_id: str = ""):
    """
    Get a list of conversations for a user

    Args:
        user_id: The user ID
        req: The FastAPI request object
        model_id: The model ID to filter by

    Returns:
        A list of conversation objects
    """
    try:
        # Get MongoDB instance
        mongodb = req.app.mongodb

        # Get conversations
        conversations = await mongodb.get_persona_summaries(
            user_id=user_id,
            model_id=model_id,
            conversation_id="",
            conversation_status=True,
        )
        persona_names = [item["persona_name"] for item in conversations]
        persona_names = list(set(persona_names))
        return persona_names
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str, user_id: str, req: Request):
    """
    Delete a conversation by ID

    Args:
        conversation_id: The ID of the conversation to delete
        user_id: The ID of the user (for security check)
        req: The FastAPI request object

    Returns:
        Success message if the conversation was deleted
    """
    try:
        # Access MongoDB instance
        mongodb = req.app.mongodb

        # Get the conversation for security check
        conversation = await mongodb.get_conversation(conversation_id=conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Security check - ensure the conversation belongs to the requesting user
        if conversation.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to delete this conversation",
            )

        # Delete the conversation
        result = await mongodb.delete_conversation(conversation_id=conversation_id)

        if result:
            return {"status": "success", "message": "Conversation deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete conversation")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error deleting conversation: {str(e)}"
        )


async def get_generalized_personalization(
    user_id, db, audience_id
) -> List[Dict[str, Any]]:
    """
    Retrieve generalized personalization data from the database

    Args:
        user_id: The ID of the user
        audience_id: Optional audience ID filter

    Returns:
        List of generalized personalization documents
    """
    try:
        logger.info(
            f"Retrieving generalized personalization data for user_id={user_id}, audience_id={audience_id}"
        )

        # Get the data from the database
        return await db.get_generalized_personalization(
            user_id=user_id, audience_id=audience_id
        )
    except Exception as e:
        logger.error(f"Error in get_generalized_personalization: {str(e)}")
        return []


@router.get("/audiance_name/{user_id}", response_model=dict)
async def get_user_history(user_id: str, req: Request, audience_id: str = ""):
    """
    Get a list of conversations for a user

    Args:
        user_id: The user ID
        req: The FastAPI request object
        model_id: The model ID to filter by

    Returns:
        A list of conversation objects
    """
    try:
        mongodb = req.app.mongodb
        generalized_personalization_data = await get_generalized_personalization(
            user_id, mongodb, audience_id
        )

        print("generalized_personalization_data", generalized_personalization_data)
        if generalized_personalization_data:
            return {
                "persona_available": True,
                "data": generalized_personalization_data[0]["filter_criteria"],
            }
        else:
            return {"persona_available": False, "data": []}
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
