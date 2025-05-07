import os
import logging
import uuid
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from typing import Optional, Dict, Any, List
import json
from bson import ObjectId

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

def convert_objectid(obj: Any) -> Any:
    """
    Convert MongoDB ObjectId to string in a nested structure
    
    Args:
        obj: The object to convert
        
    Returns:
        The converted object
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_objectid(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid(item) for item in obj]
    else:
        return obj

class MongoDB:
    """
    MongoDB connection manager using Motor for async operations
    """
    _instance = None
    _client = None
    _db = None
    
    def __new__(cls):
        """
        Singleton pattern to ensure only one database connection
        """
        if cls._instance is None:
            cls._instance = super(MongoDB, cls).__new__(cls)
            cls._instance._initialize_db()
        return cls._instance
    
    def _initialize_db(self):
        """
        Initialize the database connection
        """
        try:
            # Get MongoDB connection string from environment variables
            mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
            db_name = os.getenv("MONGO_DB_NAME", "persona_tool_db")
            
            # Connect to MongoDB
            self._client = AsyncIOMotorClient(mongo_uri)
            self._db = self._client[db_name]
            
            # Collection initialization is now handled by the async ensure_collections method
            # called in app startup
            
            logger.info(f"Connected to MongoDB database: {db_name}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}", exc_info=True)
            raise
    
    @property
    def db(self):
        """
        Get the database instance
        
        Returns:
            The AsyncIOMotorDatabase instance
        """
        return self._db
    
    @property
    def client(self):
        """
        Get the client instance
        
        Returns:
            The AsyncIOMotorClient instance
        """
        return self._client
    
    async def close(self):
        """
        Close the database connection
        """
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")
    
    # Conversation Management Operations
    
    async def create_conversation(self, user_id: str, initial_message: str, conversation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new conversation
        
        Args:
            user_id: The ID of the user
            initial_message: The initial message from the user
            conversation_name: Optional name for the conversation
            
        Returns:
            The conversation document
        """
        collection = self._db.conversations
        conversation_id = str(uuid.uuid4())
        
        # Create the document
        now = datetime.utcnow().isoformat()
        doc = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
            "messages": [
                {
                    "role": "user",
                    "content": initial_message,
                    "timestamp": now
                }
            ],
            "context": {},
            "active_persona_narratives": [],
            "model_id": None,
            "conversation_name": conversation_name,
            "is_ended": False
        }
        
        # Insert the document
        await collection.insert_one(doc)
        
        return doc
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID
        
        Args:
            conversation_id: The ID of the conversation
            
        Returns:
            The conversation document or None if not found
        """
        collection = self._db.conversations
        
        try:
            # Find the conversation by ID
            conversation = await collection.find_one({"conversation_id": conversation_id})
            # cursor = await collection.find({"conversation_id": conversation_id})
            # conversation = await cursor.to_list(length=None)
            print("^^^^^^^^^^^^^^^^^^^^^conversation", conversation)
            
            # Convert ObjectId to string
            return convert_objectid(conversation)
        except Exception as e:
            logger.error(f"Error getting conversation: {str(e)}")
            return None
    
    async def get_user_conversations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get a list of conversations for a user
        
        Args:
            user_id: The ID of the user
            limit: The maximum number of conversations to retrieve
            
        Returns:
            A list of conversation documents
        """
        collection = self._db.conversations
        
        try:
            # Get the conversations for the user
            cursor = collection.find({"user_id": user_id}).sort("last_updated", -1)
            
            # Apply limit if specified
            if limit > 0:
                cursor = cursor.limit(limit)
            
            # Convert to list
            conversations = await cursor.to_list(length=None)
            
            # Convert ObjectId to string
            return convert_objectid(conversations)
        except Exception as e:
            logger.error(f"Error getting user conversations: {str(e)}")
            return []
    
    async def update_query_context(
            self,
            user_id: str,
            conversation_id: str,
            new_params: Dict[str, Any],
            expected_keys: Optional[List[str]] = None
        ) -> Optional[Dict[str, Any]]:
            """
            Update or overwrite the context (extracted parameters) for a given user and conversation.
            If all expected keys are provided in new_params, it overwrites the context.
            Otherwise, merges new_params into existing context.
            """
            collection = self._db.conversations
            conversation = await collection.find_one({"user_id": user_id, "conversation_id": conversation_id})
            if not conversation:
                logger.error(f"Conversation not found for user_id={user_id} and conversation_id={conversation_id}")
                return None

            current_context = conversation.get("context", {})
            if expected_keys and set(new_params.keys()) == set(expected_keys):
                updated_context = new_params  # overwrite if complete
            else:
                updated_context = {**current_context, **new_params}  # merge if partial

            await collection.update_one(
                {"user_id": user_id, "conversation_id": conversation_id},
                {"$set": {"context": updated_context, "updated_at": datetime.utcnow().isoformat()}}
            )
            return updated_context
    
    async def get_missing_context_fields(
        self,
        user_id: str,
        conversation_id: str,
        expected_keys: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Return a dict of missing expected keys from the stored context for a given user and conversation.
        """
        collection = self._db.conversations
        conversation = await collection.find_one({"user_id": user_id, "conversation_id": conversation_id})
        if not conversation:
            logger.error(f"Conversation not found for user_id={user_id} and conversation_id={conversation_id}")
            return None

        current_context_ = conversation.get("context", {})
        current_context = {k: v for k, v in current_context_.items() if v != ''}
        print("current_context current_context", current_context)
        print("expected_keys expected_keys", expected_keys)
        missing = {key: None for key in expected_keys if key not in current_context}
        if missing is None:
            missing = []
        if current_context is None:
            current_context = {}
        return {"missing": missing, "current_context": current_context}

    async def add_message_to_conversation(self, 
                                        conversation_id: str, 
                                        role: str, 
                                        content: str,
                                        problem_statement=False) -> bool:
        """
        Add a message to a conversation
        
        Args:
            conversation_id: The ID of the conversation
            role: The role of the message sender (user or assistant)
            content: The message content
            
        Returns:
            True if successful, False otherwise
        """
        collection = self._db.conversations
        
        # Create the message
        now = datetime.utcnow().isoformat()
        message = {
            "role": role,
            "content": content,
            "problem_statement": problem_statement,
            "timestamp": now
        }
        
        # Update the document
        result = await collection.update_one(
            {"conversation_id": conversation_id},
            {
                "$push": {"messages": message},
                "$set": {"updated_at": now}
            }
        )
        
        return result.modified_count > 0
    
    async def update_conversation_context(self, 
                                        conversation_id: str, 
                                        context_updates: Dict[str, Any]) -> bool:
        """
        Update the context of a conversation
        
        Args:
            conversation_id: The ID of the conversation
            context_updates: The context updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        collection = self._db.conversations
        
        # Update the document
        now = datetime.utcnow().isoformat()
        result = await collection.update_one(
            {"conversation_id": conversation_id},
            {
                "$set": {
                    **{f"context.{k}": v for k, v in context_updates.items()},
                    "updated_at": now
                }
            }
        )
        
        return result.modified_count > 0
    
    async def set_conversation_model_id(self, 
                                      conversation_id: str, 
                                      model_id: str) -> bool:
        """
        Set the model ID for a conversation
        
        Args:
            conversation_id: The ID of the conversation
            model_id: The model ID
            
        Returns:
            True if successful, False otherwise
        """
        collection = self._db.conversations
        
        # Update the document
        now = datetime.utcnow().isoformat()
        result = await collection.update_one(
            {"conversation_id": conversation_id},
            {
                "$set": {
                    "model_id": model_id,
                    "updated_at": now
                }
            }
        )
        
        return result.modified_count > 0
    
    async def add_persona_narrative(self, 
                                  conversation_id: str, 
                                  persona_name: str) -> bool:
        """
        Add a persona narrative to a conversation
        
        Args:
            conversation_id: The ID of the conversation
            persona_name: The name of the persona
            
        Returns:
            True if successful, False otherwise
        """
        collection = self._db.conversations
        
        # Update the document
        now = datetime.utcnow().isoformat()
        result = await collection.update_one(
            {"conversation_id": conversation_id},
            {
                "$addToSet": {"active_persona_narratives": persona_name},
                "$set": {"updated_at": now}
            }
        )
        
        return result.modified_count > 0
    
    async def get_latest_conversation(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest conversation for a user
        
        Args:
            user_id: The ID of the user
            
        Returns:
            The conversation document, or None if not found
        """
        collection = self._db.conversations
        
        # Execute the query
        result = await collection.find_one(
            {"user_id": user_id},
            sort=[("updated_at", -1)]
        )
        
        return result
    
    async def update_conversation_name(self, conversation_id: str, conversation_name: str, category: Optional[str] = None) -> bool:
        """
        Update the name of a conversation
        
        Args:
            conversation_id: The ID of the conversation
            conversation_name: The new name for the conversation
            category: Optional category for the conversation to enable sorting
            
        Returns:
            True if successful, False otherwise
        """
        collection = self._db.conversations
        
        # Update the document
        now = datetime.utcnow().isoformat()
        update_data = {
            "conversation_name": conversation_name,
            "updated_at": now
        }
        
        # Add category if provided
        if category:
            update_data["category"] = category
        
        # Update the document
        result = await collection.update_one(
            {"conversation_id": conversation_id},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
    
    async def mark_conversation_ended(self, conversation_id: str, conversation_name: Optional[str] = None, category: Optional[str] = None) -> bool:
        """
        Mark a conversation as ended and optionally set its name
        
        Args:
            conversation_id: The ID of the conversation
            conversation_name: Optional final name for the conversation
            category: Optional category for the conversation to enable sorting
            
        Returns:
            True if successful, False otherwise
        """
        collection = self._db.conversations
        
        # Prepare update data
        now = datetime.utcnow().isoformat()
        update_data = {
            "is_ended": True,
            "updated_at": now
        }
        
        # Add conversation name if provided
        if conversation_name:
            update_data["conversation_name"] = conversation_name
        else:
            # If no name provided, set to "General Conversation" only if no name exists
            conversation = await self.get_conversation(conversation_id)
            if conversation and not conversation.get("conversation_name"):
                update_data["conversation_name"] = "General Conversation"
        
        # Add category if provided
        if category:
            update_data["category"] = category
        
        # Update the document
        result = await collection.update_one(
            {"conversation_id": conversation_id},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
    
    # Tool Data Operations
    
    async def save_tool_execution(self, 
                                  tool_name: str, 
                                  parameters: Dict[str, Any], 
                                  result: Dict[str, Any], 
                                  user_id: Optional[str] = None,
                                  conversation_id: Optional[str] = None) -> str:
        """
        Save a tool execution record
        
        Args:
            tool_name: The name of the tool
            parameters: The parameters passed to the tool
            result: The result of the tool execution
            user_id: The ID of the user who made the request
            conversation_id: Optional conversation ID for the execution context
            
        Returns:
            The ID of the new record
        """
        collection = self._db.tool_executions
        
        # Create the document
        doc = {
            "tool_name": tool_name,
            "parameters": parameters,
            "result": result,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "timestamp": {"$date": {"$now": True}}
        }
        
        # Insert the document
        result = await collection.insert_one(doc)
        return str(result.inserted_id)
    
    async def get_tool_execution_history(self, 
                                         user_id: Optional[str] = None, 
                                         conversation_id: Optional[str] = None,
                                         tool_name: Optional[str] = None, 
                                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of tool executions
        
        Args:
            user_id: Filter by user ID
            conversation_id: Filter by conversation ID
            tool_name: Filter by tool name
            limit: Maximum number of records to return
            
        Returns:
            A list of tool execution records
        """
        collection = self._db.tool_executions
        
        # Build the query
        query = {}
        if user_id:
            query["user_id"] = user_id
        if conversation_id:
            query["conversation_id"] = conversation_id
        if tool_name:
            query["tool_name"] = tool_name
        
        # Execute the query
        cursor = collection.find(query).sort("timestamp", -1).limit(limit)
        
        # Convert to list
        return await cursor.to_list(length=limit)
    
    # User Data Operations
    
    async def save_user_data(self, user_id: str, data: Dict[str, Any]) -> bool:
        """
        Save user data
        
        Args:
            user_id: The ID of the user
            data: The user data to save
            
        Returns:
            True if successful, False otherwise
        """
        collection = self._db.users
        
        # Update or insert the document
        result = await collection.update_one(
            {"user_id": user_id},
            {"$set": data},
            upsert=True
        )
        
        return result.acknowledged
    
    async def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user data
        
        Args:
            user_id: The ID of the user
            
        Returns:
            The user data, or None if not found
        """
        collection = self._db.users
        
        # Execute the query
        result = await collection.find_one({"user_id": user_id})
        
        return result
    
    # Persona Data Operations
    
    async def save_persona_summary(self, 
                                user_id: str, 
                                model_id: str, 
                                output_persona_json: dict,
                                session_token: Optional[str] = None,
                                is_summary: bool = True,
                                conversation_id: Optional[str] = None,) -> str:
        """
        Save a persona summary
        
        Args:
            user_id: The ID of the user
            model_id: The ID of the model
            persona_name: The name of the persona
            summary: The persona summary text
            session_token: Optional session token
            is_summary: Whether this is a summary (True) or a conversation message (False)
            conversation_id: Optional conversation ID
            
        Returns:
            The ID of the saved persona summary
        """
        collection = self._db.persona_summaries

        try:
            # Set previous entries for this user and conversation_id to conversation_status = False
            # if conversation_id:
            await collection.update_many(
                {"user_id": user_id, "model_id": model_id},
                {"$set": {"conversation_status": False}}
            )
        except Exception as e:
            logger.error(f"Error while setting conversation_status to false. Error: {e}")

        summary_id_list = []
        for persona_name, summary in output_persona_json.items():
            logger.info(f"Storing persona: {persona_name}")
            # Create the new document with conversation_status = True
            now = datetime.utcnow().isoformat()
            summary_id = str(uuid.uuid4())

            doc = {
                "summary_id": summary_id,
                "user_id": user_id,
                "model_id": model_id,
                "persona_name": persona_name,
                "data": summary,
                "is_summary": is_summary,
                "created_at": now,
                "ai_data": True,
                "user_data": False,
                "session_token": session_token,
                "conversation_status": True  # New field added here
            }

            # Insert the new document
            await collection.insert_one(doc)

            logger.info(f"Saved persona summary for {persona_name}")
            summary_id_list.append(summary_id)
        
        return summary_id_list
    
    async def save_parameters_to_context(self, user_id: str, conversation_id: str, new_parameters: Dict[str, Any]):
        """
        Save or update parameters for a given user and conversation.
        Only overwrite fields that are not None or empty strings.
        """
        collection = self._db.parameter_context
        query = {"user_id": user_id, "conversation_id": conversation_id}
        existing = await collection.find_one(query)

        if existing:
            updated_parameters = existing.get("parameters", {})
            for key, value in new_parameters.items():
                if value not in [None, ""]:
                    updated_parameters[key] = value
            await collection.update_one(query, {"$set": {"parameters": updated_parameters}})
        else:
            filtered_parameters = {k: v for k, v in new_parameters.items() if v not in [None, ""]}
            doc = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "parameters": filtered_parameters
            }
            await collection.insert_one(doc)

    async def get_parameters_from_context(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Retrieve parameters for a given user and conversation.
        """
        collection = self._db.parameter_context
        query = {"user_id": user_id, "conversation_id": conversation_id}
        document = await collection.find_one(query)
        return convert_objectid(document.get("parameters", {})) if document else {}

    async def get_persona_summaries(self, 
                                    user_id: str, 
                                    model_id: str, 
                                    conversation_id: str, 
                                    conversation_status: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve persona summaries based on user_id, model_id, conversation_id, and conversation_status.

        Args:
            user_id: The ID of the user
            model_id: The ID of the model
            conversation_id: The conversation ID to filter by
            conversation_status: Whether to get the latest (True) or older (False) entries

        Returns:
            A list of persona summary documents
        """
        collection = self._db.persona_summaries

        query = {
            "user_id": user_id,
            "model_id": model_id,
            #"conversation_id": conversation_id,
            "conversation_status": conversation_status
        }

        cursor = collection.find(query)
        results = await cursor.to_list(length=None)
        
        # Convert ObjectId to string in the results
        return convert_objectid(results)

    # Email Personalization Operations
    
    async def save_email_personalization(self, 
                                         user_id: str, 
                                         model_id: str, 
                                         persona_name: str,
                                         incentive: str,
                                         call_to_action: str,
                                         personalized_content: str,
                                         problem_statement: Optional[str] = None,
                                         session_token: Optional[str] = None,
                                         conversation_id: Optional[str] = None) -> str:
        """
        Save an email personalization
        
        Args:
            user_id: The ID of the user
            model_id: The ID of the model
            persona_name: The name of the persona
            incentive: The incentive text
            call_to_action: The call to action text
            personalized_content: The personalized content
            problem_statement: Optional problem statement
            session_token: Optional session token
            conversation_id: Optional conversation ID
            
        Returns:
            The ID of the new record
        """
        collection = self._db.email_personalizations
        
        # Create the document
        doc = {
            "user_id": user_id,
            "model_id": model_id,
            "persona_name": persona_name,
            "incentive": incentive,
            "call_to_action": call_to_action,
            "personalized_content": personalized_content,
            "problem_statement": problem_statement,
            "session_token": session_token,
            "conversation_id": conversation_id,
            "created_at": {"$date": {"$now": True}}
        }
        
        # Insert the document
        result = await collection.insert_one(doc)
        return str(result.inserted_id)
    
    async def get_email_personalizations(self, 
                                         user_id: str, 
                                         model_id: Optional[str] = None,
                                         conversation_id: Optional[str] = None,
                                         problem_statement: Optional[str] = None,
                                         session_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get email personalizations
        
        Args:
            user_id: The ID of the user
            model_id: Optional ID of the model
            conversation_id: Optional conversation ID
            problem_statement: Optional problem statement
            session_token: Optional session token
            
        Returns:
            A list of email personalization records
        """
        collection = self._db.email_personalizations
        
        # Build the query
        query = {
            "user_id": user_id,
        }
        if model_id:
            query["model_id"] = model_id
        if conversation_id:
            query["conversation_id"] = conversation_id    
        if problem_statement:
            query["problem_statement"] = problem_statement
        if session_token:
            query["session_token"] = session_token
        
        # Execute the query
        cursor = collection.find(query).sort("created_at", -1)
        
        # Convert to list
        return await cursor.to_list(length=100)
    
    async def save_direct_mail_personalization(self, 
                                         user_id: str, 
                                         model_id: str, 
                                         persona_name: str,
                                         incentive: str,
                                         call_to_action: str,
                                         personalized_content: str,
                                         problem_statement: Optional[str] = None,
                                         session_token: Optional[str] = None,
                                         conversation_id: Optional[str] = None) -> str:
        """
        Save a direct mail personalization
        
        Args:
            user_id: The ID of the user
            model_id: The ID of the model
            persona_name: The name of the persona
            incentive: The incentive text
            call_to_action: The call to action text
            personalized_content: The personalized content
            problem_statement: Optional problem statement
            session_token: Optional session token
            conversation_id: Optional conversation ID
            
        Returns:
            The ID of the new record
        """
        collection = self._db.directmail_personalizations
        
        # Create the document
        doc = {
            "user_id": user_id,
            "model_id": model_id,
            "persona_name": persona_name,
            "incentive": incentive,
            "call_to_action": call_to_action,
            "personalized_content": personalized_content,
            "problem_statement": problem_statement,
            "session_token": session_token,
            "conversation_id": conversation_id,
            "created_at": {"$date": {"$now": True}}
        }
        
        # Insert the document
        result = await collection.insert_one(doc)
        return str(result.inserted_id)
    
    async def get_direct_mail_personalizations(self, 
                                         user_id: str, 
                                         model_id: Optional[str] = None,
                                         conversation_id: Optional[str] = None,
                                         problem_statement: Optional[str] = None,
                                         session_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get direct mail personalizations
        
        Args:
            user_id: The ID of the user
            model_id: Optional ID of the model
            conversation_id: Optional conversation ID
            problem_statement: Optional problem statement
            session_token: Optional session token
            
        Returns:
            A list of direct mail personalization records
        """
        collection = self._db.directmail_personalizations
        
        # Build the query
        query = {
            "user_id": user_id,
        }
        if model_id:
            query["model_id"] = model_id
        if conversation_id:
            query["conversation_id"] = conversation_id    
        if problem_statement:
            query["problem_statement"] = problem_statement
        if session_token:
            query["session_token"] = session_token
        
        # Execute the query
        cursor = collection.find(query).sort("created_at", -1)
        
        # Convert to list
        return await cursor.to_list(length=100)
    
    async def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get all messages for a conversation
        
        Args:
            conversation_id: The ID of the conversation
            limit: The maximum number of messages to retrieve (0 means no limit)
            
        Returns:
            A list of message documents
        """
        try:
            conversation = await self.get_conversation(conversation_id)
            print("***************conversation", conversation)
            
            if not conversation:
                return []
            
            messages = conversation.get('messages', [])
            
            # # Apply limit if specified and greater than 0
            # if limit > 0 and len(messages) > limit:
            #     messages = messages[-limit:]
            
            # Convert ObjectId to string in all messages
            messages = convert_objectid(messages)
            
            return messages
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
            
    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation by ID
        
        Args:
            conversation_id: The ID of the conversation to delete
            
        Returns:
            True if the conversation was deleted, False otherwise
        """
        collection = self._db.conversations
        
        try:
            # Delete the conversation
            result = await collection.delete_one({"conversation_id": conversation_id})
            
            # Check if the conversation was deleted
            if result.deleted_count == 1:
                logger.info(f"Conversation {conversation_id} deleted successfully")
                return True
            else:
                logger.warning(f"Conversation {conversation_id} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            return False

    async def find_documents(self, collection: str, query: dict, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Find documents in a collection based on a query
        
        Args:
            collection: The name of the collection
            query: The query to filter documents
            limit: The maximum number of documents to retrieve
            
        Returns:
            A list of document objects
        """
        try:
            # Get the collection
            coll = self._db[collection]
            
            # Execute the query
            cursor = coll.find(query)
            if limit > 0:
                cursor = cursor.limit(limit)
            
            # Convert to list
            results = await cursor.to_list(length=None)
            
            # Convert ObjectId to string
            return convert_objectid(results)
        except Exception as e:
            logger.error(f"Error finding documents in {collection}: {str(e)}")
            return []
    
    async def ensure_collections(self):
        """
        Ensure required collections exist
        """
        try:
            # Get a list of existing collections
            collections = await self._db.list_collection_names()
            
            # Create collections if they don't exist
            required_collections = [
                "conversations", 
                "messages", 
                "tool_executions", 
                "persona_summaries",
                "email_personalizations",
                "directmail_personalizations",
                "digitalad_personalizations",
                "users",
                "generalized_personalization"
            ]
            
            for collection_name in required_collections:
                if collection_name not in collections:
                    logger.info(f"Creating collection: {collection_name}")
                    await self._db.create_collection(collection_name)
                    
            logger.info("Ensured required collections exist")
            
        except Exception as e:
            logger.error(f"Error ensuring collections: {str(e)}")
            # Continue even if this fails to avoid breaking startup 

    async def delete_persona_summaries(self, user_id: str, model_id: str) -> bool:
        """
        Delete all persona summaries for a given user_id and model_id
        
        Args:
            user_id: The ID of the user
            model_id: The ID of the model
            
        Returns:
            True if successful, False otherwise
        """
        collection = self._db.persona_summaries
        
        try:
            # Delete all persona summaries for this user and model
            result = await collection.delete_many({
                "user_id": user_id,
                "model_id": model_id
            })
            
            logger.info(f"Deleted {result.deleted_count} persona summaries for user_id={user_id}, model_id={model_id}")
            return result.acknowledged
        except Exception as e:
            logger.error(f"Error deleting persona summaries: {str(e)}")
            return False 

    async def get_context_value(self, user_id: str, conversation_id: str, key: str) -> Optional[Any]:
        """
        Get a specific value from the context for a conversation
        
        Args:
            user_id: The ID of the user
            conversation_id: The ID of the conversation
            key: The key to retrieve from the context
            
        Returns:
            The value if found, or None if not found
        """
        collection = self._db.conversations
        
        # Find the conversation
        conversation = await collection.find_one({"user_id": user_id, "conversation_id": conversation_id})
        
        if not conversation:
            logger.warning(f"Conversation not found for user_id={user_id} and conversation_id={conversation_id}")
            return None
            
        # Get the context
        context = conversation.get("context", {})
        
        # Return the value if it exists, or None
        return context.get(key) 

    async def save_digitalad_personalization(self, 
                                         user_id: str, 
                                         model_id: str, 
                                         persona_name: str,
                                         incentive: str,
                                         headlines: list,
                                         call_to_action: str,
                                         personalized_content: str,
                                         problem_statement: Optional[str] = None,
                                         session_token: Optional[str] = None,
                                         conversation_id: Optional[str] = None) -> str:
        """
        Save a digital ad personalization
        
        Args:
            user_id: The ID of the user
            model_id: The ID of the model
            persona_name: The name of the persona
            incentive: The incentive text
            headlines: The headlines for the ad
            call_to_action: The call to action text
            personalized_content: The personalized content
            recommended_platforms: List of recommended platforms for the ad
            problem_statement: Optional problem statement
            session_token: Optional session token
            conversation_id: Optional conversation ID
            
        Returns:
            The ID of the new record
        """
        collection = self._db.digitalad_personalizations
        
        # Create the document
        doc = {
            "user_id": user_id,
            "model_id": model_id,
            "persona_name": persona_name,
            "incentive": incentive,
            "headlines": headlines,
            "call_to_action": call_to_action,
            "personalized_content": personalized_content,
            "problem_statement": problem_statement,
            "session_token": session_token,
            "conversation_id": conversation_id,
            "created_at": {"$date": {"$now": True}}
        }
        
        # Insert the document
        result = await collection.insert_one(doc)
        return str(result.inserted_id) 

    async def get_digitalad_personalizations(self, 
                                         user_id: str, 
                                         model_id: Optional[str] = None,
                                         conversation_id: Optional[str] = None,
                                         problem_statement: Optional[str] = None,
                                         session_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get digital ad personalizations for a user
        
        Args:
            user_id: The ID of the user
            model_id: Optional model ID filter
            conversation_id: Optional conversation ID filter
            problem_statement: Optional problem statement filter
            session_token: Optional session token
            
        Returns:
            List of digital ad personalization documents
        """
        collection = self._db.digitalad_personalizations
        
        # Build the query
        query = {"user_id": user_id}
        if model_id:
            query["model_id"] = model_id
        if conversation_id:
            query["conversation_id"] = conversation_id
        if problem_statement:
            query["problem_statement"] = problem_statement
        
        try:
            # Get the personalizations
            cursor = collection.find(query).sort("created_at", -1)
            
            # Convert to list
            personalizations = await cursor.to_list(length=None)
            
            # Convert ObjectId to string
            return convert_objectid(personalizations)
        except Exception as e:
            logger.error(f"Error getting digital ad personalizations: {str(e)}")
            return []
            
    async def save_generalized_personalization(self, 
                                           user_id: str, 
                                           audience_id: str,
                                           filter_criteria: List[str],
                                           conversation_id: Optional[str] = None) -> str:
        """
        Save generalized personalization data based on audience filter criteria
        
        Args:
            user_id: The ID of the user
            audience_id: The ID of the audience
            filter_criteria: List of filter criteria descriptions
            conversation_id: Optional conversation ID
            
        Returns:
            The ID of the saved document
        """
        collection = self._db.generalized_personalization
        
        # Create the document
        now = datetime.utcnow().isoformat()
        doc = {
            "user_id": user_id,
            "audience_id": audience_id,
            "filter_criteria": filter_criteria,
            "conversation_id": conversation_id,
            "created_at": now,
            "updated_at": now
        }
        
        try:
            # Insert the document
            result = await collection.insert_one(doc)
            
            # Return the ID
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving generalized personalization: {str(e)}")
            return None
            
    async def get_generalized_personalization(self, 
                                          user_id: str,
                                          audience_id: Optional[str] = None,
                                          conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get generalized personalization data
        
        Args:
            user_id: The ID of the user
            audience_id: Optional audience ID filter
            conversation_id: Optional conversation ID filter
            
        Returns:
            List of generalized personalization documents
        """
        collection = self._db.generalized_personalization
        
        # Build the query
        query = {"user_id": user_id}
        if audience_id:
            query["audience_id"] = audience_id
        if conversation_id:
            query["conversation_id"] = conversation_id
        
        try:
            # Get the personalizations
            cursor = collection.find(query).sort("created_at", -1)
            
            # Convert to list
            personalizations = await cursor.to_list(length=None)
            
            # Convert ObjectId to string
            return convert_objectid(personalizations)
        except Exception as e:
            logger.error(f"Error getting generalized personalization: {str(e)}")
            return [] 