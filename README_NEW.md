# Persona Tool API - MainAgent Architecture

## Overview

This project implements a unified agent architecture for a marketing personalization system. The core of the architecture is the `MainAgent` that handles all requests, automatically determines which tools to use, and manages conversations without requiring manual parameter passing between components.

## Architecture

### Key Components

1. **MainAgent**: The central component that coordinates all operations
   - Tool selection
   - Parameter extraction and validation
   - Tool execution
   - Conversation management

2. **Tools**: Specialized components for specific tasks
   - Persona Summary Tool
   - Email Personalization Tool 
   - DirectMail Personalization Tool
   - Digital Ad Personalization Tool

3. **ChatHandler**: Handles general conversations that don't require specific tools

4. **MongoDB**: Persistent storage for conversations and context

### How It Works

1. The user sends a request to the API endpoint.
2. The `MainAgent` analyzes the request using LLM-based tool selection.
3. If a specific tool is needed, the agent:
   - Extracts parameters from the request
   - Validates parameters and checks for missing ones
   - Attempts to retrieve missing parameters from conversation context
   - Executes the appropriate tool
   - Returns the result

4. If no specific tool is needed, the agent forwards the request to the `ChatHandler` for general conversation.

5. All conversation history is maintained in MongoDB for context awareness.

## Advantages Over Previous Architecture

The new architecture provides several improvements:

1. **Unified Processing**: A single agent handles all requests, eliminating the need to manually pass tool outputs to chat processors.

2. **Simplified Flow**: Streamlined request handling with consistent parameter extraction and validation.

3. **Better Context Management**: Improved tracking and retrieval of context from conversation history.

4. **Easier Maintenance**: Centralized error handling and logging.

5. **Enhanced Extensibility**: Adding new tools requires minimal changes to the core system.

## API Usage

### Process Endpoint

```
POST /api/v1/process
```

Request Body:
```json
{
  "query": "Create a persona for model ID 12345",
  "user_id": "user123",
  "model_id": "12345",
  "session_token": "token123",
  "conversation_id": "conv456",
  "conversation_name": "My Persona Project",
  "is_conversation_end": false
}
```

Response:
```json
{
  "status": "success",
  "message": "Persona created successfully...",
  "data": { ... },
  "tool_used": "persona_summary",
  "conversation_id": "conv456",
  "conversation_name": "My Persona Project",
  "conversation_stage": "completed_persona",
  "suggested_next_message": "Would you like to create email content for this persona?"
}
```

## Setup and Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables in `.env` file:
   ```
   openai_key=your_openai_key
   LANGCHAIN_API_KEY=your_langchain_key
   MONGODB_CONNECTION_STRING=your_mongodb_connection_string
   HOST=0.0.0.0
   PORT=8000
   DEBUG=True
   ```
4. Run the application: `python run.py`

## Development

### Adding New Tools

1. Create a new tool class that extends `BaseTool` in the `app/tools` directory
2. Implement the required methods: `get_name`, `get_description`, `get_required_params`, `get_optional_params`, and `execute`
3. Register the tool in `ToolRegistry._initialize_tools()`

### Modifying Tool Selection

The tool selection logic can be customized by modifying the `TOOL_SELECTION_PROMPT` in the `MainAgent` class.

## Conclusion

This architecture provides a robust and maintainable solution for handling various personalization tasks through a unified agent interface, eliminating the need for manual parameter passing between components. 