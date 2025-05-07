# Migration Guide

This guide explains how to migrate from the previous architecture to the new MainAgent-based architecture.

## Overview of Changes

1. Replaced the separate GraphProcessor and ChatProcessor with a unified MainAgent
2. Created a dedicated ChatHandler for general conversations
3. Updated API routes to use the MainAgent
4. Simplified parameter passing and tool execution
5. Enhanced conversation context management

## Step-by-Step Migration

### 1. Directory Structure Changes

Create the agent directory structure:

```
app/
└── agent/
    ├── __init__.py
    ├── main_agent.py
    └── chat_handler.py
```

### 2. Update Dependencies

No new dependencies are required, but ensure all the following are installed:

```
fastapi
uvicorn
langchain
langchain-openai
python-dotenv
motor
pydantic
```

### 3. Code Migration

#### 3.1 API Routes

Replace the complex routing logic in `app/api/routes.py` with the simplified MainAgent approach:

```python
@router.post("/process")
async def process_request(request: UserRequest, req: Request):
    try:
        # Get MongoDB instance
        mongodb = req.app.mongodb
        
        # Handle conversation ending
        if request.is_conversation_end:
            # Update conversation as ended
            if request.user_id and request.conversation_id:
                await mongodb.update_conversation(
                    conversation_id=request.conversation_id,
                    user_id=request.user_id,
                    updates={"is_ended": True},
                    conversation_name=request.conversation_name
                )
                return ApiResponse(
                    status="success",
                    message="Conversation ended successfully",
                    conversation_id=request.conversation_id,
                    is_conversation_end=True
                )
        
        # Create MainAgent and process the request
        main_agent = MainAgent(mongodb)
        response = await main_agent.process_request(request)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return ApiResponse(
            status="error",
            message=f"An error occurred: {str(e)}",
            conversation_id=request.conversation_id
        )
```

#### 3.2 Database Integration

Ensure your database utilities support these methods that are used by the MainAgent:

- `get_conversation`
- `get_conversation_history`
- `add_message_to_conversation`
- `create_conversation`
- `get_context_value`
- `update_conversation`

#### 3.3 Tools Integration

Tools require no changes as long as they implement the BaseTool interface with:

- `get_name()`
- `get_description()`
- `get_required_params()`
- `get_optional_params()`
- `execute(parameters)`

### 4. Configuration Updates

Ensure your `.env` file has these variables:

```
openai_key=your_openai_key
LANGCHAIN_API_KEY=your_langchain_key
MONGODB_CONNECTION_STRING=your_mongodb_connection_string
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

### 5. Testing Migration

1. Start with sanity tests:
   ```bash
   python run.py
   ```

2. Test basic conversation:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/process" \
     -H "Content-Type: application/json" \
     -d '{"query": "Hello, how can you help me?", "user_id": "test_user"}'
   ```

3. Test tool execution:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/process" \
     -H "Content-Type: application/json" \
     -d '{"query": "Create a persona for model ID 12345", "user_id": "test_user", "model_id": "12345"}'
   ```

## Troubleshooting

### Known Issues

1. **Missing Parameters**: If you notice "missing parameters" errors, check that all required parameters are being properly extracted from the user query and passed to tools.

2. **Tool Selection**: If the wrong tool is being selected, you might need to adjust the `TOOL_SELECTION_PROMPT` in the MainAgent class.

3. **Database Connection**: Ensure that MongoDB is correctly configured and accessible.

### Migration Verification

To verify your migration was successful:

1. Check that conversations are properly stored in the database
2. Confirm that tools are being correctly selected based on user queries
3. Verify that conversation history is maintained between requests
4. Test that parameter extraction works correctly

## Rollback Plan

If issues arise, you can revert to the previous architecture by:

1. Restoring the original `app/api/routes.py` file
2. Removing the new `app/agent/` directory
3. Ensuring the original `app/utils/graph_processor.py` and `app/utils/chat_processor.py` are still in place

## Benefits of Migration

After successful migration, you should see:

1. Simpler code and easier maintenance
2. More reliable tool selection and parameter handling
3. Better conversation management
4. Easier extension with new tools
5. Improved error handling and context management 