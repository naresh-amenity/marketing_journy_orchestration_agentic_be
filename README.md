# Persona Tool API

A FastAPI-based application that processes user requests using LangGraph and LLM to determine which tool to call to generate the response. The application uses MongoDB for data persistence and supports conversational interactions.

## Overview

This API provides:
1. A conversational chat interface for natural interactions
2. A unified processing endpoint for direct tool calls
3. Stateful conversation tracking across user sessions
4. Intelligent tool selection based on user intent
5. Automatic guidance when prerequisites are missing

The system maintains conversation context between interactions, allowing users to have natural conversations without needing to provide all parameters in every request.

## Conversational Flow

The chat interface supports a natural conversational flow:
1. Users can start with general greetings ("Hello, how are you?")
2. The system responds conversationally when no tool calls are needed
3. When users express intent to use a tool (e.g., "I want to create a persona narrative"), the system:
   - Identifies the needed tool
   - Requests any missing parameters naturally
   - Guides users when prerequisites are missing (e.g., suggesting to create personas before personalization)
   - Executes the appropriate tool when all requirements are met
4. Context is maintained throughout the conversation, eliminating the need to repeatedly specify the same information

## LangGraph Implementation

The API uses LangGraph to create a stateful processing pipeline with the following nodes:
1. **Tool Decision**: Uses LLM to analyze the user request and determine which tool to use
2. **Validate Tool**: Checks if the selected tool exists and validates required parameters
3. **Execute Tool**: Executes the selected tool with the extracted parameters
4. **Missing Params**: Handles cases where required parameters are missing
5. **Error**: Handles any errors that occur during processing

The flow between these nodes is determined by conditional edges, allowing for dynamic routing based on the state of the request processing.

## MongoDB Integration

The application uses MongoDB for data persistence with the following collections:
- **conversations**: Stores all conversation data, including messages, context, and active persona narratives
- **tool_executions**: Stores all tool execution records with parameters, results, and timestamps
- **persona_summaries**: Stores persona summaries created by the persona_summary tool
- **email_personalizations**: Stores email personalizations created by the email_personalization tool
- **users**: Stores user data

The MongoDB integration provides:
- Asynchronous operations using Motor
- History tracking for all tool executions and conversations
- Data persistence for all tool results
- Query capabilities for retrieving historic data

## Tools Available

- **Persona Summary**: Creates persona summaries
- **Email Personalization**: Creates personalized email content
- **Direct Mail Personalization**: Creates personalized direct mail content
- **Digital Ad Personalization**: Creates personalized digital ad content
- **General Email Personalization**: Creates general email content
- **General Direct Mail Personalization**: Creates general direct mail content
- **General Digital Ad Personalization**: Creates general digital ad content
- **Persona Narrative**: Creates a persona narrative
- **History Conversation**: Retrieves conversation history
- **Journey Creation**: Creates a new journey
- **Journey Status**: Checks journey status
- **Journey Price**: Gets journey pricing
- **Journey Draft**: Handles journey draft upload/download

## Journey Tool

The Journey Tool provides functionality to create and manage marketing journeys:

- Create new marketing journeys
- Check status of existing journeys
- Generate journey reports
- Process journey reports with persona files

### New Document Handling Features

#### Downloading Journey Documents

The API now supports downloading generated journey documents through the standard `/api/v1/process` endpoint:

```json
{
  "action": "document_operation",
  "document_action": "download",
  "journey_id": "your_journey_id",
  "document_name": "optional_specific_document_name.docx"
}
```

The response will contain the document content as base64-encoded data:

```json
{
  "status": "success",
  "message": "Journey document retrieved successfully.",
  "data": {
    "journey_id": "your_journey_id",
    "document_name": "journey_your_journey_id_20240501_123456.docx",
    "document_content": "base64_encoded_content",
    "file_size": 12345,
    "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  }
}
```

You can also use the `get_journey_report` action with the `request_download` parameter:

```json
{
  "action": "get_journey_report",
  "journey_id": "your_journey_id",
  "request_download": true
}
```

#### Uploading Modified Documents

You can upload modified journey documents through the standard `/api/v1/process` endpoint:

```json
{
  "action": "document_operation",
  "document_action": "upload",
  "journey_id": "your_journey_id",
  "document_content": "base64_encoded_content"
}
```

The response will include details about the uploaded file:

```json
{
  "status": "success",
  "message": "Document uploaded successfully.",
  "data": {
    "journey_id": "your_journey_id",
    "file_name": "journey_your_journey_id_20240501_123456_modified.docx",
    "file_path": "journey_reports/journey_your_journey_id_20240501_123456_modified.docx",
    "file_size": 12345,
    "timestamp": "20240501_123456"
  }
}
```

You can also provide the document content directly in the `check_journey_report` action:

```json
{
  "action": "check_journey_report",
  "journey_id": "your_journey_id",
  "document_content": "base64_encoded_content"
}
```

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- MongoDB 4.4+ (local or cloud instance)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Create a `.env` file in the root directory with your OpenAI API key and MongoDB settings:
   ```
   OPENAI_API_KEY=your_api_key_here
   MONGO_URI=mongodb://localhost:27017
   MONGO_DB_NAME=persona_tool_db
   ```

### Running the API

1. Make sure MongoDB is running
2. From the project root directory, run:
   ```
   uvicorn app.main:app --reload
   ```
   or
   ```
   python run.py
   ```
3. The API will be available at `http://localhost:8000`
4. API documentation is available at `http://localhost:8000/docs`

## API Usage

### Endpoint: `/api/chat`

**Method**: POST

**Request Body**:
```json
{
  "query": "Hello, how can you help me with personalization?",
  "user_id": "user123",
  "conversation_id": "optional_conversation_id_for_continuing_chat"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Hello! I'm here to help you with marketing personalization. I can assist you with creating persona narratives and personalizing marketing content like emails, digital ads, and direct mail. What would you like to do today?",
  "conversation_id": "new_conversation_id"
}
```

**Example Flow**:
1. User: "I want to create a persona narrative"
2. System: "To create a persona narrative, I need to know which model ID to use. Could you please provide a model ID?"
3. User: "My model ID is ABC123"
4. System: *creates persona narratives* "I've created the following persona narratives for model ABC123: [persona details]"
5. User: "Now I want to create an email personalization"
6. System: *creates email personalization using previously created personas*

### Endpoint: `/api/process`

**Method**: POST

**Request Body**:
```json
{
  "query": "Create a persona summary for model ABC123",
  "user_id": "user123",
  "model_id": "ABC123",
  "conversation_id": "optional_conversation_id"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Persona summaries created successfully",
  "data": {
    "personas": {
      "Persona A": "Description...",
      "Persona B": "Description...",
      "Persona C": "Description..."
    }
  },
  "tool_used": "persona_summary",
  "pricing": {
    "base_price": 5.0,
    "currency": "USD",
    "unit": "per summary set",
    "additional_fees": {
      "complexity_fee": {
        "amount": 2.0,
        "description": "Additional fee for complex persona summaries"
      }
    }
  },
  "conversation_id": "conversation_id"
}
```

### Endpoint: `/api/conversations/{user_id}`

**Method**: GET

**Parameters**:
- `user_id`: The ID of the user (path parameter)
- `limit`: Maximum number of conversations to return (query parameter, default: 10)

**Response**:
```json
[
  {
    "conversation_id": "conv123",
    "user_id": "user123",
    "created_at": "2023-07-14T12:34:56.789Z",
    "updated_at": "2023-07-14T12:45:56.789Z",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how can you help me?",
        "timestamp": "2023-07-14T12:34:56.789Z"
      },
      {
        "role": "assistant",
        "content": "Hello! I'm here to help with personalization...",
        "timestamp": "2023-07-14T12:34:58.123Z"
      }
    ],
    "context": {},
    "active_persona_narratives": ["Persona A", "Persona B"],
    "model_id": "ABC123"
  }
]
```

### Endpoint: `/api/conversation/{conversation_id}`

**Method**: GET

**Parameters**:
- `conversation_id`: The ID of the conversation (path parameter)

**Response**:
```json
{
  "conversation_id": "conv123",
  "user_id": "user123",
  "created_at": "2023-07-14T12:34:56.789Z",
  "updated_at": "2023-07-14T12:45:56.789Z",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how can you help me?",
      "timestamp": "2023-07-14T12:34:56.789Z"
    },
    {
      "role": "assistant",
      "content": "Hello! I'm here to help with personalization...",
      "timestamp": "2023-07-14T12:34:58.123Z"
    }
  ],
  "context": {},
  "active_persona_narratives": ["Persona A", "Persona B"],
  "model_id": "ABC123"
}
```

### Endpoint: `/api/history/{user_id}`

**Method**: GET

**Parameters**:
- `user_id`: The ID of the user (path parameter)
- `limit`: Maximum number of records to return (query parameter, default: 10)
- `conversation_id`: Optional conversation ID to filter by (query parameter)

**Response**:
```json
[
  {
    "tool_name": "persona_summary",
    "parameters": {
      "user_id": "user123",
      "model_id": "ABC123"
    },
    "result": {
      "status": "success",
      "message": "Persona summaries created successfully",
      "data": {
        "personas": {
          "Persona A": "Description...",
          "Persona B": "Description...",
          "Persona C": "Description..."
        }
      }
    },
    "user_id": "user123",
    "conversation_id": "conv123",
    "timestamp": "2023-07-14T12:34:56.789Z"
  }
]
```

## Adding New Tools

To add a new tool:

1. Create a new file in the `app/tools` directory
2. Implement a class that inherits from `BaseTool`
3. Register the tool in the `ToolRegistry._initialize_tools` method
4. Add MongoDB operations in the execute method to store tool-specific data
5. Update the ChatProcessor's intent analysis system prompt to include the new tool

## Integration with Existing Project

This API is designed to integrate with the existing persona_poc project:

1. You can reuse utility functions by importing them directly
2. Existing business logic can be wrapped in tool classes
3. MongoDB replaces the original SQL database for improved scalability and flexibility 