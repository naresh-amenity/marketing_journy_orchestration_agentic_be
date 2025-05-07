# Persona Tool API Architecture

## Directory Structure

```
persona_tool_api/
├── .env                       # Environment variables
├── run.py                     # Entry point to run the application
├── requirements.txt           # Python dependencies
├── app/                       # Main application package
│   ├── __init__.py            # Package initialization
│   ├── main.py                # FastAPI application setup
│   ├── agent/                 # Agent architecture components
│   │   ├── __init__.py        # Agent package initialization
│   │   ├── main_agent.py      # MainAgent - core coordinator
│   │   └── chat_handler.py    # Handler for general conversations
│   ├── api/                   # API endpoints
│   │   ├── __init__.py        # API package initialization
│   │   └── routes.py          # API route definitions
│   ├── models/                # Data models
│   │   ├── __init__.py        # Models package initialization
│   │   └── model.py           # Pydantic models for request/response
│   ├── tools/                 # Tool implementations
│   │   ├── __init__.py        # Tools package initialization
│   │   ├── base_tool.py       # Abstract base class for tools
│   │   ├── persona_summary_tool.py          # Persona creation tool
│   │   ├── email_personalization_tool.py    # Email content generation
│   │   └── directmail_personalization.py    # Direct mail content
│   └── utils/                 # Utility functions and classes
│       ├── __init__.py        # Utils package initialization
│       ├── db.py              # MongoDB connection and operations
│       └── tool_registry.py   # Registry for tool management
```

## Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                         FastAPI                         │
└───────────────────────────┬─────────────────────────────┘
                            │
                ┌───────────▼───────────┐
                │     API Routes        │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │      MainAgent        │ ◄────────────────┐
                └───────────┬───────────┘                  │
                            │                              │
        ┌─────────────────┬─┴───────────┬────────────────┐ │
        │                 │             │                │ │
┌───────▼─────┐   ┌───────▼─────┐ ┌─────▼──────┐  ┌─────▼──▼───┐
│ Tool Selection│   │ Parameter   │ │   Tool     │  │    DB     │
│     LLM      │   │ Management  │ │ Execution  │  │ Operations │
└───────┬─────┘   └─────────────┘ └──────┬─────┘  └────────────┘
        │                                │
        │           ┌────────────────────┘
        │           │
┌───────▼───────┐   │
│ ChatHandler   │   │
└───────────────┘   │
                    │
          ┌─────────▼──────────┐
          │    Tool Registry   │
          └─────────┬──────────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
┌──────▼─────┐ ┌────▼─────┐ ┌───▼──────┐
│  Persona   │ │  Email   │ │DirectMail│
│   Tool     │ │   Tool   │ │   Tool   │
└────────────┘ └──────────┘ └──────────┘
```

## Data Flow

1. **Request Processing**:
   ```
   Client → API Routes → MainAgent → Tool Selection → Selected Tool → Response
   ```

2. **Conversation Flow**:
   ```
   Client Query → MainAgent → ChatHandler → Response
   ```

3. **Tool Parameter Resolution**:
   ```
   MainAgent → Parameter Check → DB Lookup → Tool Execution
   ```

4. **Tool Registration**:
   ```
   Tool Registry → Load All Tools → Make Available to MainAgent
   ```

## Key Interactions

- **MainAgent** coordinates all activities, making decisions on tool selection
- **Tool Registry** maintains the available tools and provides them on demand
- **ChatHandler** processes general conversations that don't need specific tools
- **MongoDB** stores conversation history and context for continuity

## Benefits of the Architecture

1. **Centralized Coordination**: One agent manages all tools and decisions
2. **Clear Separation of Concerns**: Each component has a specific responsibility 
3. **Flexible Tool Discovery**: Tools are registered at startup and can be added easily
4. **Context Continuity**: Conversation history is preserved across interactions
5. **Unified Interface**: Single endpoint for all operations 