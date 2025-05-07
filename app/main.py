from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.utils.db import MongoDB
from app.agent.main_agent import MainAgent
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log API keys (masked for security)
openai_key = os.getenv("openai_key")
if openai_key:
    masked_key = openai_key[:8] + "..." + openai_key[-4:] if len(openai_key) > 12 else "***"
    logger.info(f"OpenAI API key loaded: {masked_key}")
else:
    logger.warning("OpenAI API key not found in environment variables")

langchain_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_key:
    masked_key = langchain_key[:8] + "..." + langchain_key[-4:] if len(langchain_key) > 12 else "***"
    logger.info(f"LangChain API key loaded: {masked_key}")
    os.environ["LANGCHAIN_API_KEY"] = langchain_key
else:
    logger.warning("LangChain API key not found in environment variables")

app = FastAPI(
    title="Persona Tool API",
    description="""API for processing user requests with intelligent agent architecture:
    - Main agent coordinates all tools and conversations
    - Single endpoint handles both conversational and tool-specific requests
    - Automatically detects appropriate tools based on user queries
    - Maintains conversation state and history
    - Uses LLM to determine the appropriate response or tool to use""",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure journey_reports directory exists
journey_reports_dir = "journey_reports"
if not os.path.exists(journey_reports_dir):
    os.makedirs(journey_reports_dir, exist_ok=True)

# Include the router
app.include_router(router, prefix="/api/v1")

# Startup event to initialize database
@app.on_event("startup")
async def startup_db_client():
    try:
        # Initialize MongoDB connection
        logger.info("Initializing MongoDB connection...")
        app.mongodb = MongoDB()
        
        # Ensure collections exist (async method)
        logger.info("Ensuring collections exist...")
        await app.mongodb.ensure_collections()
        
        logger.info("MongoDB connection initialized successfully")
        
        # Initialize MainAgent (just to log any potential issues at startup)
        logger.info("Testing MainAgent initialization...")
        try:
            main_agent = MainAgent(app.mongodb)
            logger.info("MainAgent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MainAgent: {str(e)}", exc_info=True)
            logger.warning("API will still start, but MainAgent initialization failed")
            
    except Exception as e:
        logger.error(f"Error initializing MongoDB: {str(e)}", exc_info=True)
        # We don't want to stop the app from starting, but log the error

# Shutdown event to close database connection
@app.on_event("shutdown")
async def shutdown_db_client():
    try:
        logger.info("Closing MongoDB connection...")
        if hasattr(app, "mongodb"):
            await app.mongodb.close()
        logger.info("MongoDB connection closed successfully")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {str(e)}", exc_info=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 