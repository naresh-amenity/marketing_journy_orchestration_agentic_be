from typing import Dict, Any, List, Optional, Type
import logging
import importlib
import inspect
from app.tools.base_tool import BaseTool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolRegistry:
    """
    Registry for all available tools
    """
    _instance = None
    _tools = {}
    
    def __new__(cls):
        """
        Implements singleton pattern to ensure only one registry instance exists
        """
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            # Initialize the tools registry
            cls._instance._initialize_tools()
        return cls._instance
    
    def _initialize_tools(self):
        """
        Initialize the tools registry by importing all tool modules
        """
        try:
            # Import all tool modules
            from app.tools.persona_summary_tool import PersonaSummaryTool
            from app.tools.email_personalization_tool import EmailPersonalizationTool
            from app.tools.directmail_personalization import DirectMailPersonalizationTool
            from app.tools.persona_explain_tool import PersonaExplainTool
            from app.tools.digitalad_personalization import DigitalAdPersonalizationTool
            from app.tools.journey_tool import JourneyTool
            # from app.tools.directmail_personalization_tool import DirectMailPersonalizationTool
            # from app.tools.digitalad_personalization_tool import DigitalAdPersonalizationTool
            # from app.tools.general_email_personalization_tool import GeneralEmailPersonalizationTool
            # from app.tools.general_directmail_personalization_tool import GeneralDirectMailPersonalizationTool
            # from app.tools.general_digitalad_personalization_tool import GeneralDigitalAdPersonalizationTool
            # from app.tools.persona_narrative_tool import PersonaNarrativeTool
            # from app.tools.history_conversation_tool import HistoryConversationTool
            # from app.tools.journey_creation_tool import JourneyCreationTool
            # from app.tools.journey_status_tool import JourneyStatusTool
            # from app.tools.journey_price_tool import JourneyPriceTool
            # from app.tools.journey_draft_tool import JourneyDraftTool
            
            # Register each tool
            self.register_tool("persona_summary", PersonaSummaryTool())
            self.register_tool("email_personalization", EmailPersonalizationTool())
            self.register_tool("directmail_personalization", DirectMailPersonalizationTool())
            self.register_tool("persona_explain", PersonaExplainTool())
            self.register_tool("digitalad_personalization", DigitalAdPersonalizationTool())
            self.register_tool("journy_tool", JourneyTool())
            # self.register_tool("directmail_personalization", DirectMailPersonalizationTool())
            # self.register_tool("digitalad_personalization", DigitalAdPersonalizationTool())
            # self.register_tool("general_email_personalization", GeneralEmailPersonalizationTool())
            # self.register_tool("general_directmail_personalization", GeneralDirectMailPersonalizationTool())
            # self.register_tool("general_digitalad_personalization", GeneralDigitalAdPersonalizationTool())
            # self.register_tool("persona_narrative", PersonaNarrativeTool())
            # self.register_tool("history_conversation", HistoryConversationTool())
            # self.register_tool("journey_creation", JourneyCreationTool())
            # self.register_tool("journey_status", JourneyStatusTool())
            # self.register_tool("journey_price", JourneyPriceTool())
            # self.register_tool("journey_draft", JourneyDraftTool())
            
        except Exception as e:
            logger.error(f"Error initializing tools: {str(e)}", exc_info=True)
    
    def register_tool(self, name: str, tool: BaseTool):
        """
        Register a tool with the registry
        
        Args:
            name: The name of the tool
            tool: The tool instance
        """
        self._tools[name] = tool
        logger.info(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name
        
        Args:
            name: The name of the tool
            
        Returns:
            The tool instance or None if not found
        """
        return self._tools.get(name)
    
    def has_tool(self, name: str) -> bool:
        """
        Check if a tool exists in the registry
        
        Args:
            name: The name of the tool
            
        Returns:
            True if the tool exists, False otherwise
        """
        return name in self._tools
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """
        Get all registered tools
        
        Returns:
            A dictionary of all registered tools
        """
        return self._tools 