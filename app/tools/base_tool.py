from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from app.models.model import ToolResponse

class BaseTool(ABC):
    """
    Base class for all tools
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the tool
        
        Returns:
            The name of the tool
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get the description of the tool
        
        Returns:
            The description of the tool
        """
        pass
    
    @abstractmethod
    def get_required_params(self) -> List[str]:
        """
        Get the list of required parameters for the tool
        
        Returns:
            A list of required parameter names
        """
        pass
    
    @abstractmethod
    def get_optional_params(self) -> List[str]:
        """
        Get the list of optional parameters for the tool
        
        Returns:
            A list of optional parameter names
        """
        pass
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResponse:
        """
        Execute the tool with the given parameters
        
        Args:
            parameters: The parameters for the tool
            
        Returns:
            The response from the tool
        """
        pass
    
    def check_required_params(self, parameters: Dict[str, Any]) -> List[str]:
        """
        Check if all required parameters are present
        
        Args:
            parameters: The parameters for the tool
            
        Returns:
            A list of missing required parameters, or an empty list if all are present
        """
        required_params = self.get_required_params()
        missing_params = []
        
        for param in required_params:
            if param not in parameters or not parameters[param]:
                missing_params.append(param)
                
        return missing_params
    
    def get_pricing(self) -> Dict[str, Any]:
        """
        Get the pricing information for the tool
        
        Returns:
            A dictionary with pricing information
        """
        return {
            "base_price": self._get_base_price(),
            "currency": "USD",
            "unit": self._get_pricing_unit(),
            "additional_fees": self._get_additional_fees()
        }
        
    def _get_base_price(self) -> float:
        """
        Get the base price for the tool
        
        Returns:
            The base price as a float
        """
        return 0.0
        
    def _get_pricing_unit(self) -> str:
        """
        Get the pricing unit for the tool
        
        Returns:
            The pricing unit (e.g., "per request", "per token", etc.)
        """
        return "per request"
        
    def _get_additional_fees(self) -> Dict[str, Any]:
        """
        Get any additional fees for the tool
        
        Returns:
            A dictionary of additional fees
        """
        return {} 