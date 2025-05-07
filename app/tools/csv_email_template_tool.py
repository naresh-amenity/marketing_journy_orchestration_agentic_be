from typing import Dict, Any, List, Optional
import logging
import os
from langchain_openai import ChatOpenAI
from app.tools.base_tool import BaseTool
from app.models.model import ToolResponse
from app.utils.audience_utils import find_audience_by_id, analyze_csv_columns, process_data_in_batches, structured_data_tool, filter_csv_with_segments, create_email_template_from_csv
from app.utils.db import MongoDB
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from io import StringIO
import pandas as pd
import json
import uuid
import requests

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("openai_key")
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key not found in environment variables")

prompt_csv_email_template = """
You are a powerful email marketing specialist tasked with creating a generalized email template for an entire audience based on CSV data. Your goal is to create a template that includes placeholders that can be filled with individual data for each recipient, while maintaining a cohesive and engaging message.

<problem_statement>
{problem_statement}
</problem_statement>

<csv_data_sample>
{csv_data_sample}
</csv_data_sample>

<csv_columns_info>
{csv_columns_info}
</csv_columns_info>

<segmentation_criteria>
{segmentation_criteria}
</segmentation_criteria>

Your task is to provide the following elements for a generalized email marketing campaign:

1. Incentives: Create 2-3 compelling offers that would appeal to this audience based on the CSV data and segmentation criteria. These should:
   - Directly address likely pain points or desires of this audience
   - Use value propositions that would resonate with this audience segment
   - Include specific details (exact discount amounts, specific free items, etc.)
   - Be presented in a way that feels personalized using placeholders from the CSV data

2. Email Subject Lines: Craft 3 attention-grabbing subject line templates (30-50 characters each) that:
   - Would achieve high open rates with this audience
   - Create curiosity or urgency without appearing as spam
   - Include placeholders for personalization elements from the CSV data
   - Directly connect to the incentives and audience characteristics

3. Call to Action Options: Provide 3 effective call-to-action templates that:
   - Are clear, compelling, and simple to execute
   - Use action-oriented language that suits this audience
   - Can include placeholders for personalization from the CSV data
   - Each has a brief explanation of when it would be most effective

4. Email Template Content: Create a highly effective, generalizable email template that:
   - Has a personalized greeting using placeholders like {{first_name}}
   - Includes placeholders for other personalization points from the CSV data
   - Addresses the likely needs, desires, and pain points of this audience
   - Uses appropriate language, tone, and terminology for this audience
   - Includes placeholders where the offer/incentive would be inserted
   - Structures the email for easy scanning (bullet points, short paragraphs)
   - Incorporates a sense of urgency or exclusivity
   - Concludes with a strong call to action using one of your CTA templates
   - Includes specific instructions on which columns to use for personalization

5. Personalization Strategy: Provide clear guidance on:
   - Which CSV columns should be used for personalization and where
   - How to effectively use each data point in the email
   - Alternative texts for missing data points
   - Advanced personalization techniques based on combinations of data points

Return your response in the following JSON format:

{{
  "incentives": [
    "{{first_name}}, enjoy 25% off your next purchase of {{product_category}} products!",
    "Free shipping on all orders over $50 for our {{loyalty_tier}} members like you, {{first_name}}!",
    "Buy one {{product_category}} item and get the second 50% off - exclusively for {{first_name}}!"
  ],
  "subject_lines": [
    "{{first_name}}, Your Special {{product_category}} Offer Inside!",
    "Exclusive Deal for Our {{loyalty_tier}} Members",
    "{{first_name}}, We've Missed You! Come Back & Save"
  ],
  "call_to_action_options": [
    {{
      "text": "Shop Now with Your {{discount_percentage}}% Discount",
      "usage": "Best for time-limited offers with specific discount amounts"
    }},
    {{
      "text": "Redeem Your {{loyalty_tier}} Member Benefits Today",
      "usage": "Effective for loyalty program members where tier status is known"
    }},
    {{
      "text": "Complete Your Collection - View {{product_category}} Recommendations",
      "usage": "Ideal for cross-selling based on previous purchase categories"
    }}
  ],
  "email_template": "Hi {{first_name}},\\n\\nWe noticed it's been {{days_since_last_purchase}} days since your last purchase, and we wanted to reach out with something special just for you.\\n\\n[INCENTIVE]\\n\\nAs one of our valued {{loyalty_tier}} members, we've selected items we think you'll love based on your interest in {{product_category}}.\\n\\n[PRODUCT RECOMMENDATIONS]\\n\\nRemember, this offer is only available until [EXPIRATION DATE], so don't wait too long!\\n\\n[CALL TO ACTION]\\n\\nThank you for being a loyal customer.\\n\\nBest regards,\\nThe Team",
  "personalization_strategy": {{
    "key_columns": [
      {{
        "column_name": "first_name",
        "usage": "Use in greeting and throughout email for personal touch",
        "fallback": "Valued Customer"
      }},
      {{
        "column_name": "loyalty_tier",
        "usage": "Highlight member status to increase exclusivity feeling",
        "fallback": "Preferred"
      }},
      {{
        "column_name": "product_category",
        "usage": "Tailor offers to customer's previous interests",
        "fallback": "featured"
      }},
      {{
        "column_name": "days_since_last_purchase",
        "usage": "Create urgency based on recency",
        "fallback": "recent"
      }}
    ],
    "advanced_techniques": [
      "If days_since_last_purchase > 90, use 'We miss you!' in subject line",
      "For customers with loyalty_tier = 'Gold', include an extra bonus incentive",
      "If purchase_frequency > 5, acknowledge their loyalty with special recognition"
    ]
  }}
}}

Remember that effective email templates:
- Include easy-to-replace placeholders in {{double_curly_braces}}
- Balance personalization with a template that works for all users
- Have clear fallback options for missing data
- Are designed for mobile-friendly viewing
- Create a sense of urgency and exclusivity
- Have clear and compelling CTAs
"""

class CSVEmailTemplateSchema(BaseModel):
    incentives: List[str] = Field(description="List of incentive templates with placeholders")
    subject_lines: List[str] = Field(description="List of subject line templates with placeholders")
    call_to_action_options: List[Dict] = Field(description="List of CTA options with usage guidance")
    email_template: str = Field(description="Complete email template with placeholders")
    personalization_strategy: Dict = Field(description="Strategy for personalizing the email")

class CSVEmailTemplateTool(BaseTool):
    """
    Tool for creating generalized email templates from CSV data
    """
    def __init__(self):
        self.db = MongoDB()
    
    def get_name(self) -> str:
        return "csv_email_template"
    
    def get_description(self) -> str:
        return "Creates a generalized email template based on uploaded CSV data with placeholders for personalization"
    
    def get_required_params(self) -> List[str]:
        return ["user_id", "problem_statement", "target_id"]
    
    def get_optional_params(self) -> List[str]:
        return ["session_token", "conversation_id", "brand_voice"]
    
    def _extract_filter_criteria(self, filter_data: Dict[str, Any]) -> List[str]:
        """
        Extracts segmentation criteria from filter data
        
        Args:
            filter_data: The filter data from structured_data_tool
            
        Returns:
            List of criteria strings
        """
        all_criteria = []
        
        # Try different potential structures of filter data
        filter_sets = []
        if "user_upload_filter_columns_result" in filter_data:
            filter_sets = filter_data["user_upload_filter_columns_result"].get("output", [])
        else:
            filter_sets = filter_data.get("output", [])
            
        # Handle direct nested structure if needed
        if not filter_sets and isinstance(filter_data.get("filter_sets", None), list):
            filter_sets = filter_data.get("filter_sets", [])
        
        # Extract criteria from filter sets
        for filter_set in filter_sets:
            if isinstance(filter_set, dict) and "explanation" in filter_set and filter_set["explanation"]:
                all_criteria.append(filter_set["explanation"])
            elif isinstance(filter_set, dict) and "column_names" in filter_set:
                column_names = filter_set.get("column_names", [])
                if column_names:
                    criteria_desc = f"Criteria based on: {', '.join(str(col) for col in column_names)}"
                    all_criteria.append(criteria_desc)
            elif isinstance(filter_set, list) and len(filter_set) >= 3:
                # Assuming format [column_names, filter_values, explanation]
                if isinstance(filter_set[2], str) and filter_set[2]:
                    all_criteria.append(filter_set[2])
                else:
                    column_names = filter_set[0] if isinstance(filter_set[0], list) else []
                    if column_names:
                        criteria_desc = f"Criteria based on: {', '.join(str(col) for col in column_names)}"
                        all_criteria.append(criteria_desc)
        
        return all_criteria
    
    async def save_email_template(self, 
                                user_id: str,
                                template_data: Dict[str, Any],
                                problem_statement: str,
                                csv_analysis: Dict[str, Any],
                                session_token: Optional[str] = None,
                                conversation_id: Optional[str] = None) -> str:
        """
        Save the generated email template to the database
        
        Args:
            user_id: The ID of the user
            template_data: The generated template data
            problem_statement: The problem statement/campaign goal
            csv_analysis: The CSV analysis results
            session_token: Optional session token
            conversation_id: Optional conversation ID
            
        Returns:
            The ID of the saved template
        """
        collection = self._db.csv_email_templates
        
        # Create the document
        doc = {
            "user_id": user_id,
            "template_id": template_data.get("template_id", str(uuid.uuid4())),
            "incentives": template_data.get("incentives", []),
            "subject_lines": template_data.get("subject_lines", []),
            "call_to_action_options": template_data.get("call_to_action_options", []),
            "email_template": template_data.get("email_template", ""),
            "personalization_strategy": template_data.get("personalization_strategy", {}),
            "csv_analysis": {
                "total_rows": csv_analysis.get("total_rows", 0),
                "total_columns": csv_analysis.get("total_columns", 0),
                "column_names": csv_analysis.get("column_names", [])
            },
            "problem_statement": problem_statement,
            "segmentation_criteria_used": template_data.get("segmentation_criteria_used", []),
            "session_token": session_token,
            "conversation_id": conversation_id,
            "created_at": {"$date": {"$now": True}}
        }
        
        # Insert the document
        result = await collection.insert_one(doc)
        return str(result.inserted_id)
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResponse:
        """
        Execute the tool with the given parameters
        
        Args:
            parameters: The parameters for the tool
            
        Returns:
            The response from the tool
        """
        try:
            # Extract parameters
            user_id = parameters.get("user_id") # required
            problem_statement = parameters.get("problem_statement") # required
            target_id = parameters.get("target_id") # required - audience target ID
            session_token = parameters.get("session_token", "") # optional
            conversation_id = parameters.get("conversation_id") # optional
            brand_voice = parameters.get("brand_voice", "professional and engaging") # optional
            
            # Check for missing required parameters
            missing_params = []
            if not user_id:
                missing_params.append("user_id")
            if not problem_statement:
                missing_params.append("problem_statement")
            if not target_id:
                missing_params.append("target_id")
            
            if missing_params:
                # Create a conversational response based on what's missing
                response_message = ""
                
                if "target_id" in missing_params:
                    response_message = "I'll need a target audience ID to create a personalized email template. Please provide a target_id parameter."
                elif "problem_statement" in missing_params:
                    response_message = "To create an effective email template, I need to understand your campaign goals. Please provide a brief problem statement or campaign objective."
                else:
                    response_message = "To create a personalized email template from your CSV data, I need a bit more information. " + ", ".join(missing_params) + " would be helpful."
                
                return ToolResponse(
                    status="input_required",
                    message=response_message,
                    required_inputs=missing_params,
                    data=None
                )
            
            # Step 1: Find audience data using find_audience_by_id
            logger.info(f"Finding audience data for target ID: {target_id}")
            USER_ID_PERSONA = user_id
            SESSION_TOKEN_PERSONA = session_token
            TARGET_ID = target_id
            
            audience_data = find_audience_by_id(user_id=USER_ID_PERSONA, SESSION_TOKEN=SESSION_TOKEN_PERSONA, target_id=TARGET_ID)
            if audience_data:
                logger.info(f"Found audience data: {audience_data.get('_id')}")
                if audience_data.get("fileURL") == "dummy_url":
                    # Using dummy data provided by the function
                    file_path = audience_data.get("csv_data")
                    logger.info("Using sample data for audience")
                else:
                    # Real audience data with URL
                    csv_url = audience_data.get("fileURL")
                    response = requests.get(csv_url)
                    csv_data = response.text
                    file_path = csv_data
            else:
                return ToolResponse(
                    status="error",
                    message=f"Could not find audience data for target ID: {target_id}",
                    data=None
                )
            
            # Step 2: Analyze CSV columns
            logger.info("Analyzing CSV columns")
            column_data = analyze_csv_columns(file_path, [], is_propensity_data=False, rows="All", seed=42)
            logger.info(f"Column analysis complete. Found {len(column_data)} columns.")
            
            # Step 3: Process data in batches
            logger.info("Processing data in batches")
            data_batches = process_data_in_batches(column_data)
            logger.info("Batch processing complete")
            
            # Create explanations for columns based on the actual columns in the data
            column_explanations = []
            for col in column_data:
                col_name = col.get("column", "")
                data_type = col.get("data_type", "unknown")
                column_explanations.append(f"{col_name}: {col_name.replace('_', ' ').title()} ({data_type})")
            
            # Step 4: Generate filter data using structured_data_tool
            logger.info("Generating filter data")
            filter_data = structured_data_tool(
                openai_key=OPENAI_API_KEY,
                audiance_data_dict=data_batches,
                problem_statement=problem_statement,
                additional_requirements="Focus on creating segments that will be useful for a general audience campaign",
                explanation=column_explanations,
                other_requirements="Ensure segments are meaningful for creating an effective email template for all users"
            )
            logger.info("Filter data generation complete")
            
            # Step 5: Filter CSV with segments
            logger.info("Filtering CSV with segments")
            filtered_results = filter_csv_with_segments(
                filter_data=filter_data, 
                csv_file_path=file_path, 
                return_dataframe=True
            )
            logger.info(f"Filtering complete. Found {len(filtered_results.get('segments', []))} segments.")
            
            # Step 6: Extract campaign context from filter data
            all_filter_criteria = self._extract_filter_criteria(filter_data)
            
            # Product information template (can be customized or provided in parameters)
            product_info = {
                "name": "Customer Engagement Program",
                "key_benefits": [
                    "Personalized offers based on preferences",
                    "Regular updates on new services", 
                    "Exclusive members-only content"
                ],
                "promotion_code": "WELCOME2023",
                "promotion_discount": "15% off your next purchase"
            }
            
            # Step 7: Create a generalized email template for all users
            logger.info("Creating email template for the entire user database")
            email_template = create_email_template_from_csv(
                csv_data=file_path,
                campaign_context=all_filter_criteria,
                product_info=product_info,
                brand_voice=brand_voice,
                sample_rows=10,
                openai_key=OPENAI_API_KEY
            )
            
            # Add CSV analysis data to the template result
            csv_analysis = {
                "total_rows": filtered_results.get("original_row_count", 0),
                "total_columns": len(column_data),
                "column_names": [col.get("column", "") for col in column_data]
            }
            
            email_template["csv_analysis"] = csv_analysis
            email_template["segmentation_criteria_used"] = all_filter_criteria
            email_template["template_id"] = str(uuid.uuid4())
            
            # Save the template to the database
            template_id = await self.save_email_template(
                user_id=user_id,
                template_data=email_template,
                problem_statement=problem_statement,
                csv_analysis=csv_analysis,
                session_token=session_token,
                conversation_id=conversation_id
            )
            
            # Add the saved template ID to the result
            email_template["database_id"] = template_id
            
            # Save tool execution record
            await self.db.save_tool_execution(
                tool_name=self.get_name(),
                parameters={
                    "user_id": user_id,
                    "problem_statement": problem_statement,
                    "target_id": target_id,
                    "csv_row_count": csv_analysis["total_rows"],
                    "csv_column_count": csv_analysis["total_columns"]
                },
                result={"template_id": email_template.get("template_id")},
                user_id=user_id,
                conversation_id=conversation_id
            )
            
            # Create a rich response for the user
            email_template_preview = email_template.get("email_template", "")
            # Truncate if too long for display
            if len(email_template_preview) > 300:
                email_template_preview = email_template_preview[:297] + "..."
                
            # Format a sample subject line with placeholders
            sample_subject = email_template.get("subject_line_template", "")
            
            message = f"""
I've created a personalized email template based on your audience data with {csv_analysis["total_rows"]} records and {csv_analysis["total_columns"]} columns.

The template includes:
• Personalized subject line template
• Email body with dynamic placeholders
• Tailored call-to-action that fits your audience
• Detailed personalization strategy using your audience data

Sample subject line: {sample_subject}

This template is designed to be merged with your CSV data, automatically inserting the right personalization elements for each recipient. The template was created based on {len(all_filter_criteria)} audience segments identified in your data.
"""
            
            return ToolResponse(
                status="success",
                message=message,
                data={
                    "email_template": email_template,
                    "csv_analysis": csv_analysis,
                    "template_preview": email_template_preview,
                    "segmentation_criteria": all_filter_criteria,
                    "segment_count": len(filtered_results.get("segments", [])),
                    "segment_details": [
                        {
                            "name": segment.get("segment_name"),
                            "explanation": segment.get("explanation"),
                            "size": segment.get("row_count"),
                            "percentage": segment.get("percentage_of_total")
                        }
                        for segment in filtered_results.get("segments", [])
                    ]
                }
            )
            
        except Exception as e:
            logger.error(f"Error executing CSVEmailTemplateTool: {str(e)}", exc_info=True)
            return ToolResponse(
                status="error",
                message=f"Error creating email template: {str(e)}",
                data=None
            )
    
    def _get_base_price(self) -> float:
        """
        Get the base price for the tool
        
        Returns:
            The base price as a float
        """
        return 5.0
        
    def _get_pricing_unit(self) -> str:
        """
        Get the pricing unit for the tool
        
        Returns:
            The pricing unit (e.g., "per request", "per token", etc.)
        """
        return "per template"
        
    def _get_additional_fees(self) -> Dict[str, Any]:
        """
        Get any additional fees for the tool
        
        Returns:
            A dictionary of additional fees
        """
        return {
            "large_csv_fee": {
                "amount": 2.0,
                "description": "Additional fee for CSVs with more than 10,000 rows"
            }
        } 