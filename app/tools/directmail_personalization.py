import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import requests
from bson import ObjectId
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from app.models.model import ToolResponse
from app.tools.base_tool import BaseTool
from app.utils.audience_utils import create_genralize_directemail_template
from app.utils.db import MongoDB
from app.utils.langfush_utils import config_llm_callback, get_prompt_config

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("openai_key")
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key not found in environment variables")

prompt_direct_mail_personalization, config_direct_mail_personalization = (
    get_prompt_config(
        prompt_tag="prompt_direct_mail_personalization_persona", label="stage_v1"
    )
)


class PersonalizationSchema(BaseModel):
    persona_name: List[str] = Field(description="list of persona_name_list")
    Incentive: List[list] = Field(
        description="offer for each persona in nested list format"
    )
    Call_to_Action: List[str] = Field(
        description="selected call to action option in the list format"
    )
    Personalized_Content: List[str] = Field(
        description="Personalized content for each persona in the list format"
    )


class DirectMailPersonalizationTool(BaseTool):
    """
    Tool for creating personalized direct mail content
    """

    def __init__(self):
        self.db = MongoDB()

    def get_name(self) -> str:
        return "directmail_personalization"

    def get_description(self) -> str:
        return "Creates personalized direct mail content based on personas and problem statements"

    def get_required_params(self) -> List[str]:
        return ["user_id", "problem_statement", "personalization_data_type"]

    def get_optional_params(self) -> List[str]:
        return [
            "session_token",
            "model_id",
            "persona_name",
            "audience_id",
            "brand_voice",
        ]

    def personalization_direct_mail(
        self,
        problem_statement,
        persona_details,
        persona_name_list,
        conversation_id,
        user_id,
    ):
        llm = ChatOpenAI(
            model=config_direct_mail_personalization["model"],
            temperature=config_direct_mail_personalization["temperature"],
            api_key=OPENAI_API_KEY,
        )

        parser = JsonOutputParser(pydantic_object=PersonalizationSchema)

        prompt = PromptTemplate(
            template=prompt_direct_mail_personalization,
            input_variables=["problem_statement", "persona_details", "persona_name"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser

        config = config_llm_callback(
            run_name="directmail_personalization_persona",
            tag="directmail_personalization_persona",
            conversation_id=conversation_id,
            user_id=user_id,
        )

        output = chain.invoke(
            {
                "problem_statement": problem_statement,
                "persona_details": persona_details,
                "persona_name": persona_name_list,
            },
            config=config,
        )

        return output

    def direct_mail_llm(
        self,
        problem_statement_,
        persona_details_,
        persona_names_,
        conversation_id,
        user_id,
    ):
        max_retries = 3

        (
            persona_name_list,
            Incentive_list,
            Call_to_Action_list,
            Personalized_Content_list,
        ) = ([], [], [], [])
        for attempt in range(max_retries):
            direct_mail_dict = self.personalization_direct_mail(
                problem_statement=problem_statement_,
                persona_details=persona_details_,
                persona_name_list=persona_names_,
                conversation_id=conversation_id,
                user_id=user_id,
            )

            persona_name_list = direct_mail_dict["persona_name"]
            Incentive_list = direct_mail_dict["Incentive"]
            Call_to_Action_list = direct_mail_dict["Call_to_Action"]
            Personalized_Content_list = direct_mail_dict["Personalized_Content"]

            len_persona_name_list = len(persona_name_list)
            len_Incentive_list = len(Incentive_list)
            len_Call_to_Action_list = len(Call_to_Action_list)
            len_Personalized_Content_list = len(Personalized_Content_list)

            len_list = [
                len_persona_name_list,
                len_Incentive_list,
                len_Call_to_Action_list,
                len_Personalized_Content_list,
            ]

            set_len_list = list(set(len_list))

            if len(set(set_len_list)) == 1:
                return (
                    persona_name_list,
                    Incentive_list,
                    Call_to_Action_list,
                    Personalized_Content_list,
                )

            if attempt == max_retries - 1:
                break
        return (
            persona_name_list,
            Incentive_list,
            Call_to_Action_list,
            Personalized_Content_list,
        )

    async def store_personalization_data_type(
        self, doc_id: str, personalization_data_type: str
    ) -> bool:
        """
        Update an existing direct mail personalization document to add the personalization_data_type field

        Args:
            doc_id: The ID of the direct mail personalization document
            personalization_data_type: The personalization data type to store

        Returns:
            True if successful, False otherwise
        """
        try:
            # Access the directmail_personalizations collection
            collection = self.db._db.directmail_personalizations

            # Convert string ID to ObjectId
            try:
                object_id = ObjectId(doc_id)
            except Exception as e:
                logger.error(f"Invalid ObjectId format: {doc_id}, error: {str(e)}")
                return False

            # Update the document
            result = await collection.update_one(
                {"_id": object_id},
                {"$set": {"personalization_data_type": personalization_data_type}},
            )

            if result.modified_count > 0:
                logger.info(
                    f"Updated direct mail personalization {doc_id} with personalization_data_type={personalization_data_type}"
                )
                return True
            else:
                logger.warning(f"No documents were updated for ID {doc_id}")
                return False

        except Exception as e:
            logger.error(
                f"Error updating direct mail personalization with personalization data type: {str(e)}"
            )
            return False

    async def save_audience_directmail_template(
        self,
        user_id: str,
        template_data: Dict[str, Any],
        problem_statement: str,
        csv_analysis: Dict[str, Any],
        audience_id: str,
        session_token: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        """
        Save the generated audience-based direct mail template to the database

        Args:
            user_id: The ID of the user
            template_data: The generated template data
            problem_statement: The problem statement/campaign goal
            csv_analysis: The CSV analysis results
            audience_id: The audience ID
            session_token: Optional session token
            conversation_id: Optional conversation ID

        Returns:
            The ID of the saved template
        """
        # Use direct attribute access to the collection through _db
        collection_name = "audience_directmail_templates"

        try:
            # Create the document
            doc = {
                "user_id": user_id,
                "template_id": template_data.get("template_id", str(uuid.uuid4())),
                "audience_id": audience_id,
                "incentives": template_data.get("incentives", []),
                "call_to_action_options": template_data.get(
                    "call_to_action_options", []
                ),
                "personalized_content": template_data.get("directmail_template", ""),
                "personalization_strategy": template_data.get(
                    "personalization_strategy", {}
                ),
                "csv_analysis": csv_analysis,
                "problem_statement": problem_statement,
                "segmentation_criteria_used": template_data.get(
                    "segmentation_criteria_used", []
                ),
                "session_token": session_token,
                "conversation_id": conversation_id,
                "personalization_data_type": "audience data",
                "created_at": {"$date": {"$now": True}},
            }

            # Insert the document using _db syntax
            result = await self.db._db[collection_name].insert_one(doc)
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Error saving audience direct mail template: {str(e)}")
            # If insertion fails, try to create the collection first and then insert
            try:
                # Ensure the collection exists
                required_collections = await self.db._db.list_collection_names()
                if collection_name not in required_collections:
                    await self.db._db.create_collection(collection_name)
                    logger.info(f"Created collection: {collection_name}")

                # Try to insert again
                result = await self.db._db[collection_name].insert_one(doc)
                return str(result.inserted_id)

            except Exception as inner_e:
                logger.error(
                    f"Failed to create collection and save template: {str(inner_e)}"
                )
                # Return a placeholder ID if we can't save to the database
                return str(uuid.uuid4())

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
            user_id = parameters.get("user_id")  # required
            problem_statement = parameters.get("problem_statement")  # required
            personalization_data_type = parameters.get(
                "personalization_data_type"
            )  # required
            model_id = parameters.get(
                "model_id"
            )  # required for persona data, optional for audience data
            persona_name = parameters.get(
                "persona_name"
            )  # required for persona data, optional for audience data
            audience_id = parameters.get(
                "audience_id"
            )  # required for audience data, optional for persona data
            brand_voice = parameters.get(
                "brand_voice", "professional and engaging"
            )  # optional
            session_token = parameters.get("session_token")  # optional
            conversation_id = parameters.get("conversation_id")  # optional

            # Log the parameters
            logger.info(
                f"Executing DirectMailPersonalizationTool with parameters: user_id={user_id}, personalization_data_type={personalization_data_type}"
            )

            # Validate required parameters
            missing_params = []
            if not user_id:
                missing_params.append("user_id")

            # Add explicit debug logging for problem statement
            print("problem_statement value:", problem_statement)
            print("problem_statement type:", type(problem_statement))

            # Check if problem_statement is None, empty string, or just whitespace
            if problem_statement is None or (
                isinstance(problem_statement, str) and problem_statement.strip() == ""
            ):
                missing_params.append("problem_statement")

            # Check personalization_data_type
            if not personalization_data_type:
                missing_params.append("personalization_data_type")

            # Add explicit debug logging
            print("personalization_data_type:", personalization_data_type)
            print("model_id:", model_id)
            print("persona_name:", persona_name)
            print("audience_id:", audience_id)
            print("problem_statement:", problem_statement)

            # Determine required parameters based on personalization_data_type
            if (
                personalization_data_type
                and personalization_data_type.lower() == "persona data"
            ):
                # For persona data, model_id and persona_name are required, audience_id is optional
                if not model_id:
                    missing_params.append("model_id")

                # Check if persona_name is None, empty string, or just whitespace - but allow "all" as a valid value
                # if persona_name is None or (isinstance(persona_name, str) and persona_name.strip() == "" and persona_name != "all"):
                #     missing_params.append("persona_name")

            elif (
                personalization_data_type
                and personalization_data_type.lower() == "audience data"
            ):
                # For audience data, audience_id is required, model_id and persona_name are optional
                if not audience_id:
                    missing_params.append("audience_id")

            print("missing_params:", missing_params)

            if missing_params:
                # Create a conversational response based on what's missing
                response_message = ""
                conversation_stage = ""

                if "personalization_data_type" in missing_params:
                    response_message = "To create personalized direct mail content, I need to know whether you want to use persona data or audience data. Please specify your preference."
                    conversation_stage = "selecting_personalization_type"
                elif (
                    personalization_data_type
                    and personalization_data_type.lower() == "persona data"
                    and "model_id" in missing_params
                ):
                    response_message = "I'm excited to create personalized direct mail content using persona data! To get started, could you please share which model ID you'd like to use? This helps me find the right personas for your direct mail campaign."
                    conversation_stage = "collecting_model_id"
                elif (
                    personalization_data_type
                    and personalization_data_type.lower() == "audience data"
                    and "audience_id" in missing_params
                ):
                    response_message = "I'm excited to create personalized direct mail content using audience data! To get started, could you please share which audience ID you'd like to use?"
                    conversation_stage = "collecting_audience_id"
                elif "problem_statement" in missing_params:
                    response_message = "I need to understand what your direct mail campaign is about. Please provide a brief problem statement or campaign goal. For example: 'We're launching a new fitness product targeting health-conscious adults' or 'We need to boost summer sales for our outdoor furniture line.'"
                    conversation_stage = "collecting_problem_statement"
                    return ToolResponse(
                        status="input_required",
                        message=response_message,
                        required_inputs=["problem_statement"],
                        data={
                            "conversation_stage": conversation_stage,
                            "frontend_action": "show_problem_statement_popup",
                        },
                    )
                # elif personalization_data_type and personalization_data_type.lower() == "persona data" and "persona_name" in missing_params:
                #     response_message = "Perfect! Now, which persona would you like this direct mail to target? I can help you select from your available personas or create direct mail content for all personas."
                #     conversation_stage = "collecting_persona_name"
                else:
                    response_message = (
                        "To create personalized direct mail content, I need a bit more information. "
                        + ", ".join(missing_params)
                        + " would be helpful."
                    )
                    conversation_stage = "collecting_parameters"

                return ToolResponse(
                    status="input_required",
                    message=response_message,
                    required_inputs=missing_params,
                    data={"conversation_stage": conversation_stage},
                )
            else:
                # Handle based on personalization_data_type
                if personalization_data_type.lower() == "persona data":
                    # Use persona-based personalization
                    persona_summary_list = await self.db.get_persona_summaries(
                        user_id=str(user_id),
                        model_id=str(model_id),
                        conversation_id=str(conversation_id),
                        conversation_status=True,
                    )

                    if not persona_summary_list:
                        logger.info(
                            f"No persona summaries found for user_id={user_id}, model_id={model_id}"
                        )
                        return ToolResponse(
                            status="warning_persona_summary",
                            message=f"I notice you'd like to create personalized direct mail content using persona data, but I don't see any existing personas for the model ID: {model_id}. Before we can create effective direct mail personalization, we'll need to build some persona narratives first. Would you like me to help you create personas for this model ID?",
                            data={
                                "conversation_stage": "suggesting_persona_creation",
                                "suggested_next_message": "Would you like me to help you create persona narratives for this model ID?",
                            },
                        )

                    logger.info(f"Found {len(persona_summary_list)} persona summaries")

                    # If we made it here but persona_name is empty, we should return an error
                    # (this is a failsafe in case the earlier validation is bypassed)
                    if persona_name is None or (
                        isinstance(persona_name, str)
                        and persona_name.strip() == ""
                        and persona_name != "all"
                    ):
                        return ToolResponse(
                            status="input_required",
                            message="Please provide a persona name for creating direct mail personalization. Available personas: "
                            + ", ".join(
                                [
                                    p.get("persona_name", "unknown")
                                    for p in persona_summary_list
                                ]
                            ),
                            required_inputs=["persona_name"],
                            data={"conversation_stage": "collecting_persona_name"},
                        )

                    result = {}
                    direct_mail_personalization = []

                    # Find the matching persona or process all if 'all' is specified
                    matching_personas = []
                    for persona in persona_summary_list:
                        name = persona.get("persona_name")
                        summary = persona.get("data")

                        if not name or not summary:
                            logger.warning(
                                f"Skipping persona with missing name or summary: {persona}"
                            )
                            continue

                        if persona_name.lower() == "all":
                            matching_personas.append({"name": name, "summary": summary})
                        elif name == persona_name:
                            matching_personas.append({"name": name, "summary": summary})

                    if not matching_personas:
                        logger.warning(
                            f"No matching personas found for persona_name={persona_name}"
                        )
                        return ToolResponse(
                            status="error",
                            message=f"No persona found with name '{persona_name}'. Available personas: {', '.join([p.get('persona_name', 'unknown') for p in persona_summary_list])}",
                            data=None,
                        )

                    logger.info(
                        f"Processing {len(matching_personas)} matching personas"
                    )

                    # Process each matching persona
                    for persona_data in matching_personas:
                        name = persona_data["name"]
                        summary = persona_data["summary"]

                        try:
                            logger.info(
                                f"Generating direct mail personalization for persona: {name}"
                            )
                            (
                                persona_name_list,
                                Incentive_list,
                                Call_to_Action_list,
                                Personalized_Content_list,
                            ) = self.direct_mail_llm(
                                problem_statement,
                                summary,
                                name,
                                conversation_id,
                                user_id,
                            )

                            # Validate results
                            if (
                                not persona_name_list
                                or not Incentive_list
                                or not Call_to_Action_list
                                or not Personalized_Content_list
                            ):
                                logger.warning(f"Incomplete results for persona {name}")
                                continue

                            for incentive, call_to_action, personalized_content in zip(
                                Incentive_list,
                                Call_to_Action_list,
                                Personalized_Content_list,
                            ):
                                direct_mail_personalization.append(
                                    {
                                        "persona_name": name,
                                        "incentive": incentive,
                                        "call_to_action": call_to_action,
                                        "personalized_content": personalized_content,
                                    }
                                )
                        except Exception as e:
                            logger.error(
                                f"Error processing persona {name}: {str(e)}",
                                exc_info=True,
                            )
                            # Continue with other personas if one fails
                            continue

                    # Check if we have any direct mail personalization data
                    if not direct_mail_personalization:
                        return ToolResponse(
                            status="error",
                            message="Failed to generate direct mail personalization. Please try again with a different persona or problem statement.",
                            data=None,
                        )

                    # Save direct mail personalizations to the database
                    saved_ids = []
                    for ep in direct_mail_personalization:
                        try:
                            doc_id = await self.db.save_direct_mail_personalization(
                                user_id=user_id,
                                model_id=model_id,
                                persona_name=ep.get("persona_name"),
                                incentive=ep.get("incentive"),
                                call_to_action=ep.get("call_to_action"),
                                personalized_content=ep.get("personalized_content"),
                                problem_statement=problem_statement,
                                session_token=session_token,
                                conversation_id=conversation_id,
                            )
                            saved_ids.append(doc_id)
                        except Exception as e:
                            logger.error(
                                f"Error saving direct mail personalization: {str(e)}",
                                exc_info=True,
                            )

                    # Try to update each document with the personalization_data_type
                    for doc_id in saved_ids:
                        try:
                            await self.store_personalization_data_type(
                                doc_id, "persona data"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not store personalization_data_type for document {doc_id}: {str(e)}"
                            )
                            # Continue execution even if this fails - it's just metadata

                    # Save tool execution record
                    try:
                        await self.db.save_tool_execution(
                            tool_name=self.get_name(),
                            parameters={
                                "user_id": user_id,
                                "model_id": model_id,
                                "personalization_data_type": "persona data",
                            },
                            result={
                                "direct_mail_personalization": direct_mail_personalization
                            },
                            user_id=user_id,
                            conversation_id=conversation_id,
                        )
                    except Exception as e:
                        logger.error(
                            f"Error saving tool execution: {str(e)}", exc_info=True
                        )

                    return ToolResponse(
                        status="success",
                        message="Great news! I've successfully created personalized direct mail content based on your persona data. The content includes targeted incentives, compelling calls-to-action, and personalized messaging for your selected persona. You can now use this to craft high-converting direct mail campaigns that really connect with your audience.",
                        data={
                            "direct_mail_personalization": direct_mail_personalization,
                            "conversation_stage": "showing_results",
                            "suggested_next_message": "Would you like to create personalization for another marketing channel like email or digital ads?",
                        },
                    )

                elif personalization_data_type.lower() == "audience data":
                    all_filter_criteria = []

                    # Attempt to retrieve filter criteria from generalized_personalization collection
                    try:
                        logger.info(
                            f"Retrieving filter criteria from generalized_personalization for audience ID: {audience_id}"
                        )
                        generalized_personalization_data = (
                            await self.get_generalized_personalization(
                                user_id, audience_id
                            )
                        )

                        if (
                            generalized_personalization_data
                            and len(generalized_personalization_data) > 0
                        ):
                            # Use the most recent entry (should be sorted by created_at in descending order)
                            latest_data = generalized_personalization_data[0]
                            all_filter_criteria = latest_data.get("filter_criteria", [])
                            logger.info(
                                f"Found {len(all_filter_criteria)} filter criteria in database"
                            )
                        else:
                            logger.info(
                                "No generalized personalization data found in database"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error retrieving generalized personalization data: {str(e)}"
                        )
                        # Continue with an empty list if there's an error

                    print(
                        "all_filter_criteria-----------------------------------",
                        all_filter_criteria,
                    )

                    if not all_filter_criteria:
                        return ToolResponse(
                            status="input_required",
                            message="No persona found for the audience data, would you like to create persona for this audience?",
                            data={},
                        )

                    # Step 7: Create a generalized direct mail template for the audience
                    logger.info("Creating direct mail template based on audience data")
                    # Modify the create_email_template_from_csv function for direct mail
                    directmail_template = create_genralize_directemail_template(
                        campaign_context=all_filter_criteria,
                        openai_key=OPENAI_API_KEY,
                        problem_statement=problem_statement,
                    )

                    # csv_analysis = {
                    #     "total_rows": filtered_results.get("original_row_count", 0),
                    #     "total_columns": len(column_data),
                    #     "column_names": [col.get("column", "") for col in column_data]
                    # }
                    csv_analysis = {}

                    # directmail_template["csv_analysis"] = csv_analysis
                    # directmail_template["segmentation_criteria_used"] = all_filter_criteria
                    # directmail_template["template_id"] = str(uuid.uuid4())

                    # Save the template to the database
                    template_id = await self.save_audience_directmail_template(
                        user_id=user_id,
                        template_data=directmail_template,
                        problem_statement=problem_statement,
                        csv_analysis=csv_analysis,
                        audience_id=audience_id,
                        session_token=session_token,
                        conversation_id=conversation_id,
                    )

                    # Add the saved template ID to the result
                    directmail_template["database_id"] = template_id

                    # Save tool execution record
                    await self.db.save_tool_execution(
                        tool_name=self.get_name(),
                        parameters={
                            "user_id": user_id,
                            "problem_statement": problem_statement,
                            "audience_id": audience_id,
                            "personalization_data_type": "audience data",
                        },
                        result={"template_id": directmail_template.get("template_id")},
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )

                    # Create a rich response for the user
                    template_preview = directmail_template.get(
                        "directmail_template", ""
                    )
                    # Truncate if too long for display
                    if len(template_preview) > 300:
                        template_preview = template_preview[:297] + "..."

                    # Format sample headlines with placeholders
                    headlines = directmail_template.get("headlines", [])
                    sample_headline = (
                        headlines[0] if headlines else "Special Offer Just For You!"
                    )

                    message = f"""
                I've created a personalized direct mail template based on your audience data."""

                    # Create direct mail personalization format to match the persona-based output format
                    direct_mail_personalization = [
                        {
                            "audience_id": audience_id,
                            "incentives": directmail_template.get("incentives", []),
                            "headlines": directmail_template.get("headlines", []),
                            "call_to_action_options": directmail_template.get(
                                "call_to_action_options", []
                            ),
                            "personalized_content": directmail_template.get(
                                "directmail_template", ""
                            ),
                            "personalization_strategy": directmail_template.get(
                                "personalization_strategy", {}
                            ),
                        }
                    ]

                    return ToolResponse(
                        status="success",
                        message=message,
                        data={
                            "audience_directmail_template": directmail_template,
                        },
                    )
                else:
                    return ToolResponse(
                        status="error",
                        message=f"Unsupported personalization data type: {personalization_data_type}. Please use either 'persona data' or 'audience data'.",
                        data=None,
                    )

        except Exception as e:
            logger.error(
                f"Error executing DirectMailPersonalizationTool: {str(e)}",
                exc_info=True,
            )
            return ToolResponse(
                status="error",
                message=f"Error creating direct mail personalization: {str(e)}",
                data=None,
            )

    def _get_base_price(self) -> float:
        """
        Get the base price for the tool

        Returns:
            The base price as a float
        """
        return 3.0

    async def get_generalized_personalization(
        self, user_id: str, audience_id: Optional[str] = None
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
            return await self.db.get_generalized_personalization(
                user_id=user_id, audience_id=audience_id
            )
        except Exception as e:
            logger.error(f"Error in get_generalized_personalization: {str(e)}")
            return []

    def _get_pricing_unit(self) -> str:
        """
        Get the pricing unit for the tool

        Returns:
            The pricing unit (e.g., "per request", "per token", etc.)
        """
        return "per direct mail set"

    def _get_additional_fees(self) -> Dict[str, Any]:
        """
        Get any additional fees for the tool

        Returns:
            A dictionary of additional fees
        """
        return {
            "premium_template_fee": {
                "amount": 1.5,
                "description": "Additional fee for premium direct mail templates",
            }
        }
