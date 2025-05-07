import asyncio
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from app.models.model import ToolResponse
from app.tools.base_tool import BaseTool
from app.utils.audience_utils import (
    analyze_csv_columns,
    create_email_template_from_csv,
    create_genralize_email_template,
    filter_csv_with_segments,
    find_audience_by_id,
    process_data_in_batches,
    structured_data_tool,
)
from app.utils.db import MongoDB

# Load environment variables from .env file
load_dotenv()


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("openai_key")
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key not found in environment variables")

prompt_column_selection = """ 
        You are an AI assistant tasked with analyzing a marketing problem statement and a list of data columns to determine which columns are most relevant for improving audience engagement in marketing campaigns. Your goal is to select columns that would be most useful for email, digital ad, and direct mail campaigns.

        Here is the problem statement you need to analyze:
        <problem_statement>
        {problem_statement}
        </problem_statement>

        Here is the list of available columns:
        <column_list>
        {column_list}
        </column_list>

        Please follow these steps:

        1. Carefully read and understand the problem statement.
        2. Review the list of available columns.
        3. Analyze how each column might contribute to improving audience engagement in email, digital ad, and direct mail campaigns.
        4. Select the columns that you believe are most relevant and useful for these marketing campaigns based on the problem statement.
        5. Ensure that your selection covers a range of data points that could enhance targeting, personalization, and overall campaign effectiveness.

        After your analysis, provide your output in the following JSON format:

        {{"column_name": ["col1", "col2", ...., "coln"],
        "explanation": [<explanation for each column selection in one line and in list format>]}}

        Replace "col1", "col2", etc., with the actual names of the columns you've selected as most relevant. Include only the column names in your JSON output, without any additional explanation or justification.

        Remember to focus on columns that would be most beneficial for email, digital ad, and direct mail campaigns, keeping in mind the goal of improving audience engagement as described in the problem statement.
        """

prompt_summarize_persona = """
You are tasked with creating a summarized persona narrative based on provided JSON data. This narrative should be tailored to address a specific problem statement if one is given. If no problem statement is provided, you'll create a general persona narrative.

First, carefully review the following persona JSON data:

<persona_json>
{persona_json}
</persona_json>

Now, analyze the problem statement (if provided):

<problem_statement>
{problem_statement}
</problem_statement>

If no problem statement is given, skip this step and proceed to create a general persona narrative.

Next, summarize the key information from the persona JSON data. Focus on demographic details, behaviors, preferences, and any other relevant characteristics that define the persona named {persona_name}.

Using this summary and your analysis of the problem statement (if provided), create an exclusive persona narrative. This narrative should:

1. Provide a concise overview of {persona_name}
2. Highlight aspects of the persona that are most relevant to the problem statement (if given)
3. Include specific details that make the persona feel real and relatable
4. Be written in a cohesive, story-like format

Your output should be a single, comprehensive persona narrative. If a problem statement was provided, ensure that the narrative specifically addresses how this persona relates to or is affected by the stated problem.

Give output in below JSON format:
{{
"persona_summary": "<persona summary in the string format>"
}}

Remember to tailor the narrative to the problem statement if one is provided. If no problem statement is given, focus on creating a well-rounded description of the persona based on the JSON data.
"""


class ColumnSelection(BaseModel):
    column_name: List[str] = Field(description="list of relevant columns")
    explanation: List[str] = Field(
        description="explanation for each column selection in one line and in list format"
    )


class PersonaSummary(BaseModel):
    persona_summary: str = Field("persona summary in the string format")


class PersonaSummaryTool(BaseTool):
    """
    Tool for creating persona narratives based on user data

    This tool implements the functionality from PersonaSummarizeView in the original
    views.py file. It processes the model data and creates detailed persona narratives
    that can later be used for content personalization.
    """

    def get_json(self, modelID, session_token, userID) -> str:
        url = "https://staging-api.boostt.ai/api/cdm/model/get"
        payload = json.dumps(
            {"modelID": modelID, "session_token": session_token, "userID": userID}
        )
        headers = {"Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json()

    def column_selection(self, problem_statement, column_list):
        """
        This function uses all the provided columns and based on a provided problem statement LLM decide which columns to filter further.
        :return:
        """

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

        parser = JsonOutputParser(pydantic_object=ColumnSelection)

        prompt = PromptTemplate(
            template=prompt_column_selection,
            input_variables=["problem_statement", "column_list"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser

        output = chain.invoke(
            {"problem_statement": problem_statement, "column_list": column_list}
        )

        return output

    def persona_summary(self, problem_statement, persona_name, columns_data_str):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
        parser = JsonOutputParser(pydantic_object=PersonaSummary)
        prompt = PromptTemplate(
            template=prompt_summarize_persona,
            input_variables=["problem_statement", "persona_name", "persona_json"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm | parser
        output = chain.invoke(
            {
                "problem_statement": problem_statement,
                "persona_name": persona_name,
                "persona_json": columns_data_str,
            }
        )

        return output

    def get_name(self) -> str:
        return "persona_summary"

    def get_description(self) -> str:
        return "Creates detailed persona narratives based on user data"

    def get_required_params(self) -> List[str]:
        return ["user_id", "model_id"]

    def get_optional_params(self) -> List[str]:
        return ["session_token", "conversation_id", "action"]

    def get_model_name_by_id(self, data, model_id):
        """
        Get the model name from the model ID

        Args:
            data: The list of models
            model_id: The model ID to find

        Returns:
            The model name if found, otherwise None
        """
        for item in data:
            if item.get("modelID") == model_id:
                return item.get("name")
        return None

    def fetch_model_list(self, session_token, user_id, limit=12, page=1, status=""):
        """
        Fetch the list of models from the API

        Args:
            session_token: The session token for authentication
            user_id: The user ID
            limit: The number of models to return per page
            page: The page number
            status: The model status filter

        Returns:
            The API response as a dictionary or list
        """
        url = "https://staging-api.boostt.ai/api/cdm/model/list"
        payload = {
            "limit": limit,
            "page": page,
            "session_token": session_token,
            "status": status,
            "userID": user_id,
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raises an error for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []

    async def execute(self, parameters: Dict[str, Any]) -> ToolResponse:
        """
        Execute the tool with the given parameters

        Args:
            parameters: The parameters for the tool

        Returns:
            The response from the tool
        """
        # try:
        if True:
            # Extract parameters
            user_id = parameters.get("user_id")  # required
            model_id = parameters.get("model_id")  # required
            audience_id = parameters.get("audience_id")  # optional
            session_token = parameters.get("session_token")  # optional
            conversation_id = parameters.get("conversation_id")  # optional
            conversation_stage = parameters.get(
                "conversation_stage", "creating_persona"
            )
            recreate_personas = parameters.get(
                "recreate_personas", False
            )  # Flag to indicate if we should recreate existing personas
            personalization_data_type = parameters.get(
                "personalization_data_type", ""
            ).lower()
            action = parameters.get(
                "action", ""
            )  # Action parameter to determine operation

            # Handle request to get generalized personalization data
            if action == "get_generalized_personalization":
                logger.info(
                    f"Retrieving generalized personalization data for user_id={user_id}"
                )

                # Get generalized personalization data from the database
                generalized_data = await self.get_generalized_personalizations(
                    user_id=user_id,
                    audience_id=audience_id,
                    conversation_id=conversation_id,
                )

                if not generalized_data:
                    return ToolResponse(
                        status="not_found",
                        message="No generalized personalization data found. Please create personalization data first.",
                        data={},
                    )

                # Format the response message
                formatted_message = "### Generalized Personalization Criteria\n\n"
                for idx, item in enumerate(generalized_data, 1):
                    criteria_list = item.get("filter_criteria", [])
                    audience = item.get("audience_id", "Unknown")
                    formatted_message += f"#### Audience ID: {audience}\n"

                    if criteria_list:
                        formatted_message += "Criteria:\n"
                        for i, criteria in enumerate(criteria_list, 1):
                            formatted_message += f"{i}. {criteria}\n"
                    else:
                        formatted_message += "No criteria found.\n"

                    formatted_message += (
                        f"Created: {item.get('created_at', 'Unknown')}\n\n"
                    )

                return ToolResponse(
                    status="success",
                    message=formatted_message,
                    data={"generalized_personalization": generalized_data},
                )

            if not personalization_data_type:
                return ToolResponse(
                    status="input_required",
                    message="Do you want to create a persona narrative for audience data or persona data?",
                    required_inputs=[],
                    data={},
                )

            print(
                "conversation_id conversation_id",
                conversation_id,
                personalization_data_type,
                recreate_personas,
                model_id,
                user_id,
            )
            check = None
            if personalization_data_type == "persona data":
                check = model_id
            elif personalization_data_type == "audience data":
                check = audience_id

            # Validate required parameters
            if not user_id or not check:
                missing = []
                if not user_id:
                    missing.append("user_id")
                if not model_id and personalization_data_type == "persona data":
                    missing.append("model_id")
                if not audience_id and personalization_data_type == "audience data":
                    missing.append("audience_id")

                # Provide a more conversational response based on what's missing
                message = ""
                if "model_id" in missing:
                    message = "To get started, could you please share which model you'd like to use? "
                elif "user_id" in missing:
                    message = (
                        "Could you please provide your user ID so I can save your work?"
                    )
                elif "audience_id" in missing:
                    message = "To get started, could you please share which Audiance you'd like to use?"
                else:
                    message = (
                        "To create a persona narrative, I need a bit more information. "
                        + ", ".join(missing)
                        + " would be helpful."
                    )

                return ToolResponse(
                    status="input_required",
                    message=message,
                    required_inputs=missing,
                    data={},
                )

            # Check if personas already exist for this user and model
            # if not recreate_personas:
            #     # Import MongoDB here to avoid circular imports
            #     from app.utils.db import MongoDB
            #     db = MongoDB()

            #     try:
            #         persona_summaries = await db.get_persona_summaries(
            #             user_id=str(user_id),
            #             model_id=str(model_id),
            #             conversation_id=str(conversation_id),
            #             conversation_status=True
            #         )

            #         if persona_summaries and len(persona_summaries) > 0:
            #             # Personas already exist
            #             return ToolResponse(
            #                 status="exsist",
            #                 message="I found existing persona narratives for this model ID.",
            #                 data={

            #                 }
            #             )
            #     except Exception as e:
            #         print(f"Error checking for existing personas: {str(e)}")
            #         # Continue with creation if there's an error checking existing personas
            # else:
            #     pass
            # User wants to recreate personas - delete existing ones first
            # try:
            #     # Import MongoDB here to avoid circular imports
            #     from app.utils.db import MongoDB
            #     db = MongoDB()

            #     # Delete existing personas for this user and model
            #     await db.delete_persona_summaries(user_id=str(user_id), model_id=str(model_id))
            #     logger.info(f"Deleted existing personas for user_id={user_id}, model_id={model_id}")
            # except Exception as e:
            #     logger.error(f"Error deleting existing personas: {str(e)}")
            # Continue with creation even if deletion fails
            print("personalization_data_type", personalization_data_type)
            if personalization_data_type == "persona data":
                db = MongoDB()

                persona_summaries = await db.get_persona_summaries(
                    user_id=str(user_id),
                    model_id=str(model_id),
                    conversation_id=str(conversation_id),
                    conversation_status=True,
                )

                if persona_summaries and len(persona_summaries) > 0:
                    # Personas already exist
                    persona_dict = {
                        entry["persona_name"]: entry["data"]
                        for entry in persona_summaries
                    }
                    message = f"I found existing persona narratives for this model: {model_id}"
                    message = self.replace_model_ids_with_names(
                        message, session_token, model_id, user_id
                    )
                    return ToolResponse(
                        status="exsist",
                        message=message,
                        data={"persona_summaries": persona_dict},
                    )

                logger.info(
                    f"Creating persona narratives for user_id={user_id}, model_id={model_id}"
                )

                # Initialize MongoDB
                db = MongoDB()

                # Process the model data to generate persona summaries
                output_persona_json = await self._filter_json(
                    model_id, session_token, user_id
                )

                # Check if we received valid persona data
                if not output_persona_json:
                    return ToolResponse(
                        status="error",
                        message="Could not generate persona narratives. The model may not contain sufficient data.",
                        data=None,
                    )

                logger.info(f"Generated {len(output_persona_json)} persona narratives")

                # Format data for storage and response
                person_summary_data = []
                created_at = time.time()

                # Save to database
                await db.save_persona_summary(
                    user_id=user_id,
                    model_id=model_id,
                    output_persona_json=output_persona_json,
                    session_token=session_token,
                    is_summary=True,
                    conversation_id=conversation_id,
                )

                # Store each persona in a separate database record
                for persona_name, summary in output_persona_json.items():
                    # Format for response
                    person_summary_data.append(
                        {
                            "role": "assistant",
                            "content": f"### {persona_name}\n{summary}",
                        }
                    )

                # Save tool execution record
                await db.save_tool_execution(
                    tool_name=self.get_name(),
                    parameters=parameters,
                    result={"personas": output_persona_json},
                    user_id=user_id,
                )

                # Format the response for the client
                formatted_message = "### Persona Narratives\n\n"
                for persona_name, summary in output_persona_json.items():
                    formatted_message += f"#### {persona_name}\n"
                    formatted_message += f"{summary}\n\n"

                simplified_message = f"I've successfully created the persona narrative"
                if persona_name:
                    simplified_message += f" for '{persona_name}'"

                # Replace model ID with name if session token is available
                display_model = model_id

                simplified_message += f" using model {display_model}."
                simplified_message += " Would you like to create personalized content using this persona now?"

                formatted_message = self.replace_model_ids_with_names(
                    simplified_message, session_token, model_id, user_id
                )
                return ToolResponse(
                    status="success",
                    message=formatted_message,
                    data={
                        "personas": output_persona_json,
                        "conversation_data": person_summary_data,
                    },
                )
            if personalization_data_type == "audience data":
                # Use audience-based personalization logic from csv_email_template_tool
                logger.info(
                    f"Creating email personalization using audience data with ID: {audience_id}"
                )

                # Step 1: Find audience data using find_audience_by_id
                logger.info(f"Finding audience data for target ID: {audience_id}")
                USER_ID_PERSONA = user_id
                SESSION_TOKEN_PERSONA = session_token
                TARGET_ID = audience_id

                audience_data = find_audience_by_id(
                    user_id=USER_ID_PERSONA,
                    SESSION_TOKEN=SESSION_TOKEN_PERSONA,
                    target_id=TARGET_ID,
                )
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
                        message=f"Could not find audience data for ID: {audience_id}. Please verify that this is a valid audience ID.",
                        data=None,
                    )

                # Step 2: Analyze CSV columns
                logger.info("Analyzing CSV columns")
                column_data = analyze_csv_columns(
                    file_path, [], is_propensity_data=False, rows="All", seed=42
                )
                logger.info(
                    f"Column analysis complete. Found {len(column_data)} columns."
                )

                # Step 3: Process data in batches
                logger.info("Processing data in batches")
                data_batches = process_data_in_batches(column_data)
                logger.info("Batch processing complete")

                # Create explanations for columns based on the actual columns in the data
                column_explanations = []
                for col in column_data:
                    col_name = col.get("column", "")
                    data_type = col.get("data_type", "unknown")
                    column_explanations.append(
                        f"{col_name}: {col_name.replace('_', ' ').title()} ({data_type})"
                    )

                # Step 4: Generate filter data using structured_data_tool
                logger.info("Generating filter data")
                filter_data = structured_data_tool(
                    openai_key=OPENAI_API_KEY,
                    audiance_data_dict=data_batches,
                    problem_statement="genrate insights from the data for the marketing campaign",
                    additional_requirements="Focus on creating segments that will be useful for a targeted email campaign",
                    explanation=column_explanations,
                    other_requirements="Ensure segments are meaningful for creating an effective personalized email campaign",
                )
                logger.info("Filter data generation complete")

                # Step 5: Filter CSV with segments
                logger.info("Filtering CSV with segments")
                filtered_results = filter_csv_with_segments(
                    filter_data=filter_data,
                    csv_file_path=file_path,
                    return_dataframe=True,
                )
                logger.info(
                    f"Filtering complete. Found {len(filtered_results.get('segments', []))} segments."
                )

                print(
                    "filtered_results-----------------------------------",
                    filtered_results,
                )

                # Step 6: Extract campaign context from filter data
                all_filter_criteria = self._extract_filter_criteria(filter_data)

                # Create the collection and store the data
                db = MongoDB()
                await db.save_generalized_personalization(
                    user_id=user_id,
                    audience_id=audience_id,
                    filter_criteria=all_filter_criteria,
                    conversation_id=conversation_id,
                )

                return ToolResponse(
                    status="success",
                    message="Successfully created and stored generalized personalization data.",
                    data={"filter_criteria": all_filter_criteria},
                )
        # except Exception as e:
        #     logger.error(f"Error executing PersonaSummaryTool: {str(e)}")
        #     traceback.print_exc()
        #     return ToolResponse(
        #         status="error",
        #         message=f"Error creating persona narratives: {str(e)}",
        #         data=None
        #     )

    def replace_model_ids_with_names(self, text, session_token, model_id, user_id):
        """
        Replace model IDs in text with their corresponding model names.
        Uses regex to find potential model IDs and replaces them with model names.

        Args:
            text (str): The text containing model IDs
            session_token (str): Session token for API authentication
            user_id (str): User ID for API authentication

        Returns:
            str: Text with model IDs replaced by model names
        """
        try:
            if not text:
                return text

            response = self.fetch_model_list(
                session_token=session_token, user_id=user_id, model_id=model_id
            )

            try:
                model_name = response["name"]
            except:
                model_name = ""

            if model_name:
                text = text.replace(model_id, f"{model_name}")
        except:
            return text
        return text

    async def _filter_json(
        self, model_id: str, session_token: Optional[str], user_id: str
    ) -> Dict[str, str]:
        """
        Process the model data and extract persona narratives

        This is adapted from persona_summary_utils.filter_json in the original code

        Args:
            model_id: The model ID to process
            session_token: Optional session token
            user_id: The user ID

        Returns:
            Dictionary of persona names to summaries
        """
        try:
            logger.info(f"Processing model {model_id} for user {user_id}")

            persona_summary_dict = {}

            data = self.get_json(model_id, session_token, user_id)

            print("data", data)
            print(model_id, session_token, user_id, "&&&&&&&&&&&&&&&&&&&&")

            for i in data["report"]["personas"]["personas"]:
                persona_name = i["name"]
                print("persona_name", persona_name)
                persona_id_persona = i["id"]
                keys_bin = []
                for j in i["details"]["bins"].items():
                    key, value = j
                    keys_bin.append(key.split("/")[1])

                rem_column_name = []
                for i_ in data["report"]["breakdown"]:
                    persona_id = i_["persona_id"]
                    if persona_id_persona == persona_id:
                        for j in i_["traits"]:
                            col_name = j["trait_name"]
                            col_name = col_name.split("/")[1]
                            # print(keys_bin)
                            if col_name not in keys_bin:
                                rem_column_name.append(col_name)
                column_dict = self.column_selection(
                    problem_statement="", column_list=rem_column_name
                )
                col_filter_list = column_dict["column_name"]

                string_llm = """"""
                for i_ in data["report"]["breakdown"]:
                    persona_id = i_["persona_id"]
                    if persona_id_persona == persona_id:
                        for j in i_["traits"]:
                            col_name = j["trait_name"]
                            col_name = col_name.split("/")[1]
                            if col_name in col_filter_list:
                                df = pd.DataFrame(j["bins"])
                                if "max" in df.columns:
                                    # Merging rows where count is 0 sequentially
                                    merged_data = []
                                    temp_row = None
                                    for _, row in df.iterrows():
                                        if row["count"] == 0:
                                            if temp_row is None:  # Start merging
                                                temp_row = row.copy()
                                            else:  # Update the 'max' while keeping 'min' as is
                                                temp_row["max"] = row["max"]
                                        else:
                                            if (
                                                temp_row is not None
                                            ):  # Add the merged row before moving to non-zero count
                                                merged_data.append(temp_row)
                                                temp_row = None
                                            merged_data.append(row)

                                    # Add the last temp_row if it exists
                                    if temp_row is not None:
                                        merged_data.append(temp_row)

                                    # Create the final DataFrame
                                    df_final = pd.DataFrame(merged_data)
                                    max_list = df_final["max"].tolist()
                                    min_list = df_final["min"].tolist()
                                    count_list = df_final["count"].tolist()
                                    percent_list = df_final["percent"].tolist()
                                    string_llm += f"""column_name: {col_name}, max: {max_list}, min: {min_list}, count: {count_list}, percent: {percent_list} \n"""
                                    # print("string_llm", string_llm)

                                if "value" in df.columns:
                                    # Separate rows where count == 0
                                    zero_count_df = df[df["count"] == 0]

                                    # Combine `value` column into a list for rows where count == 0
                                    merged_row = {
                                        "count": 0,
                                        "value": list(zero_count_df["value"]),
                                        "percent": 0.0,  # Assuming percent is also zero for such rows
                                    }

                                    # Filter out rows where count == 0
                                    df_non_zero = df[df["count"] != 0]

                                    # Add the merged row to the non-zero rows
                                    df_final = pd.concat(
                                        [df_non_zero, pd.DataFrame([merged_row])],
                                        ignore_index=True,
                                    )

                                    value_list = df_final["value"].tolist()
                                    count_list = df_final["count"].tolist()
                                    percent_list = df_final["percent"].tolist()
                                    string_llm += f"""column_name: {col_name}, value: {value_list}, count: {count_list}, percent: {percent_list} \n"""
                output_persona_summary = self.persona_summary(
                    problem_statement="",
                    persona_name=persona_name,
                    columns_data_str=string_llm,
                )
                persona_summary_dict[persona_name] = output_persona_summary[
                    "persona_summary"
                ]
            return persona_summary_dict

        except Exception as e:
            logger.error(f"Error in filter_json: {str(e)}")
            traceback.print_exc()
            return {}

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
        return "per persona set"

    def _get_additional_fees(self) -> Dict[str, Any]:
        """
        Get any additional fees for the tool

        Returns:
            A dictionary of additional fees
        """
        return {
            "complexity_fee": {
                "amount": 2.0,
                "description": "Additional fee for complex persona narratives with detailed backstories",
            }
        }

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
            filter_sets = filter_data["user_upload_filter_columns_result"].get(
                "output", []
            )
        else:
            filter_sets = filter_data.get("output", [])

        # Handle direct nested structure if needed
        if not filter_sets and isinstance(filter_data.get("filter_sets", None), list):
            filter_sets = filter_data.get("filter_sets", [])

        # Extract criteria from filter sets
        for filter_set in filter_sets:
            if (
                isinstance(filter_set, dict)
                and "explanation" in filter_set
                and filter_set["explanation"]
            ):
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
                    column_names = (
                        filter_set[0] if isinstance(filter_set[0], list) else []
                    )
                    if column_names:
                        criteria_desc = f"Criteria based on: {', '.join(str(col) for col in column_names)}"
                        all_criteria.append(criteria_desc)

        return all_criteria

    async def get_generalized_personalizations(
        self,
        user_id: str,
        audience_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve generalized personalization data from the database

        Args:
            user_id: The ID of the user
            audience_id: Optional audience ID filter
            conversation_id: Optional conversation ID filter

        Returns:
            List of generalized personalization documents
        """
        db = MongoDB()
        return await db.get_generalized_personalization(
            user_id=user_id, audience_id=audience_id, conversation_id=conversation_id
        )
